#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <limits>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

// MKL
#include <mkl_blas.h>
#include <mkl_lapack.h>
#include <mkl_service.h>
#include <mkl_version.h>

#ifdef __AVX2__
#include <vectorclass.h>
#include <immintrin.h>
#endif

#include <VapourSynth.h>
#include <VSHelper.h>

#include <config.h>

static VSPlugin * myself = nullptr;

template <typename T>
static inline T square(T const & x) noexcept {
    return x * x;
}

static inline int m16(int x) noexcept {
    assert(x > 0);
    return ((x - 1) / 16 + 1) * 16;
}

namespace {
struct Workspace {
    float * intermediate; // [radius == 0] shape: (2, height, width)
    float * denoising_patch; // shape: (group_size, svd_lda) + pad (simd_lanes - 1)
    float * mean_patch; // [residual] shape: (block_size, block_size) + pad (simd_lanes - 1)
    float * current_patch; // shape: (block_size, block_size) + pad (simd_lanes - 1)
    float * svd_s; // shape: (min(square(block_size), group_size),)
    float * svd_u; // shape: (min(square(block_size), group_size), svd_ldu)
    float * svd_vt; // shape: (min(square(block_size), group_size), svd_ldvt)
    float * svd_work; // shape: (svd_lwork,)
    int * svd_iwork; // shape: (8 * min(square(block_size), group_size),)
    std::vector<std::tuple<float, int, int, int>> * errors; // shape: dynamic
    std::vector<std::tuple<float, int, int>> * center_errors; // shape: dynamic
    std::vector<std::tuple<int, int>> * search_locations; // shape: dynamic
    std::vector<std::tuple<int, int>> * new_locations; // shape: dynamic
    std::vector<std::tuple<int, int>> * locations_copy; // shape: dynamic
    std::vector<std::tuple<float, int, int>> * temporal_errors; // shape: dynamic

    void init(
        int width, int height,
        int block_size, int group_size, int radius,
        bool residual,
        int svd_lda, int svd_ldu, int svd_ldvt, int svd_lwork
    ) noexcept {

#ifdef __AVX2__
        constexpr int pad = 7;
#else
        constexpr int pad = 0;
#endif

        if (residual) {
            mean_patch = vs_aligned_malloc<float>((square(block_size) + pad) * sizeof(float), 64);
        } else {
            mean_patch = nullptr;
        }

        current_patch = vs_aligned_malloc<float>((square(block_size) + pad) * sizeof(float), 64);

        if (radius == 0) {
            intermediate = reinterpret_cast<float *>(std::malloc(2 * height * width * sizeof(float)));
        } else {
            intermediate = nullptr;
        }

        int m = square(block_size);
        int n = group_size;

        denoising_patch = vs_aligned_malloc<float>((svd_lda * n + pad) * sizeof(float), 64);

        svd_s = vs_aligned_malloc<float>(std::min(m, n) * sizeof(float), 64);

        svd_u = vs_aligned_malloc<float>(svd_ldu * std::min(m, n) * sizeof(float), 64);

        svd_vt = vs_aligned_malloc<float>(svd_ldvt * n * sizeof(float), 64);

        svd_work = vs_aligned_malloc<float>(svd_lwork * sizeof(float), 64);

        svd_iwork = vs_aligned_malloc<int>(8 * std::min(m, n) * sizeof(int), 64);

        errors = new std::remove_pointer_t<decltype(errors)>;
        center_errors = new std::remove_pointer_t<decltype(center_errors)>;
        search_locations = new std::remove_pointer_t<decltype(search_locations)>;
        new_locations = new std::remove_pointer_t<decltype(new_locations)>;
        locations_copy = new std::remove_pointer_t<decltype(locations_copy)>;
        temporal_errors = new std::remove_pointer_t<decltype(temporal_errors)>;
    }

    void release() noexcept {
        vs_aligned_free(mean_patch);
        mean_patch = nullptr;

        vs_aligned_free(current_patch);
        current_patch = nullptr;

        std::free(intermediate);
        intermediate = nullptr;

        vs_aligned_free(denoising_patch);
        denoising_patch = nullptr;

        vs_aligned_free(svd_s);
        svd_s = nullptr;

        vs_aligned_free(svd_u);
        svd_u = nullptr;

        vs_aligned_free(svd_vt);
        svd_vt = nullptr;

        vs_aligned_free(svd_work);
        svd_work = nullptr;

        vs_aligned_free(svd_iwork);
        svd_iwork = nullptr;

        delete errors;
        errors = nullptr;

        delete center_errors;
        center_errors = nullptr;

        delete search_locations;
        search_locations = nullptr;

        delete new_locations;
        new_locations = nullptr;

        delete locations_copy;
        locations_copy = nullptr;

        delete temporal_errors;
        temporal_errors = nullptr;
    }
};

struct WNNMData {
    VSNodeRef * node;
    float sigma[3];
    int block_size, block_step, group_size, bm_range;
    int radius, ps_num, ps_range;
    bool process[3];
    bool residual, adaptive_aggregation;
    VSNodeRef * ref_node; // rclip
    int svd_lwork, svd_lda, svd_ldu, svd_ldvt;

    std::unordered_map<std::thread::id, Workspace> workspaces;
    std::shared_mutex workspaces_lock;
};

enum class WnnmInfo { SUCCESS, FAILURE };
} // namespace

#ifdef __AVX2__
static inline Vec8i make_mask(int block_size_m8) noexcept {
    static constexpr int temp[16] {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

    return Vec8i().load(temp + 8 - block_size_m8);
}
#endif

#ifdef __AVX2__
namespace {
enum class BlockSizeInfo { Is8, Mod8, General };

struct Empty {};
}

template <BlockSizeInfo dispatch>
static inline void compute_block_distances_avx2(
    std::vector<std::tuple<float, int, int>> & errors,
    const float * VS_RESTRICT current_patch,
    const float * VS_RESTRICT neighbour_patch,
    int top, int bottom, int left, int right,
    int stride, int block_size
) noexcept {

    if constexpr (dispatch == BlockSizeInfo::Is8) {
        block_size = 8;
    }

    [[maybe_unused]] std::conditional_t<dispatch == BlockSizeInfo::General, Vec8i, Empty> mask;
    if constexpr (dispatch == BlockSizeInfo::General) {
        mask = make_mask(block_size % 8);
    }

    for (int bm_y = top; bm_y <= bottom; ++bm_y) {
        for (int bm_x = left; bm_x <= right; ++bm_x) {
            Vec8f vec_error {0.f};

            const float * VS_RESTRICT current_patchp = current_patch;
            const float * VS_RESTRICT neighbour_patchp = neighbour_patch;

            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                if constexpr (dispatch == BlockSizeInfo::Is8) {
                    Vec8f vec_current = Vec8f().load_a(current_patchp);
                    Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                    Vec8f diff = vec_current - vec_neighbour;
                    vec_error = mul_add(diff, diff, vec_error);

                    current_patchp += 8;
                    neighbour_patchp += stride;
                } else if constexpr (dispatch == BlockSizeInfo::Mod8) {
                    for (int patch_x = 0; patch_x < block_size; patch_x += 8) {
                        Vec8f vec_current = Vec8f().load_a(current_patchp);
                        Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                        Vec8f diff = vec_current - vec_neighbour;
                        vec_error = mul_add(diff, diff, vec_error);

                        current_patchp += 8;
                        neighbour_patchp += 8;
                    }

                    neighbour_patchp += stride - block_size;
                } else if constexpr (dispatch == BlockSizeInfo::General) {
                    for (int patch_x = 0; patch_x < (block_size & (-8)); patch_x += 8) {
                        Vec8f vec_current = Vec8f().load(current_patchp);
                        Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                        Vec8f diff = vec_current - vec_neighbour;
                        vec_error = mul_add(diff, diff, vec_error);

                        current_patchp += 8;
                        neighbour_patchp += 8;
                    }

                    {
                        Vec8f vec_current = _mm256_maskload_ps(current_patchp, mask);
                        Vec8f vec_neighbour = _mm256_maskload_ps(neighbour_patchp, mask);

                        Vec8f diff = vec_current - vec_neighbour;
                        vec_error = mul_add(diff, diff, vec_error);

                        current_patchp += block_size % 8;
                        neighbour_patchp += stride - (block_size & (-8));
                    }
                }
            }

            float error { horizontal_add(vec_error) };

            errors.emplace_back(error, bm_x, bm_y);

            neighbour_patch++;
        }

        neighbour_patch += stride - (right - left + 1);
    }
}

template <BlockSizeInfo dispatch>
static inline void compute_block_distances_avx2(
    std::vector<std::tuple<float, int, int>> & errors,
    const float * VS_RESTRICT current_patch,
    const float * VS_RESTRICT refp,
    const std::vector<std::tuple<int, int>> & search_positions,
    int stride, int block_size
) noexcept {

    if constexpr (dispatch == BlockSizeInfo::Is8) {
        block_size = 8;
    }

    [[maybe_unused]] std::conditional_t<dispatch == BlockSizeInfo::General, Vec8i, Empty> mask;
    if constexpr (dispatch == BlockSizeInfo::General) {
        mask = make_mask(block_size % 8);
    }

    for (const auto & [bm_x, bm_y]: search_positions) {
        Vec8f vec_error {0.f};

        const float * VS_RESTRICT current_patchp = current_patch;
        const float * VS_RESTRICT neighbour_patchp = &refp[bm_y * stride + bm_x];

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            if constexpr (dispatch == BlockSizeInfo::Is8) {
                Vec8f vec_current = Vec8f().load_a(current_patchp);
                Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                Vec8f diff = vec_current - vec_neighbour;
                vec_error = mul_add(diff, diff, vec_error);

                current_patchp += 8;
                neighbour_patchp += stride;
            } else if constexpr (dispatch == BlockSizeInfo::Mod8) {
                for (int patch_x = 0; patch_x < block_size; patch_x += 8) {
                    Vec8f vec_current = Vec8f().load_a(current_patchp);
                    Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                    Vec8f diff = vec_current - vec_neighbour;
                    vec_error = mul_add(diff, diff, vec_error);

                    current_patchp += 8;
                    neighbour_patchp += 8;
                }

                neighbour_patchp += stride - block_size;
            } else if constexpr (dispatch == BlockSizeInfo::General) {
                for (int patch_x = 0; patch_x < (block_size & (-8)); patch_x += 8) {
                    Vec8f vec_current = Vec8f().load(current_patchp);
                    Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                    Vec8f diff = vec_current - vec_neighbour;
                    vec_error = mul_add(diff, diff, vec_error);

                    current_patchp += 8;
                    neighbour_patchp += 8;
                }

                {
                    Vec8f vec_current = _mm256_maskload_ps(current_patchp, mask);
                    Vec8f vec_neighbour = _mm256_maskload_ps(neighbour_patchp, mask);

                    Vec8f diff = vec_current - vec_neighbour;
                    vec_error = mul_add(diff, diff, vec_error);

                    current_patchp += block_size % 8;
                    neighbour_patchp += stride - (block_size & (-8));
                }
            }
        }

        float error { horizontal_add(vec_error) };

        errors.emplace_back(error, bm_x, bm_y);
    }
}
#endif // __AVX2__

static inline void generate_search_locations(
    const std::tuple<float, int, int> * center_positions, int num_center_positions,
    int block_size, int width, int height, int bm_range,
    std::vector<std::tuple<int, int>> & search_locations,
    std::vector<std::tuple<int, int>> & new_locations,
    std::vector<std::tuple<int, int>> & locations_copy
) noexcept {

    search_locations.clear();

    for (int i = 0; i < num_center_positions; i++) {
        const auto & [_, x, y] = center_positions[i];
        int left = std::max(x - bm_range, 0);
        int right = std::min(x + bm_range, width - block_size);
        int top = std::max(y - bm_range, 0);
        int bottom = std::min(y + bm_range, height - block_size);

        new_locations.clear();
        new_locations.reserve((bottom - top + 1) * (right - left + 1));
        for (int j = top; j <= bottom; j++) {
            for (int k = left; k <= right; k++) {
                new_locations.emplace_back(k, j);
            }
        }

        locations_copy = search_locations;

        search_locations.reserve(std::size(search_locations) + std::size(new_locations));

        search_locations.clear();

        std::set_union(
            std::cbegin(locations_copy), std::cend(locations_copy),
            std::cbegin(new_locations), std::cend(new_locations),
            std::back_inserter(search_locations),
            [](const std::tuple<int, int> & a, const std::tuple<int, int> & b) -> bool {
                auto [ax, ay] = a;
                auto [bx, by] = b;
                return ay < by || (ay == by && ax < bx);
            }
        );
    }
}

static inline void compute_block_distances(
    std::vector<std::tuple<float, int, int>> & errors,
    const float * VS_RESTRICT current_patch,
    const float * VS_RESTRICT neighbour_patch,
    int top, int bottom, int left, int right,
    int stride,
    int block_size
) noexcept {

#ifdef __AVX2__
    if (block_size == 8) {
        return compute_block_distances_avx2<BlockSizeInfo::Is8>(errors, current_patch, neighbour_patch, top, bottom, left, right, stride, block_size);
    } else if ((block_size % 8) == 0) {
        return compute_block_distances_avx2<BlockSizeInfo::Mod8>(errors, current_patch, neighbour_patch, top, bottom, left, right, stride, block_size);
    } else {
        return compute_block_distances_avx2<BlockSizeInfo::General>(errors, current_patch, neighbour_patch, top, bottom, left, right, stride, block_size);
    }
#else // __AVX2__
    for (int bm_y = top; bm_y <= bottom; ++bm_y) {
        for (int bm_x = left; bm_x <= right; ++bm_x) {
            float error = 0.f;

            const float * VS_RESTRICT current_patchp = current_patch;
            const float * VS_RESTRICT neighbour_patchp = neighbour_patch;

            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                    error += square(current_patchp[patch_x] - neighbour_patchp[patch_x]);
                }

                current_patchp += block_size;
                neighbour_patchp += stride;
            }

            errors.emplace_back(error, bm_x, bm_y);

            neighbour_patch++;
        }

        neighbour_patch += stride - (right - left + 1);
    }
#endif // __AVX2__
}

static inline void compute_block_distances(
    std::vector<std::tuple<float, int, int>> & errors,
    const float * VS_RESTRICT current_patch,
    const float * VS_RESTRICT refp,
    const std::vector<std::tuple<int, int>> & search_positions,
    int stride,
    int block_size
) noexcept {

#ifdef __AVX2__
    if (block_size == 8) {
        return compute_block_distances_avx2<BlockSizeInfo::Is8>(
            errors,
            current_patch, refp, search_positions, stride, block_size
        );
    } else if ((block_size % 8) == 0) {
        return compute_block_distances_avx2<BlockSizeInfo::Mod8>(
            errors,
            current_patch, refp, search_positions, stride, block_size
        );
    } else {
        return compute_block_distances_avx2<BlockSizeInfo::General>(
            errors,
            current_patch, refp, search_positions, stride, block_size
        );
    }
#else // __AVX2__
    for (const auto & [bm_x, bm_y]: search_positions) {
        float error = 0.f;

        const float * VS_RESTRICT current_patchp = current_patch;
        const float * VS_RESTRICT neighbour_patchp = &refp[bm_y * stride + bm_x];

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                error += square(current_patchp[patch_x] - neighbour_patchp[patch_x]);
            }

            current_patchp += block_size;
            neighbour_patchp += stride;
        }

        errors.emplace_back(error, bm_x, bm_y);
    }
#endif // __AVX2__
}

#ifdef __AVX2__
template <BlockSizeInfo dispatch, bool residual>
static inline void load_patches_avx2(
    float * VS_RESTRICT denoising_patch, int svd_lda,
    std::conditional_t<residual, float * VS_RESTRICT, std::nullptr_t> mean_patch,
    const std::vector<const float *> & srcps,
    const std::vector<std::tuple<float, int, int, int>> & errors,
    int stride,
    int active_group_size,
    int block_size
) noexcept {

    if constexpr (dispatch == BlockSizeInfo::Is8) {
        block_size = 8;
    }

    [[maybe_unused]] std::conditional_t<dispatch == BlockSizeInfo::General, Vec8i, Empty> mask;
    if constexpr (dispatch == BlockSizeInfo::General) {
        mask = make_mask(block_size % 8);
    }

    assert(stride % 8 == 0);

    for (int i = 0; i < active_group_size; ++i) {
        auto [error, bm_x, bm_y, bm_t] = errors[i];

        const float * VS_RESTRICT src_patchp = &srcps[bm_t][bm_y * stride + bm_x];

        [[maybe_unused]] std::conditional_t<residual, float * VS_RESTRICT, std::nullptr_t> mean_patchp { mean_patch };

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            if constexpr (dispatch == BlockSizeInfo::Is8) {
                Vec8f vec_src = Vec8f().load(src_patchp);
                vec_src.store_a(denoising_patch);
                src_patchp += stride;
                denoising_patch += 8;

                if constexpr (residual) {
                    Vec8f vec_mean = Vec8f().load_a(mean_patchp);
                    vec_mean += vec_src;
                    vec_mean.store_a(mean_patchp);
                    mean_patchp += 8;
                }
            } else if constexpr (dispatch == BlockSizeInfo::Mod8) {
                for (int patch_x = 0; patch_x < block_size; patch_x += 8) {
                    Vec8f vec_src = Vec8f().load(src_patchp);
                    vec_src.store_a(denoising_patch);
                    src_patchp += 8;
                    denoising_patch += 8;

                    if constexpr (residual) {
                        Vec8f vec_mean = Vec8f().load_a(mean_patchp);
                        vec_mean += vec_src;
                        vec_mean.store_a(mean_patchp);
                        mean_patchp += 8;
                    }
                }

                src_patchp += stride - block_size;
            } if constexpr (dispatch == BlockSizeInfo::General) {
                for (int patch_x = 0; patch_x < (block_size & (-8)); patch_x += 8) {
                    Vec8f vec_src = Vec8f().load(src_patchp);
                    vec_src.store(denoising_patch);
                    src_patchp += 8;
                    denoising_patch += 8;

                    if constexpr (residual) {
                        Vec8f vec_mean = Vec8f().load(mean_patchp);
                        vec_mean += vec_src;
                        vec_mean.store(mean_patchp);
                        mean_patchp += 8;
                    }
                }

                {
                    Vec8f vec_src = _mm256_maskload_ps(src_patchp, mask);
                    vec_src.store(denoising_patch); // denoising_patch is padded
                    src_patchp += stride - (block_size & (-8));
                    denoising_patch += block_size % 8;

                    if constexpr (residual) {
                        Vec8f vec_mean = Vec8f().load(mean_patchp);
                        vec_mean += vec_src;
                        vec_mean.store(mean_patchp); // mean_patch is padded
                        mean_patchp += block_size % 8;
                    }
                }
            }
        }

        if constexpr (dispatch == BlockSizeInfo::General) {
            denoising_patch += svd_lda - square(block_size);
        } else {
            assert(svd_lda - square(block_size) == 0);
        }
    }
}
#endif // __AVX2__

template <bool residual>
static inline void load_patches(
    float * VS_RESTRICT denoising_patch, int svd_lda,
    std::conditional_t<residual, float * VS_RESTRICT, std::nullptr_t> mean_patch,
    const std::vector<const float *> & srcps,
    const std::vector<std::tuple<float, int, int, int>> & errors,
    int stride,
    int active_group_size,
    int block_size
) noexcept {

#ifdef __AVX2__
    if (block_size == 8) {
        return load_patches_avx2<BlockSizeInfo::Is8, residual>(
            denoising_patch, svd_lda, mean_patch,
            srcps, errors, stride,
            active_group_size, block_size
        );
    } else if ((block_size % 8) == 0) { // block_size % 8 == 0
        return load_patches_avx2<BlockSizeInfo::Mod8, residual>(
            denoising_patch, svd_lda, mean_patch,
            srcps, errors, stride,
            active_group_size, block_size
        );
    } else { // block_size % 8 != 0
        return load_patches_avx2<BlockSizeInfo::General, residual>(
            denoising_patch, svd_lda, mean_patch,
            srcps, errors, stride,
            active_group_size, block_size
        );
    }
#else // __AVX2__
    for (int i = 0, index = 0; i < active_group_size; ++i) {
        auto [error, bm_x, bm_y, bm_t] = errors[i];

        const float * VS_RESTRICT src_patchp = &srcps[bm_t][bm_y * stride + bm_x];

        float * VS_RESTRICT mean_patchp {nullptr};
        if constexpr (residual) {
            mean_patchp = mean_patch;
        }

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                float src_val = src_patchp[patch_x];

                denoising_patch[patch_x] = src_val;

                if constexpr (residual) {
                    mean_patchp[patch_x] += src_val;
                }
            }

            src_patchp += stride;
            denoising_patch += block_size;
        }

        denoising_patch += svd_lda - square(block_size);
    }
#endif // __AVX2__
}

static inline void bm_post(float * VS_RESTRICT mean_patch, float * VS_RESTRICT denoising_patch,
    int block_size, int active_group_size, int svd_lda) noexcept {

    // substract group mean

    for (int i = 0; i < square(block_size); ++i) {
        mean_patch[i] /= active_group_size;
    }

    for (int i = 0; i < active_group_size; ++i) {
        for (int j = 0; j < square(block_size); ++j) {
            denoising_patch[j] -= mean_patch[j];
        }

        denoising_patch += svd_lda;
    }
}

static inline void extend_errors(
    std::vector<std::tuple<float, int, int, int>> & errors,
    const std::vector<std::tuple<float, int, int>> & spatial_errors,
    int temporal_index
) noexcept {

    errors.reserve(std::size(errors) + std::size(spatial_errors));
    for (const auto & [error, x, y] : spatial_errors) {
        errors.emplace_back(error, x, y, temporal_index);
    }
}

template<bool residual>
static inline int block_matching(
    float * VS_RESTRICT denoising_patch, int svd_lda,
    std::vector<std::tuple<float, int, int, int>> & errors,
    float * VS_RESTRICT current_patch,
    std::conditional_t<residual, float * VS_RESTRICT, std::nullptr_t> mean_patch,
    const std::vector<const float *> & srcps, // length: 2 * radius + 1
    const std::vector<const float *> & refps, // length: 2 * radius + 1
    int width, int height, int stride,
    int x, int y,
    int block_size, int group_size, int bm_range,
    int ps_num, int ps_range,
    std::vector<std::tuple<float, int, int>> & center_errors,
    std::vector<std::tuple<int, int>> & search_locations,
    std::vector<std::tuple<int, int>> & new_locations,
    std::vector<std::tuple<int, int>> & locations_copy,
    std::vector<std::tuple<float, int, int>> & temporal_errors
) noexcept {

    errors.clear();
    center_errors.clear();

    auto radius = (static_cast<int>(std::size(srcps)) - 1) / 2;

    vs_bitblt(
        current_patch, block_size * sizeof(float),
        &refps[radius][y * stride + x], stride * sizeof(float),
        block_size * sizeof(float), block_size
    );

    int top = std::max(y - bm_range, 0);
    int bottom = std::min(y + bm_range, height - block_size);
    int left = std::max(x - bm_range, 0);
    int right = std::min(x + bm_range, width - block_size);

    compute_block_distances(
        center_errors,
        current_patch,
        &refps[radius][top * stride + left],
        top, bottom, left, right,
        stride, block_size
    );

    if (radius == 0) {
        extend_errors(errors, center_errors, radius);
    } else {
        int active_ps_num = std::min(
            ps_num,
            static_cast<int>(std::size(center_errors))
        );

        int active_num = std::min(
            std::max(group_size, ps_num),
            static_cast<int>(std::size(center_errors))
        );

        std::partial_sort(
            center_errors.begin(),
            center_errors.begin() + active_num,
            center_errors.end(),
            [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); }
        );
        center_errors.resize(active_num);
        extend_errors(errors, center_errors, radius);

        for (int direction = -1; direction <= 1; direction += 2) {
            temporal_errors = center_errors; // mutable

            for (int i = 1; i <= radius; i++) {
                auto temporal_index = radius + direction * i;

                generate_search_locations(
                    std::data(temporal_errors), active_ps_num,
                    block_size, width, height, ps_range,
                    search_locations, new_locations, locations_copy
                );

                temporal_errors.clear();

                compute_block_distances(
                    temporal_errors,
                    current_patch,
                    refps[temporal_index],
                    search_locations,
                    stride, block_size
                );

                auto active_temporal_num = std::min(
                    std::max(group_size, ps_num),
                    static_cast<int>(std::size(temporal_errors))
                );

                std::partial_sort(
                    temporal_errors.begin(),
                    temporal_errors.begin() + active_temporal_num,
                    temporal_errors.end(),
                    [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); }
                );
                temporal_errors.resize(active_temporal_num);
                extend_errors(errors, temporal_errors, temporal_index);
            }
        }
    }

    int active_group_size = std::min(group_size, static_cast<int>(std::size(errors)));
    std::partial_sort(
        errors.begin(),
        errors.begin() + active_group_size,
        errors.end(),
        [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); }
    );
    errors.resize(active_group_size);
    bool center = false;
    for (int i = 0; i < active_group_size; i++) {
        const auto & [_, bm_x, bm_y, bm_t] = errors[i];
        if (bm_x == x && bm_y == y && bm_t == radius) {
            center = true;
        }
    }
    if (!center) {
        errors[0] = std::make_tuple(0.0f, x, y, radius);
    }

    load_patches<residual>(
        denoising_patch, svd_lda, mean_patch,
        srcps, errors, stride, active_group_size, block_size);

    if constexpr (residual) {
        bm_post(mean_patch, denoising_patch, block_size, active_group_size, svd_lda);
    }

    return active_group_size;
}

template<bool residual>
static inline WnnmInfo patch_estimation(
    float * VS_RESTRICT denoising_patch, int svd_lda,
    float & adaptive_weight,
    float sigma,
    int block_size, int active_group_size,
    float * VS_RESTRICT mean_patch,
    bool adaptive_aggregation,
    float * VS_RESTRICT svd_s,
    float * VS_RESTRICT svd_u, int svd_ldu,
    float * VS_RESTRICT svd_vt, int svd_ldvt,
    float * VS_RESTRICT svd_work, int svd_lwork, int * VS_RESTRICT svd_iwork
) noexcept {

    int m = square(block_size);
    int n = active_group_size;

    int svd_info;
    sgesdd(
        "S", &m, &n,
        denoising_patch, &svd_lda,
        svd_s,
        svd_u, &svd_ldu,
        svd_vt, &svd_ldvt,
        svd_work, &svd_lwork, svd_iwork, &svd_info
    );

    if (svd_info != 0) {
        return WnnmInfo::FAILURE;
    }

    // WNNP with parameter epsilon ignored
    const float constant = 8.f * sqrtf(2.0f * n) * square(sigma);

    int k = 1;
    if constexpr (residual) {
        k = 0;
    }

    for ( ; k < std::min(m, n); ++k) {
        float s = svd_s[k];
        float tmp = square(s) - constant;
        if (tmp > 0.f) {
            svd_s[k] = (s + sqrtf(tmp)) * 0.5f;
        } else {
            break;
        }
    }

    if (adaptive_aggregation) {
        adaptive_weight = (k > 0) ? (1.f / k) : 1.0f;
    }

    // gemm
    if (m < n) {
        float * VS_RESTRICT svd_up {svd_u};

        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < m; ++j) {
                svd_up[j] *= svd_s[i];
            }
            svd_up += svd_ldu;
        }
    } else {
        float * VS_RESTRICT svd_vtp {svd_vt};

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                svd_vtp[j] *= svd_s[j];
            }
            svd_vtp += svd_ldvt;
        }
    }

    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    sgemm("N", "N", &m, &n, &k, &alpha, svd_u, &svd_ldu, svd_vt, &svd_ldvt, &beta, denoising_patch, &svd_lda);

    if constexpr (residual) {
        for (int i = 0; i < active_group_size; ++i) {
            for (int patch_i = 0; patch_i < square(block_size); ++patch_i) {
                denoising_patch[patch_i] += mean_patch[patch_i];
            }

            denoising_patch += svd_lda;
        }
    }

    return WnnmInfo::SUCCESS;
}

static inline void col2im(
    float * VS_RESTRICT intermediate,
    const float * VS_RESTRICT denoising_patch, int svd_lda,
    const std::vector<std::tuple<float, int, int, int>> & errors,
    int height, int intermediate_stride,
    int block_size, int active_group_size,
    float adaptive_weight
) noexcept {

    for (int i = 0; i < active_group_size; ++i) {
        auto [error, bm_x, bm_y, bm_t] = errors[i];

        float * VS_RESTRICT wdstp = &intermediate[(bm_t * 2 * height + bm_y) * intermediate_stride + bm_x];
        float * VS_RESTRICT weightp = &intermediate[((bm_t * 2 + 1) * height + bm_y) * intermediate_stride + bm_x];

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                wdstp[patch_x] += denoising_patch[patch_x] * adaptive_weight;
                weightp[patch_x] += adaptive_weight;
            }

            wdstp += intermediate_stride;
            weightp += intermediate_stride;
            denoising_patch += block_size;
        }

        denoising_patch += svd_lda - square(block_size);
    }
}

static void patch_estimation_skip(
    float * VS_RESTRICT intermediate,
    const std::vector<const float *> & srcps,
    const std::vector<std::tuple<float, int, int, int>> & errors,
    int height, int stride, int intermediate_stride,
    int block_size, int active_group_size
) noexcept {

    for (int i = 0; i < active_group_size; ++i) {
        auto [error, bm_x, bm_y, bm_t] = errors[i];

        const float * VS_RESTRICT srcp = &srcps[bm_t][bm_y * stride + bm_x];
        float * VS_RESTRICT wdstp = &intermediate[(bm_t * 2 * height + bm_y) * intermediate_stride + bm_x];
        float * VS_RESTRICT weightp = &intermediate[((bm_t * 2 + 1) * height + bm_y) * intermediate_stride + bm_x];

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                wdstp[patch_x] += srcp[patch_x];
                weightp[patch_x] += 1.f;
            }

            srcp += stride;
            wdstp += intermediate_stride;
            weightp += intermediate_stride;
        }
    }
}

static inline void aggregation(
    float * VS_RESTRICT dstp,
    const float * VS_RESTRICT intermediate,
    int width, int height, int stride
) noexcept {

    const float * VS_RESTRICT wdst = intermediate;
    const float * VS_RESTRICT weight = &intermediate[height * width];

    for (int y = 0; y < height; ++y) {
        int x = 0;

#ifdef __AVX2__
        const float * VS_RESTRICT vec_wdstp { wdst };
        const float * VS_RESTRICT vec_weightp { weight };
        float * VS_RESTRICT vec_dstp {dstp};

        for ( ; x < (width & (-8)); x += 8) {
            Vec8f vec_wdst = Vec8f().load(vec_wdstp);
            Vec8f vec_weight = Vec8f().load(vec_weightp);
            Vec8f vec_dst = vec_wdst * approx_recipr(vec_weight);
            vec_dst.store_a(vec_dstp);

            vec_wdstp += 8;
            vec_weightp += 8;
            vec_dstp += 8;
        }
#endif

        for ( ; x < width; ++x) {
            dstp[x] = wdst[x] / weight[x];
        }

        dstp += stride;
        wdst += width;
        weight += width;
    }
}

template<bool residual>
static void process(
    const std::vector<const VSFrameRef *> & srcs,
    const std::vector<const VSFrameRef *> & refs,
    VSFrameRef * dst,
    WNNMData * d,
    const VSAPI * vsapi
) noexcept {

    const auto threadId = std::this_thread::get_id();

#ifdef __AVX2__
    auto control_word = get_control_word();
    no_subnormals();
#endif

    Workspace workspace {};
    bool init = true;

    d->workspaces_lock.lock_shared();

    try {
        const auto & const_workspaces = d->workspaces;
        workspace = const_workspaces.at(threadId);
    } catch (const std::out_of_range &) {
        init = false;
    }

    d->workspaces_lock.unlock_shared();

    auto vi = vsapi->getVideoInfo(d->node);

    if (!init) {
        workspace.init(
            vi->width, vi->height,
            d->block_size, d->group_size, d->radius,
            d->residual,
            d->svd_lda, d->svd_ldu, d->svd_ldvt, d->svd_lwork
        );

        d->workspaces_lock.lock();
        d->workspaces.emplace(threadId, workspace);
        d->workspaces_lock.unlock();
    }

    std::conditional_t<residual, float * VS_RESTRICT, std::nullptr_t> mean_patch {};
    if constexpr (residual) {
        mean_patch = workspace.mean_patch;
    }

    std::vector<std::tuple<float, int, int, int>> & errors = *workspace.errors;

    for (int plane = 0; plane < vi->format->numPlanes; plane++) {
        if (!d->process[plane]) {
            continue;
        }

        const int width = vsapi->getFrameWidth(srcs[0], plane);
        const int height = vsapi->getFrameHeight(srcs[0], plane);
        const int stride = vsapi->getStride(srcs[0], plane) / static_cast<int>(sizeof(float));
        std::vector<const float *> srcps;
        srcps.reserve(std::size(srcs));
        for (const auto & src : srcs) {
            srcps.emplace_back(reinterpret_cast<const float *>(vsapi->getReadPtr(src, plane)));
        }
        std::vector<const float *> refps;
        refps.reserve(std::size(refs));
        for (const auto & ref : refs) {
            refps.emplace_back(reinterpret_cast<const float *>(vsapi->getReadPtr(ref, plane)));
        }
        float * const VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, plane));

        if (d->radius == 0) {
            std::memset(workspace.intermediate, 0, 2 * height * width * sizeof(float));
        } else {
            std::memset(dstp, 0, 2 * (2 * d->radius + 1) * height * stride * sizeof(float));
        }

        int temp_r = height - d->block_size;
        int temp_c = width - d->block_size;

        for (int _y = 0; _y < temp_r + d->block_step; _y += d->block_step) {
            int y = std::min(_y, temp_r); // clamp

            for (int _x = 0; _x < temp_c + d->block_step; _x += d->block_step) {
                int x = std::min(_x, temp_c); // clamp

                if constexpr (residual) {
                    std::memset(mean_patch, 0, sizeof(float) * square(d->block_size));
                }

                int active_group_size = block_matching<residual>(
                    // outputs
                    workspace.denoising_patch, d->svd_lda,
                    errors,
                    workspace.current_patch, mean_patch,
                    // inputs
                    srcps, refps, width, height, stride,
                    x, y,
                    d->block_size, d->group_size, d->bm_range,
                    d->ps_num, d->ps_range,
                    *workspace.center_errors,
                    *workspace.search_locations,
                    *workspace.new_locations,
                    *workspace.locations_copy,
                    *workspace.temporal_errors
                );

                // patch_estimation with early skipping on SVD exception
                float adaptive_weight = 1.f;
                WnnmInfo info = patch_estimation<residual>(
                    // outputs
                    workspace.denoising_patch, d->svd_lda,
                    adaptive_weight,
                    // inputs
                    d->sigma[plane],
                    d->block_size, active_group_size, mean_patch, d->adaptive_aggregation,
                    // temporaries
                    workspace.svd_s, workspace.svd_u, d->svd_ldu, workspace.svd_vt, d->svd_ldvt,
                    workspace.svd_work, d->svd_lwork, workspace.svd_iwork
                );

                switch (info) {
                    case WnnmInfo::SUCCESS: {
                        if (d->radius == 0) {
                            col2im(
                                // output
                                workspace.intermediate,
                                // inputs
                                workspace.denoising_patch, d->svd_lda,
                                errors, height, width,
                                d->block_size, active_group_size, adaptive_weight
                            );
                        } else {
                            col2im(
                                // output
                                dstp,
                                // inputs
                                workspace.denoising_patch, d->svd_lda,
                                errors, height, stride,
                                d->block_size, active_group_size, adaptive_weight
                            );
                        }
                        break;
                    }
                    case WnnmInfo::FAILURE: {
                        if (d->radius == 0) {
                            patch_estimation_skip(
                                // output
                                workspace.intermediate, srcps,
                                errors, height, stride, width,
                                d->block_size, active_group_size
                            );
                        } else {
                            patch_estimation_skip(
                                // output
                                dstp,
                                // inputs
                                srcps,
                                errors, height, stride, stride,
                                d->block_size, active_group_size
                            );
                        }
                        break;
                    }
                }
            }
        }

        if (d->radius == 0) {
            aggregation(dstp, workspace.intermediate, width, height, stride);
        }
    }

#ifdef __AVX2__
    set_control_word(control_word);
#endif
}

static void VS_CC WNNMRawInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    WNNMData * d = static_cast<WNNMData *>(*instanceData);

    if (d->radius > 0) {
        auto vi = *vsapi->getVideoInfo(d->node);
        vi.height *= 2 * (2 * d->radius + 1);
        vsapi->setVideoInfo(&vi, 1, node);
    } else {
        auto vi = vsapi->getVideoInfo(d->node);
        vsapi->setVideoInfo(vi, 1, node);
    }
}

static const VSFrameRef *VS_CC WNNMRawGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto * d = static_cast<WNNMData *>(*instanceData);

    if (activationReason == arInitial) {
        auto vi = vsapi->getVideoInfo(d->node);

        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, vi->numFrames - 1);

        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        if (d->ref_node) {
            for (int i = start_frame; i <= end_frame; ++i) {
                vsapi->requestFrameFilter(i, d->ref_node, frameCtx);
            }
        }
    } else if (activationReason == arAllFramesReady) {
        auto vi = vsapi->getVideoInfo(d->node);

        std::vector<const VSFrameRef *> srcs;
        srcs.reserve(2 * d->radius + 1);
        for (int i = -d->radius; i <= d->radius; i++) {
            auto frame_id = std::clamp(n + i, 0, vi->numFrames - 1);
            srcs.emplace_back(vsapi->getFrameFilter(frame_id, d->node, frameCtx));
        }

        std::vector<const VSFrameRef *> refs;
        if (d->ref_node) {
            refs.reserve(2 * d->radius + 1);
            for (int i = -d->radius; i <= d->radius; i++) {
                auto frame_id = std::clamp(n + i, 0, vi->numFrames - 1);
                refs.emplace_back(vsapi->getFrameFilter(frame_id, d->ref_node, frameCtx));
            }
        } else {
            refs = srcs;
        }

        const auto & center_src = srcs[d->radius];
        VSFrameRef * dst;
        if (d->radius == 0) {
            const VSFrameRef * fr[] {
                d->process[0] ? nullptr : center_src,
                d->process[1] ? nullptr : center_src,
                d->process[2] ? nullptr : center_src
            };
            const int pl[] { 0, 1, 2 };
            dst = vsapi->newVideoFrame2(vi->format, vi->width, vi->height, fr, pl, center_src, core);
        } else {
            dst = vsapi->newVideoFrame(vi->format, vi->width, 2 * (2 * d->radius + 1) * vi->height, center_src, core);
        }

        if (d->residual) {
            process<true>(srcs, refs, dst, d, vsapi);
        } else {
            process<false>(srcs, refs, dst, d, vsapi);
        }

        for (const auto & src : srcs) {
            vsapi->freeFrame(src);
        }

        if (d->ref_node) {
            for (const auto & ref : refs) {
                vsapi->freeFrame(ref);
            }
        }

        return dst;
    }

    return nullptr;
}

static void VS_CC WNNMRawFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = static_cast<WNNMData *>(instanceData);

    vsapi->freeNode(d->node);

    if (d->ref_node) {
        vsapi->freeNode(d->ref_node);
    }

    for (auto & [_, workspace] : d->workspaces) {
        workspace.release();
    }

    delete d;
}

static void VS_CC WNNMRawCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = std::make_unique<WNNMData>();

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);

    auto set_error = [&](const std::string & error) -> void {
        vsapi->setError(out, ("WNNM: " + error).c_str());
        vsapi->freeNode(d->node);
        return ;
    };

    auto vi = vsapi->getVideoInfo(d->node);

    if (!isConstantFormat(vi) || vi->format->sampleType == stInteger ||
        (vi->format->sampleType == stFloat && vi->format->bitsPerSample != 32)
    ) {
        return set_error("only constant format 32 bit float input supported");
    }

    int error;

    for (unsigned i = 0; i < std::size(d->sigma); i++) {
        d->sigma[i] = static_cast<float>(vsapi->propGetFloat(in, "sigma", i, &error));
        if (error) {
            d->sigma[i] = (i == 0) ? 3.0f : d->sigma[i - 1];
        }
        if (d->sigma[i] < 0.0f) {
            return set_error("\"sigma\" must be positive");
        }
    }

    for (unsigned i = 0; i < std::size(d->sigma); ++i) {
        if (d->sigma[i] < std::numeric_limits<float>::epsilon()) {
            d->process[i] = false;
        } else {
            d->process[i] = true;
            d->sigma[i] /= 255.f;
        }
    }

    d->block_size = int64ToIntS(vsapi->propGetInt(in, "block_size", 0, &error));
    if (error) {
        // d->block_size = 6;
        d->block_size = 8; // more optimized
    } else if (d->block_size <= 0) {
        return set_error("\"sigma\" must be positive");
    }

    d->block_step = int64ToIntS(vsapi->propGetInt(in, "block_step", 0, &error));
    if (error) {
        // d->block_step = 6;
        d->block_step = 8; // follows the change in block_step
    } else if (d->block_step <= 0 || d->block_step > d->block_size) {
        return set_error("\"block_step\" must be positive and no larger than \"block_size\"");
    }

    d->group_size = int64ToIntS(vsapi->propGetInt(in, "group_size", 0, &error));
    if (error) {
        d->group_size = 8;
    } else if (d->group_size <= 0) {
        return set_error("\"group_size\" must be positive");
    }

    d->bm_range = int64ToIntS(vsapi->propGetInt(in, "bm_range", 0, &error));
    if (error) {
        d->bm_range = 7;
    } else if (d->bm_range < 0) {
        return set_error("\"bm_range\" must be non-negative");
    }

    d->radius = int64ToIntS(vsapi->propGetInt(in, "radius", 0, &error));
    if (error) {
        d->radius = 0;
    } else if (d->radius < 0) {
        return set_error("\"radius\" must be non-negative");
    }

    d->ps_num = int64ToIntS(vsapi->propGetInt(in, "ps_num", 0, &error));
    if (error) {
        d->ps_num = 2;
    } else if (d->ps_num <= 0) {
        return set_error("\"ps_num\" must be positive");
    }

    d->ps_range = int64ToIntS(vsapi->propGetInt(in, "ps_range", 0, &error));
    if (error) {
        d->ps_range = 4;
    } else if (d->ps_range < 0) {
        return set_error("\"ps_range\" must be non-negative");
    }

    d->svd_lda = m16(square(d->block_size));
    d->svd_ldu = m16(square(d->block_size));
    d->svd_ldvt = m16(std::min(square(d->block_size), d->group_size));

    d->residual = !!vsapi->propGetInt(in, "residual", 0, &error);
    if (error) {
        d->residual = false;
    }

    d->adaptive_aggregation = !!vsapi->propGetInt(in, "adaptive_aggregation", 0, &error);
    if (error) {
        d->adaptive_aggregation = true;
    }

    d->ref_node = vsapi->propGetNode(in, "rclip", 0, &error);
    if (error) {
        d->ref_node = nullptr;
    } else {
        auto ref_vi = vsapi->getVideoInfo(d->ref_node);
        if (!isSameFormat(vi, ref_vi) || vi->numFrames != ref_vi->numFrames) {
            return set_error("\"rclip\" must be of the same format and number of frames as \"clip\"");
        }
    }

    VSCoreInfo core_info;
    vsapi->getCoreInfo2(core, &core_info);
    auto numThreads = core_info.numThreads;
    d->workspaces.reserve(numThreads);

    int svd_m = square(d->block_size);
    int svd_n = d->group_size;
    d->svd_lwork = std::min(svd_m, svd_n) * (6 + 4 * std::min(svd_m, svd_n)) + std::max(svd_m, svd_n);

    vsapi->createFilter(in, out, "WNNMRaw", WNNMRawInit, WNNMRawGetFrame, WNNMRawFree, fmParallel, 0, d.release(), core);
}

struct VAggregateData {
    VSNodeRef * node;

    VSNodeRef * src_node;
    const VSVideoInfo * src_vi;

    std::array<bool, 3> process; // sigma != 0

    int radius;

    std::unordered_map<std::thread::id, float *> buffer;
    std::shared_mutex buffer_lock;
};

static void VS_CC VAggregateInit(
    VSMap *in, VSMap *out, void **instanceData, VSNode *node,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto * d = static_cast<VAggregateData *>(*instanceData);

    vsapi->setVideoInfo(d->src_vi, 1, node);
}

static const VSFrameRef *VS_CC VAggregateGetFrame(
    int n, int activationReason, void **instanceData, void **frameData,
    VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto * d = static_cast<VAggregateData *>(*instanceData);

    if (activationReason == arInitial) {
        int start_frame = std::max(n - d->radius, 0);
        int end_frame = std::min(n + d->radius, d->src_vi->numFrames - 1);

        for (int i = start_frame; i <= end_frame; ++i) {
            vsapi->requestFrameFilter(i, d->node, frameCtx);
        }
        vsapi->requestFrameFilter(n, d->src_node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src_frame = vsapi->getFrameFilter(n, d->src_node, frameCtx);

        std::vector<const VSFrameRef *> frames;
        frames.reserve(2 * d->radius + 1);
        for (int i = n - d->radius; i <= n + d->radius; ++i) {
            auto frame_id = std::clamp(i, 0, d->src_vi->numFrames - 1);
            frames.emplace_back(vsapi->getFrameFilter(frame_id, d->node, frameCtx));
        }

        float * buffer {};
        {
            const auto thread_id = std::this_thread::get_id();
            bool init = true;

            d->buffer_lock.lock_shared();

            try {
                const auto & const_buffer = d->buffer;
                buffer = const_buffer.at(thread_id);
            } catch (const std::out_of_range &) {
                init = false;
            }

            d->buffer_lock.unlock_shared();

            if (!init) {
                assert(d->process[0] || d->src_vi->format->numPlanes > 1);

                const int max_width {
                    d->process[0] ?
                    vsapi->getFrameWidth(src_frame, 0) :
                    vsapi->getFrameWidth(src_frame, 1)
                };

                buffer = reinterpret_cast<float *>(std::malloc(2 * max_width * sizeof(float)));

                std::lock_guard _ { d->buffer_lock };
                d->buffer.emplace(thread_id, buffer);
            }
        }

        const VSFrameRef * fr[] {
            d->process[0] ? nullptr : src_frame,
            d->process[1] ? nullptr : src_frame,
            d->process[2] ? nullptr : src_frame
        };
        constexpr int pl[] { 0, 1, 2 };
        auto dst_frame = vsapi->newVideoFrame2(
            d->src_vi->format,
            d->src_vi->width, d->src_vi->height,
            fr, pl, src_frame, core);

        for (int plane = 0; plane < d->src_vi->format->numPlanes; ++plane) {
            if (d->process[plane]) {
                int plane_width = vsapi->getFrameWidth(src_frame, plane);
                int plane_height = vsapi->getFrameHeight(src_frame, plane);
                int plane_stride = vsapi->getStride(src_frame, plane) / sizeof(float);

                std::vector<const float *> srcps;
                srcps.reserve(2 * d->radius + 1);
                for (int i = 0; i < 2 * d->radius + 1; ++i) {
                    srcps.emplace_back(reinterpret_cast<const float *>(vsapi->getReadPtr(frames[i], plane)));
                }

                auto dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst_frame, plane));

                for (int y = 0; y < plane_height; ++y) {
                    std::memset(buffer, 0, 2 * plane_width * sizeof(float));
                    for (int i = 0; i < 2 * d->radius + 1; ++i) {
                        auto agg_src = srcps[i];
                        agg_src += ((2 * d->radius - i) * 2 * plane_height + y) * plane_stride;
                        for (int x = 0; x < plane_width; ++x) {
                            buffer[x] += agg_src[x];
                        }
                        agg_src += plane_height * plane_stride;
                        for (int x = 0; x < plane_width; ++x) {
                            buffer[plane_width + x] += agg_src[x];
                        }
                    }
                    for (int x = 0; x < plane_width; ++x) {
                        dstp[x] = buffer[x] / buffer[plane_width + x];
                    }
                    dstp += plane_stride;
                }
            }
        }

        for (const auto & frame : frames) {
            vsapi->freeFrame(frame);
        }
        vsapi->freeFrame(src_frame);

        return dst_frame;
    }

    return nullptr;
}

static void VS_CC VAggregateFree(
    void *instanceData, VSCore *core, const VSAPI *vsapi
) noexcept {

    auto * d = static_cast<VAggregateData *>(instanceData);

    for (const auto & [_, ptr] : d->buffer) {
        std::free(ptr);
    }

    vsapi->freeNode(d->src_node);
    vsapi->freeNode(d->node);

    delete d;
}

static void VS_CC VAggregateCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    auto d = std::make_unique<VAggregateData>();

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto vi = vsapi->getVideoInfo(d->node);
    d->src_node = vsapi->propGetNode(in, "src", 0, nullptr);
    d->src_vi = vsapi->getVideoInfo(d->src_node);

    d->radius = (vi->height / d->src_vi->height - 2) / 4;

    d->process.fill(false);
    int num_planes_args = vsapi->propNumElements(in, "planes");
    for (int i = 0; i < num_planes_args; ++i) {
        int plane = int64ToIntS(vsapi->propGetInt(in, "planes", i, nullptr));
        d->process[plane] = true;
    }

    VSCoreInfo core_info;
    vsapi->getCoreInfo2(core, &core_info);
    d->buffer.reserve(core_info.numThreads);

    vsapi->createFilter(
        in, out, "VAggregate",
        VAggregateInit, VAggregateGetFrame, VAggregateFree,
        fmParallel, 0, d.release(), core);
}

static void VS_CC WNNMCreate(
    const VSMap *in, VSMap *out, void *userData,
    VSCore *core, const VSAPI *vsapi
) noexcept {

    std::array<bool, 3> process;
    process.fill(true);

    int num_sigma_args = vsapi->propNumElements(in, "sigma");
    for (int i = 0; i < std::min(3, num_sigma_args); ++i) {
        auto sigma = vsapi->propGetFloat(in, "sigma", i, nullptr);
        if (sigma < std::numeric_limits<float>::epsilon()) {
            process[i] = false;
        }
    }

    bool skip = true;
    auto src = vsapi->propGetNode(in, "clip", 0, nullptr);
    auto src_vi = vsapi->getVideoInfo(src);
    for (int i = 0; i < src_vi->format->numPlanes; ++i) {
        skip &= !process[i];
    }
    if (skip) {
        vsapi->propSetNode(out, "clip", src, paReplace);
        vsapi->freeNode(src);
        return ;
    }

    auto map = vsapi->invoke(myself, "WNNMRaw", in);
    if (auto error = vsapi->getError(map); error) {
        vsapi->setError(out, error);
        vsapi->freeMap(map);
        vsapi->freeNode(src);
        return ;
    }

    int err;
    int radius = int64ToIntS(vsapi->propGetInt(in, "radius", 0, &err));
    if (err) {
        radius = 0;
    }
    if (radius == 0) {
        // spatial WNNM should handle everything itself
        auto node = vsapi->propGetNode(map, "clip", 0, nullptr);
        vsapi->freeMap(map);
        vsapi->propSetNode(out, "clip", node, paReplace);
        vsapi->freeNode(node);
        vsapi->freeNode(src);
        return ;
    }

    vsapi->propSetNode(map, "src", src, paReplace);
    vsapi->freeNode(src);

    for (int i = 0; i < 3; ++i) {
        if (process[i]) {
            vsapi->propSetInt(map, "planes", i, paAppend);
        }
    }

    auto map2 = vsapi->invoke(myself, "VAggregate", map);
    vsapi->freeMap(map);
    if (auto error = vsapi->getError(map2); error) {
        vsapi->setError(out, error);
        vsapi->freeMap(map2);
        return ;
    }

    auto node = vsapi->propGetNode(map2, "clip", 0, nullptr);
    vsapi->freeMap(map2);
    vsapi->propSetNode(out, "clip", node, paReplace);
    vsapi->freeNode(node);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(
    VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin
) noexcept {

    myself = plugin;

    configFunc(
        "com.wolframrhodium.wnnm",
        "wnnm", "Weighted Nuclear Norm Minimization Denoiser",
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    constexpr auto wnnm_args {
        "clip:clip;"
        "sigma:float[]:opt;"
        "block_size:int:opt;"
        "block_step:int:opt;"
        "group_size:int:opt;"
        "bm_range:int:opt;"
        "radius:int:opt;"
        "ps_num:int:opt;"
        "ps_range:int:opt;"
        "residual:int:opt;"
        "adaptive_aggregation:int:opt;"
        "rclip:clip:opt;"
    };

    registerFunc("WNNMRaw", wnnm_args, WNNMRawCreate, nullptr, plugin);

    registerFunc(
        "VAggregate",
        "clip:clip;"
        "src:clip;"
        "planes:int[];",
        VAggregateCreate, nullptr, plugin);

    registerFunc("WNNM", wnnm_args, WNNMCreate, nullptr, plugin);

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);

        std::ostringstream mkl_version_build_str;
        mkl_version_build_str << __INTEL_MKL__ << '.' << __INTEL_MKL_MINOR__ << '.' << __INTEL_MKL_UPDATE__;

        vsapi->propSetData(out, "mkl_version_build", mkl_version_build_str.str().c_str(), -1, paReplace);

        MKLVersion version;
        mkl_get_version(&version);

        vsapi->propSetData(out, "mkl_processor", version.Processor, -1, paReplace);

        std::ostringstream mkl_version_str;
        mkl_version_str << version.MajorVersion << '.' << version.MinorVersion << '.' << version.UpdateVersion;

        vsapi->propSetData(out, "mkl_version", mkl_version_str.str().c_str(), -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);
}
