#include <algorithm>
#include <cmath>
#include <cfloat>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

// MKL
#include <mkl.h>

#ifdef __AVX2__
#include <vectorclass.h>
#include <immintrin.h>
#endif

#include <VapourSynth.h>
#include <VSHelper.h>

#include <config.h>

template <typename T>
static inline T square(T const & x) {
    return x * x;
}

static inline int m16(int x) noexcept {
    return ((x - 1) / 16 + 1) * 16;
}

namespace {
struct Workspace {
    bool * denoised;
    float * wdst;
    float * weight;
    float * denoising_patch;
    float * mean_patch;
    float * current_patch;
    float * svd_s;
    float * svd_u;
    float * svd_vt;
    float * svd_work;
    int * svd_iwork;

    void init(
        int width, int height, 
        int block_size, int group_size, int bm_range, 
        bool residual, bool fast,
        int svd_lda, int svd_ldu, int svd_ldvt, int svd_lwork
    ) noexcept {

        if (residual) {
            mean_patch = vs_aligned_malloc<float>(sizeof(float) * m16(square(block_size)), 64);
        } else {
            mean_patch = nullptr;
        }

        if (fast) {
            denoised = vs_aligned_malloc<bool>(sizeof(bool) * height * width, 64);
        } else {
            denoised = nullptr;
        }

        current_patch = vs_aligned_malloc<float>(sizeof(float) * m16(square(block_size)), 64);

        wdst = vs_aligned_malloc<float>(sizeof(float) * height * width, 64);

        weight = vs_aligned_malloc<float>(sizeof(float) * height * width, 64);

        int m = square(block_size);
        int n = VSMIN(group_size, square(2 * bm_range + 1));

        denoising_patch = vs_aligned_malloc<float>(sizeof(float) * svd_lda * n, 64);

        svd_s = vs_aligned_malloc<float>(sizeof(float) * VSMIN(m, n), 64);

        svd_u = vs_aligned_malloc<float>(sizeof(float) * svd_ldu * VSMIN(m, n), 64);

        svd_vt = vs_aligned_malloc<float>(sizeof(float) * svd_ldvt * n, 64);

        svd_work = vs_aligned_malloc<float>(sizeof(float) * svd_lwork, 64);

        svd_iwork = vs_aligned_malloc<int>(sizeof(int) * 8 * VSMIN(m, n), 64);
    }

    void release() noexcept {
        vs_aligned_free(mean_patch);
        mean_patch = nullptr;

        vs_aligned_free(denoised);
        denoised = nullptr;

        vs_aligned_free(current_patch);
        current_patch = nullptr;

        vs_aligned_free(wdst);
        wdst = nullptr;

        vs_aligned_free(weight);
        weight = nullptr;

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
    }
};

struct WNNMData {
    VSNodeRef * node;
    const VSVideoInfo * vi;
    float sigma[3];
    int block_size, block_step, group_size, bm_range;
    int svd_lwork, svd_lda, svd_ldu, svd_ldvt;
    bool process[3];
    bool residual, adaptive_aggregation, fast;

    std::unordered_map<std::thread::id, Workspace> workspaces;
    std::shared_mutex workspaces_lock;
};

enum class WnnmInfo { SUCCESS, FAILURE, EMPTY };
} // namespace

#ifdef __AVX2__
static inline Vec8i make_mask(int block_size_m8) noexcept {
    static constexpr int temp[16] {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};

    return Vec8i().load(temp + 8 - block_size_m8);
}
#endif

template<bool residual, bool fast>
static inline void bm_sorted(std::vector<std::tuple<float, int, int>> & errors, const float * VS_RESTRICT srcp, 
    int width, int x, int y, int top, int bottom, int left, int right, int block_size, int group_size, 
    int stride, float * VS_RESTRICT denoising_patch, float * VS_RESTRICT mean_patch, 
    float * VS_RESTRICT current_patch, int svd_lda, bool * VS_RESTRICT denoised) noexcept {

    errors.clear();

    {
        float * VS_RESTRICT current_patchp = current_patch;
        const float * VS_RESTRICT src_patchp = srcp + y * stride + x;

#ifdef __AVX2__
        if (block_size == 8) {
            constexpr int block_size8 {8};

            for (int patch_y = 0; patch_y < block_size8; ++patch_y) {
                Vec8f vec_src = Vec8f().load(src_patchp);
                vec_src.store_a(current_patchp);

                src_patchp += stride;
                current_patchp += 8;
            }
        } else if ((block_size & 7) == 0) { // block_size % 8 == 0
            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                for (int patch_x = 0; patch_x < block_size; patch_x += 8) {
                    Vec8f vec_src = Vec8f().load(src_patchp);
                    vec_src.store_a(current_patchp);

                    src_patchp += 8;
                    current_patchp += 8;
                }
                src_patchp += stride - block_size;
            }
        } else {
#else
        {
#endif
            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                memcpy(current_patchp, src_patchp, sizeof(float) * block_size);
                current_patchp += block_size;
                src_patchp += stride;
            }
        }
    }

    const float * VS_RESTRICT neighbour_patchpc = srcp + top * stride;

#ifdef __AVX2__
    if (block_size == 8) {
        constexpr int block_size8 {8};

        for (int bm_y = top; bm_y <= bottom; ++bm_y) {
            for (int bm_x = left; bm_x <= right; ++bm_x) {
                Vec8f vec_error {0.f};

                const float * VS_RESTRICT current_patchp = current_patch;
                const float * VS_RESTRICT neighbour_patchp = neighbour_patchpc + bm_x;

                for (int patch_y = 0; patch_y < block_size8; ++patch_y) {
                    Vec8f vec_current = Vec8f().load_a(current_patchp);
                    Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                    Vec8f diff = vec_current - vec_neighbour;
                    vec_error = mul_add(diff, diff, vec_error);

                    current_patchp += 8;
                    neighbour_patchp += stride;
                }

                float error { horizontal_add(vec_error) };

                errors.push_back(std::make_tuple(error, bm_x, bm_y));
            }

            neighbour_patchpc += stride;
        }
    } else if ((block_size & 7) == 0) { // block_size % 8 == 0 && block_size >= 16
        for (int bm_y = top; bm_y <= bottom; ++bm_y) {
            for (int bm_x = left; bm_x <= right; ++bm_x) {
                Vec8f vec_error {0.f};

                const float * VS_RESTRICT current_patchp = current_patch;
                const float * VS_RESTRICT neighbour_patchp = neighbour_patchpc + bm_x;

                for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                    for (int patch_x = 0; patch_x < block_size; patch_x += 8) {
                        Vec8f vec_current = Vec8f().load_a(current_patchp);
                        Vec8f vec_neighbour = Vec8f().load(neighbour_patchp);

                        Vec8f diff = vec_current - vec_neighbour;
                        vec_error = mul_add(diff, diff, vec_error);

                        current_patchp += 8;
                        neighbour_patchp += 8;
                    }

                    neighbour_patchp += stride - block_size;
                }

                float error {horizontal_add(vec_error)};

                errors.push_back(std::make_tuple(error, bm_x, bm_y));
            }

            neighbour_patchpc += stride;
        }
    } else { // block_size % 8 != 0
        Vec8i mask { make_mask(block_size & 7) };

        for (int bm_y = top; bm_y <= bottom; ++bm_y) {
            for (int bm_x = left; bm_x <= right; ++bm_x) {
                Vec8f vec_error {0.f};

                const float * VS_RESTRICT current_patchp = current_patch;
                const float * VS_RESTRICT neighbour_patchp = neighbour_patchpc + bm_x;

                for (int patch_y = 0; patch_y < block_size; ++patch_y) {
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

                        current_patchp += block_size & 7;
                        neighbour_patchp += stride - (block_size & (-8));
                    }
                }

                float error {horizontal_add(vec_error)};

                errors.push_back(std::make_tuple(error, bm_x, bm_y));
            }

            neighbour_patchpc += stride;
        }
    }
#else
    {
        for (int bm_y = top; bm_y <= bottom; ++bm_y) {
            for (int bm_x = left; bm_x <= right; ++bm_x) {
                const float * VS_RESTRICT neighbour_patchp = neighbour_patchpc + bm_x;
                float * VS_RESTRICT current_patchp {current_patch};
                float error = 0.f;

                for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                    for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                        error += square(current_patchp[patch_x] - neighbour_patchp[patch_x]);
                    }

                    current_patchp += block_size;
                    neighbour_patchp += stride;
                }

                errors.push_back(std::make_tuple(error, bm_x, bm_y));
            }

            neighbour_patchpc += stride;
        }
    }
#endif

    std::partial_sort(errors.begin(), errors.begin() + group_size, errors.end(), [](auto a, auto b){
        return std::get<0>(a) < std::get<0>(b);
    });

    float * VS_RESTRICT denoising_patchp = denoising_patch;

#ifdef __AVX2__
    if (block_size == 8) {
        constexpr int block_size8 {8};

        for (int i = 0; i < group_size; ++i) {
            auto [error, bm_x, bm_y] = errors[i];

            if constexpr (fast) {
                denoised[bm_y * width + bm_x] = true;
            }

            const float * VS_RESTRICT src_patchp = srcp + bm_y * stride + bm_x;

            float * VS_RESTRICT mean_patchp {nullptr};
            if constexpr (residual) {
                mean_patchp = mean_patch;
            }

            for (int patch_y = 0; patch_y < block_size8; ++patch_y) {
                Vec8f vec_src = Vec8f().load(src_patchp);
                vec_src.store_a(denoising_patchp);
                src_patchp += stride;
                denoising_patchp += 8;

                if constexpr (residual) {
                    Vec8f vec_mean = Vec8f().load_a(mean_patchp);
                    vec_mean += vec_src;
                    vec_mean.store_a(mean_patchp);
                    mean_patchp += 8;
                }
            }
        }
    } else if ((block_size & 7) == 0) { // block_size % 8 == 0
        for (int i = 0; i < group_size; ++i) {
            auto [error, bm_x, bm_y] = errors[i];

            if constexpr (fast) {
                denoised[bm_y * width + bm_x] = true;
            }

            const float * VS_RESTRICT src_patchp = srcp + bm_y * stride + bm_x;

            float * VS_RESTRICT mean_patchp {nullptr};
            if constexpr (residual) {
                mean_patchp = mean_patch;
            }

            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                for (int patch_x = 0; patch_x < block_size; patch_x += 8) {
                    Vec8f vec_src = Vec8f().load(src_patchp);
                    vec_src.store_a(denoising_patchp);
                    src_patchp += 8;
                    denoising_patchp += 8;

                    if constexpr (residual) {
                        Vec8f vec_mean = Vec8f().load_a(mean_patchp);
                        vec_mean += vec_src;
                        vec_mean.store_a(mean_patchp);
                        mean_patchp += 8;
                    }
                }

                src_patchp += stride - block_size;
            }
        }
    } else { // block_size % 8 != 0
        Vec8i mask { make_mask(block_size & 7) };

        for (int i = 0; i < group_size; ++i) {
            auto [error, bm_x, bm_y] = errors[i];

            if constexpr (fast) {
                denoised[bm_y * width + bm_x] = true;
            }

            const float * VS_RESTRICT src_patchp = srcp + bm_y * stride + bm_x;

            float * VS_RESTRICT mean_patchp {nullptr};
            if constexpr (residual) {
                mean_patchp = mean_patch;
            }

            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                for (int patch_x = 0; patch_x < (block_size & (-8)); patch_x += 8) {
                    Vec8f vec_src = Vec8f().load(src_patchp);
                    vec_src.store(denoising_patchp);
                    src_patchp += 8;
                    denoising_patchp += 8;

                    if constexpr (residual) {
                        Vec8f vec_mean = Vec8f().load(mean_patchp);
                        vec_mean += vec_src;
                        vec_mean.store(mean_patchp);
                        mean_patchp += 8;
                    }
                }

                {
                    Vec8f vec_src = _mm256_maskload_ps(src_patchp, mask);
                    vec_src.store(denoising_patchp); // denoising_patch is padded 
                    src_patchp += stride - (block_size & (-8));
                    denoising_patchp += block_size & 7;

                    if constexpr (residual) {
                        Vec8f vec_mean = Vec8f().load(mean_patchp);
                        vec_mean += vec_src;
                        vec_mean.store(mean_patchp); // mean_patch is padded 
                        mean_patchp += block_size & 7;
                    }
                }
            }

            denoising_patchp += svd_lda - square(block_size);
        }
    }
#else
    for (int i = 0, index = 0; i < group_size; ++i) {
        auto [error, bm_x, bm_y] = errors[i];

        if constexpr (fast) {
            denoised[bm_y * width + bm_x] = true;
        }

        const float * VS_RESTRICT src_patchp = srcp + bm_y * stride + bm_x;

        float * VS_RESTRICT mean_patchp {nullptr};
        if constexpr (residual) {
            mean_patchp = mean_patch;
        }

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                float src_val = src_patchp[patch_x];

                denoising_patchp[patch_x] = src_val;

                if constexpr (residual) {
                    mean_patchp[patch_x] += src_val;
                }
            }

            src_patchp += stride;
            denoising_patchp += block_size;
        }

        denoising_patchp += svd_lda - square(block_size);
    }
#endif
}

template<bool residual, bool fast>
static inline void bm_full(const float * VS_RESTRICT src, 
    int width, int x, int y, int top, int bottom, int left, int right, int block_size, 
    int stride, float * VS_RESTRICT denoising_patch, float * VS_RESTRICT mean_patch, 
    int svd_lda, bool * VS_RESTRICT denoised) noexcept {

    src += top * stride;

    if constexpr (fast) {
        denoised += top * width;
    }

    for (int bm_y = top; bm_y <= bottom; ++bm_y) {
        for (int bm_x = left; bm_x <= right; ++bm_x) {
            if constexpr (fast) {
                denoised[bm_x] = true;
            }

            const float * VS_RESTRICT srcp = src + bm_x;
            float * VS_RESTRICT mean_patchp {mean_patch};

            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                    float src_val = srcp[patch_x];

                    denoising_patch[patch_x] = src_val;

                    if constexpr (residual) {
                        mean_patchp[patch_x] += src_val;
                    }
                }

                srcp += stride;
                denoising_patch += block_size;
                mean_patchp += block_size;
            }

            denoising_patch += svd_lda - square(block_size);
        }

        src += stride;

        if constexpr (fast) {
            denoised += width;
        }
    }
}

static inline void bm_post(float * VS_RESTRICT mean_patch, float * VS_RESTRICT denoising_patch, 
    int block_size, int group_size, int num_neighbours, int svd_lda) noexcept {

    // substract group mean
    for (int i = 0; i < square(block_size); ++i) {
        mean_patch[i] /= VSMIN(num_neighbours, group_size);
    }

    for (int i = 0; i < VSMIN(num_neighbours, group_size); ++i) {
        for (int j = 0; j < square(block_size); ++j) {
            denoising_patch[j] -= mean_patch[j];
        }

        denoising_patch += svd_lda;
    }
}

template<bool residual>
static inline WnnmInfo patch_estimation(float * VS_RESTRICT wdst, float * VS_RESTRICT weight, float & adaptive_weight,  
    float sigma, 
    int block_size, int group_size, int num_neighbours, float * VS_RESTRICT denoising_patch, 
    float * VS_RESTRICT mean_patch, int svd_lda, 
    float * VS_RESTRICT svd_s, float * VS_RESTRICT svd_u, int svd_ldu, 
    float * VS_RESTRICT svd_vt, int svd_ldvt, float * VS_RESTRICT svd_work, int svd_lwork, 
    int * VS_RESTRICT svd_iwork, bool adaptive_aggregation) noexcept {

    int m = square(block_size);
    int n = VSMIN(num_neighbours, group_size);

    int svd_info = LAPACKE_sgesdd_work(LAPACK_COL_MAJOR, 'S', m, n, denoising_patch, svd_lda, 
        svd_s, svd_u, svd_ldu, svd_vt, svd_ldvt, svd_work, svd_lwork, svd_iwork);

    if (svd_info != 0) {
        return WnnmInfo::FAILURE;
    }

    // WNNP with parameter epsilon ignored
    float constant = 8.f * sqrtf(2.f) * sqrtf(static_cast<float>(n)) * static_cast<float>(square(sigma));

    int k = 1;
    if constexpr (residual) {
        k = 0;
    }

    for ( ; k < VSMIN(m, n); ++k) { 
        float s = svd_s[k];
        float tmp = square(s) - constant;
        if (tmp > 0.f) {
            svd_s[k] = (s + sqrtf(tmp)) * 0.5f;
        } else {
            break;
        }
    }

    if constexpr (residual) {
        if (k == 0) {
            // simply copy the mean_patch
            return WnnmInfo::EMPTY;
        }
    }

    if (adaptive_aggregation) {
        adaptive_weight = 1.f / k;
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

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, 
        m, n, k, 1.0f, svd_u, svd_ldu, svd_vt, svd_ldvt, 0.f, 
        denoising_patch, svd_lda);

    if constexpr (residual) {
        for (int i = 0; i < VSMIN(num_neighbours, group_size); ++i) {
            for (int patch_i = 0; patch_i < square(block_size); ++patch_i) {
                denoising_patch[patch_i] += mean_patch[patch_i];
            }

            denoising_patch += svd_lda;
        }
    }

    return WnnmInfo::SUCCESS;
}

static void patch_estimation_skip1_sorted(float * VS_RESTRICT wdst, float * VS_RESTRICT weight, 
    const float * VS_RESTRICT src, 
    const std::vector<std::tuple<float, int, int>> & errors, int width, 
    int stride, int block_size, int group_size) noexcept {

    for (int i = 0; i < group_size; ++i) {
        auto [error, bm_x, bm_y] = errors[i];

        const float * VS_RESTRICT srcp = src + bm_y * stride + bm_x;
        float * VS_RESTRICT wdstp = wdst + bm_y * width + bm_x;
        float * VS_RESTRICT weightp = weight + bm_y * width + bm_x;

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                wdstp[patch_x] += srcp[patch_x];
                weightp[patch_x] += 1.f;
            }

            srcp += stride;
            wdstp += width;
            weightp += width;
        }
    }
} 

static void patch_estimation_skip1_full(float * VS_RESTRICT wdst, float * VS_RESTRICT weight, 
    const float * VS_RESTRICT src, 
    int left, int right, int top, int bottom, int width, int stride, 
    int block_size) noexcept { 

    src += top * stride;
    wdst += top * width;
    weight += top * width;

    for (int bm_y = top; bm_y <= bottom; ++bm_y) {
        for (int bm_x = left; bm_x <= right; ++bm_x) {
            const float * VS_RESTRICT srcp = src + bm_x;
            float * VS_RESTRICT wdstp = wdst + bm_x;
            float * VS_RESTRICT weightp = weight + bm_x;

            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                    wdstp[patch_x] += srcp[patch_x];
                    weightp[patch_x] += 1.f;
                }

                srcp += stride;
                wdstp += width;
                weightp += width;
            }
        }

        src += stride;
        wdst += width;
        weight += width;
    }
}

static inline void patch_estimation_skip2_sorted(float * VS_RESTRICT wdst, float * VS_RESTRICT weight, 
    const std::vector<std::tuple<float, int, int>> & errors, int width, 
    int block_size, int group_size, const float * VS_RESTRICT mean_patch) noexcept {

    for (int i = 0; i < group_size; ++i) {
        auto [error, bm_x, bm_y] = errors[i];

        const float * VS_RESTRICT mean_patchp {mean_patch};
        float * VS_RESTRICT wdstp = wdst + bm_y * width + bm_x;
        float * VS_RESTRICT weightp = weight + bm_y * width + bm_x;

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                wdstp[patch_x] += mean_patchp[patch_x];
                weightp[patch_x] += 1.f;
            }

            mean_patchp += block_size;
            wdstp += width;
            weightp += width;
        }
    }
}

static inline void patch_estimation_skip2_full(float * VS_RESTRICT wdst, float * VS_RESTRICT weight, 
    int left, int right, int top, int bottom, int width, 
    int block_size, const float * VS_RESTRICT mean_patch) noexcept {

    wdst += top * width;
    weight += top * width;

    for (int bm_y = top; bm_y <= bottom; ++bm_y) {
        for (int bm_x = left; bm_x <= right; ++bm_x) {
            const float * VS_RESTRICT mean_patchp {mean_patch};
            float * VS_RESTRICT wdstp = wdst + bm_x;
            float * VS_RESTRICT weightp = weight + bm_x;

            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                    wdstp[patch_x] += mean_patchp[patch_x];
                    weightp[patch_x] += 1.f;
                }

                mean_patchp += block_size;
                wdstp += width;
                weightp += width;
            }
        }

        wdst += width;
        weight += width;
    }
}

static inline void col2im_sorted(float * VS_RESTRICT wdst, float * VS_RESTRICT weight, 
    const std::vector<std::tuple<float, int, int>> & errors, int width, 
    int block_size, int group_size, const float * VS_RESTRICT denoising_patch, 
    int svd_lda, float adaptive_weight) noexcept {

    for (int i = 0; i < group_size; ++i) {
        auto [error, bm_x, bm_y] = errors[i];

        float * VS_RESTRICT wdstp = wdst + bm_y * width + bm_x;
        float * VS_RESTRICT weightp = weight + bm_y * width + bm_x;

        for (int patch_y = 0; patch_y < block_size; ++patch_y) {
            for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                wdstp[patch_x] += denoising_patch[patch_x] * adaptive_weight;
                weightp[patch_x] += adaptive_weight;
            }

            wdstp += width;
            weightp += width;
            denoising_patch += block_size;
        }

        denoising_patch += svd_lda - square(block_size);
    }
}

static inline void col2im_full(float * VS_RESTRICT wdst, float * VS_RESTRICT weight, 
    int top, int bottom, int left, int right, int width, 
    int block_size, const float * VS_RESTRICT denoising_patch, 
    int svd_lda, float adaptive_weight) noexcept {

    wdst += top * width;
    weight += top * width;

    for (int bm_y = top; bm_y <= bottom; ++bm_y) {
        for (int bm_x = left; bm_x <= right; ++bm_x) {
            float * VS_RESTRICT wdstp = wdst + bm_x;
            float * VS_RESTRICT weightp = weight + bm_x;

            for (int patch_y = 0; patch_y < block_size; ++patch_y) {
                for (int patch_x = 0; patch_x < block_size; ++patch_x) {
                    wdstp[patch_x] += denoising_patch[patch_x] * adaptive_weight;
                    weightp[patch_x] += adaptive_weight;
                }

                wdstp += width;
                weightp += width;
                denoising_patch += block_size;
            }

            denoising_patch += svd_lda - square(block_size);
        }

        wdst += width;
        weight += width;
    }
}

static inline void aggregation(float * VS_RESTRICT dstp, const float * VS_RESTRICT srcp, 
    int width, int height, int stride, const float * VS_RESTRICT wdst, 
    const float * VS_RESTRICT weight) noexcept {

#ifdef __AVX2__
    const Vec8f vec_eps { FLT_EPSILON };
#endif

    for (int y = 0; y < height; ++y) {
        int x = 0;

#ifdef __AVX2__
        const float * VS_RESTRICT vec_srcp {srcp};
        const float * VS_RESTRICT vec_wdstp {wdst};
        const float * VS_RESTRICT vec_weightp {weight};
        float * VS_RESTRICT vec_dstp {dstp};

        for ( ; x < (width & (-8)); x += 8) {
            Vec8f vec_src = Vec8f().load_a(vec_srcp);
            Vec8f vec_wdst = Vec8f().load(vec_wdstp);
            Vec8f vec_weight = Vec8f().load(vec_weightp);
            Vec8f vec_dst = mul_add(vec_src, vec_eps, vec_wdst) * approx_recipr(vec_weight + vec_eps);
            vec_dst.store_a(vec_dstp);

            vec_srcp += 8;
            vec_wdstp += 8;
            vec_weightp += 8;
            vec_dstp += 8;
        }
#endif

        for ( ; x < width; ++x) {
            dstp[x] = (wdst[x] + srcp[x] * FLT_EPSILON) / (weight[x] + FLT_EPSILON);
        }

        srcp += stride;
        dstp += stride;
        wdst += width;
        weight += width;
    }
}

template<bool residual, bool fast>
static void process(const VSFrameRef * src, VSFrameRef * dst, WNNMData * d, const VSAPI * vsapi) noexcept {
    const auto threadId = std::this_thread::get_id();

#ifdef __AVX2__
    auto control_word = get_control_word();
    no_subnormals();
#endif

    Workspace workspace;
    bool init = true;

    d->workspaces_lock.lock_shared();

    try {
        const auto & const_workspaces = d->workspaces;
        workspace = const_workspaces.at(threadId);
    } catch (const std::out_of_range &) {
        init = false;
    }

    d->workspaces_lock.unlock_shared();

    if (!init) {
        workspace.init(
            d->vi->width, d->vi->height, 
            d->block_size, d->group_size, d->bm_range, d->residual, d->fast, 
            d->svd_lda, d->svd_ldu, d->svd_ldvt, d->svd_lwork
        );

        d->workspaces_lock.lock();
        d->workspaces.emplace(threadId, workspace);
        d->workspaces_lock.unlock();
    }

    float * const VS_RESTRICT wdst = workspace.wdst;
    float * const VS_RESTRICT weight = workspace.weight;
    float * const VS_RESTRICT denoising_patch = workspace.denoising_patch;

    float * VS_RESTRICT mean_patch {nullptr};
    if constexpr (residual) {
         mean_patch = workspace.mean_patch;
    }

    float * VS_RESTRICT current_patch = workspace.current_patch;

    bool * VS_RESTRICT denoised {nullptr};
    if constexpr (fast) {
        denoised = workspace.denoised;
    }

    float * const VS_RESTRICT svd_s = workspace.svd_s;
    float * const VS_RESTRICT svd_u = workspace.svd_u;
    float * const VS_RESTRICT svd_vt = workspace.svd_vt;
    float * const VS_RESTRICT svd_work = workspace.svd_work;
    int * const VS_RESTRICT svd_iwork = workspace.svd_iwork;
    const int svd_lwork = d->svd_lwork;
    const int block_size = d->block_size;
    const int block_step = d->block_step;
    const int group_size = d->group_size;
    const int bm_range = d->bm_range;
    const int svd_lda = d->svd_lda;
    const int svd_ldu = d->svd_ldu;
    const int svd_ldvt = d->svd_ldvt;
    const bool adaptive_aggregation = d->adaptive_aggregation;

    std::vector<std::tuple<float, int, int>> errors;
    if (d->group_size < square(2 * d->bm_range + 1)) {
        errors.reserve(square(2 * d->bm_range + 1));
    }

    for (int plane = 0; plane < d->vi->format->numPlanes; plane++) {
        if (d->process[plane]) {
            const int width = vsapi->getFrameWidth(src, plane);
            const int height = vsapi->getFrameHeight(src, plane);
            const int stride = vsapi->getStride(src, plane) / sizeof(float);
            const float * const srcp = reinterpret_cast<const float *>(vsapi->getReadPtr(src, plane));
            float * const VS_RESTRICT dstp = reinterpret_cast<float *>(vsapi->getWritePtr(dst, plane));

            memset(wdst, 0, sizeof(float) * height * width);
            memset(weight, 0, sizeof(float) * height * width);

            if constexpr (fast) {
                memset(denoised, 0, sizeof(bool) * height * width);
            }

            int temp_r = height - block_size;
            int temp_c = width - block_size;

            for (int y = 0;; y += block_step) {
                if (y >= temp_r + block_step) {
                    break;
                } else if (y > temp_r) {
                    y = temp_r;
                }

                const int top = VSMAX(y - bm_range, 0);
                const int bottom = VSMIN(y + bm_range, temp_r);

                for (int x = 0;; x += block_step) {
                    if (x >= temp_c + block_step) {
                        break;
                    } else if (x > temp_c) {
                        x = temp_c;
                    }

                    if constexpr (fast) {
                        if (denoised[y * width + x]) {
                            continue;
                        }
                    }

                    if constexpr (residual) {
                        memset(mean_patch, 0, sizeof(float) * square(block_size));
                    }

                    const int left = VSMAX(x - bm_range, 0);
                    const int right = VSMIN(x + bm_range, temp_c);

                    int num_neighbours = (bottom - top + 1) * (right - left + 1);

                    if (num_neighbours > group_size) {
                        bm_sorted<residual, fast>(errors, srcp, width, x, y, top, bottom, left, right, 
                            block_size, group_size, stride, denoising_patch, mean_patch, current_patch, 
                            svd_lda, denoised);
                    } else {
                        bm_full<residual, fast>(srcp, width, x, y, top, bottom, left, right, 
                            block_size, stride, denoising_patch, mean_patch, svd_lda, denoised);
                    }

                    // substract group mean
                    if constexpr (residual) {
                        bm_post(mean_patch, denoising_patch, block_size, group_size, num_neighbours, svd_lda);
                    }

                    // patch_estimation with early skipping on SVD exception
                    float adaptive_weight = 1.f;
                    WnnmInfo info = patch_estimation<residual>(wdst, weight, adaptive_weight, d->sigma[plane], 
                        block_size, group_size, num_neighbours, denoising_patch, mean_patch, svd_lda, 
                        svd_s, svd_u, svd_ldu, svd_vt, svd_ldvt, svd_work, svd_lwork, svd_iwork, 
                        adaptive_aggregation);

                    switch (info) {
                        case WnnmInfo::SUCCESS: {
                            if (num_neighbours > group_size) {
                                col2im_sorted(wdst, weight, errors, width, 
                                    block_size, group_size, denoising_patch, svd_lda, adaptive_weight);
                            } else {
                                col2im_full(wdst, weight, top, bottom, left, right, width, 
                                    block_size, denoising_patch, svd_lda, adaptive_weight);
                            }
                            break;
                        }

                        case WnnmInfo::FAILURE: {
                            if (num_neighbours > group_size) {
                                patch_estimation_skip1_sorted(wdst, weight, srcp, 
                                    errors, width, stride, block_size, group_size);
                            } else {
                                patch_estimation_skip1_full(wdst, weight, srcp, 
                                    left, right, top, bottom, width, stride, block_size);
                            }
                            break;
                        }

                        case WnnmInfo::EMPTY: {
                            if (num_neighbours > group_size) {
                                patch_estimation_skip2_sorted(wdst, weight, 
                                    errors, width, block_size, group_size, mean_patch);
                            } else {
                                patch_estimation_skip2_full(wdst, weight, 
                                    left, right, top, bottom, width, block_size, mean_patch);
                            }
                            break;
                        }
                    }
                }
            }

            aggregation(dstp, srcp, width, height, stride, wdst, weight);
        }
    }

#ifdef __AVX2__
    set_control_word(control_word);
#endif
}

static void VS_CC WNNMInit(VSMap *in, VSMap *out, void **instanceData, VSNode *node, VSCore *core, const VSAPI *vsapi) {
    WNNMData * d = static_cast<WNNMData *>(*instanceData);
    vsapi->setVideoInfo(d->vi, 1, node);
}

static const VSFrameRef *VS_CC WNNMGetFrame(int n, int activationReason, void **instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi) {
    WNNMData * d = static_cast<WNNMData *>(*instanceData);

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
    } else if (activationReason == arAllFramesReady) {
        const VSFrameRef * src = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrameRef * fr[] = { d->process[0] ? nullptr : src, d->process[1] ? nullptr : src, d->process[2] ? nullptr : src };
        const int pl[] = { 0, 1, 2 };
        VSFrameRef * dst = vsapi->newVideoFrame2(d->vi->format, d->vi->width, d->vi->height, fr, pl, src, core);

        if (d->residual) {
            if (d->fast) {
                process<true, true>(src, dst, d, vsapi);
            } else {
                process<true, false>(src, dst, d, vsapi);
            }
        } else {
            if (d->fast) {
                process<false, true>(src, dst, d, vsapi);
            } else {
                process<false, false>(src, dst, d, vsapi);
            }
        }

        vsapi->freeFrame(src);
        return dst;
    }

    return nullptr;
}

static void VS_CC WNNMFree(void *instanceData, VSCore *core, const VSAPI *vsapi) {
    WNNMData * d = static_cast<WNNMData *>(instanceData);

    vsapi->freeNode(d->node);

    for (auto & [_, workspace] : d->workspaces) {
        workspace.release();
    }

    delete d;
}

static void VS_CC WNNMCreate(const VSMap *in, VSMap *out, void *userData, VSCore *core, const VSAPI *vsapi) {
    std::unique_ptr<WNNMData> d{ new WNNMData{} };

    d->node = vsapi->propGetNode(in, "clip", 0, nullptr);
    d->vi = vsapi->getVideoInfo(d->node);

    auto set_error = [&](const std::string & error) -> void {
        vsapi->setError(out, ("WNNM: " + error).c_str());
        vsapi->freeNode(d->node);
        return ;
    };

    if (!isConstantFormat(d->vi) || d->vi->format->sampleType == stInteger ||
        (d->vi->format->sampleType == stFloat && d->vi->format->bitsPerSample != 32)
    ) {
        return set_error("only constant format 32 bit float input supported");
    }

    int error;

    int m = vsapi->propNumElements(in, "sigma");

    if (m > 0) {
        int i;

        if (m > 3) {
            m = 3;
        }

        for (i = 0; i < m; ++i) {
            d->sigma[i] = static_cast<float>(vsapi->propGetFloat(in, "sigma", i, nullptr));

            if (d->sigma[i] < 0) {
                return set_error("\"sigma\" must be positive");
            }
        }

        for (; i < 3; ++i) {
            d->sigma[i] = d->sigma[i - 1];
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            d->sigma[i] = 3.0f;
        }
    }

    for (int i = 0; i < 3; ++i) {
        if (d->sigma[i] == 0.f) {
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

    d->svd_lda = m16(square(d->block_size));
    d->svd_ldu = m16(square(d->block_size));
    d->svd_ldvt = m16(VSMIN(square(d->block_size), VSMIN(d->group_size, square(2 * d->bm_range + 1))));

    d->residual = !!vsapi->propGetInt(in, "residual", 0, &error);
    if (error) {
        d->residual = false;
    }

    d->adaptive_aggregation = !!vsapi->propGetInt(in, "adaptive_aggregation", 0, &error);
    if (error) {
        d->adaptive_aggregation = true;
    }

    d->fast = !!vsapi->propGetInt(in, "fast", 0, &error);
    if (error) {
        d->fast = false;
    }
    d->fast = d->fast && (d->bm_range >= d->block_step);

    VSCoreInfo core_info;
    vsapi->getCoreInfo2(core, &core_info);
    auto numThreads = core_info.numThreads;

    d->workspaces.reserve(numThreads);

    int svd_m = square(d->block_size);
    int svd_n = VSMIN(d->group_size, square(2 * d->bm_range + 1));
    d->svd_lwork = VSMIN(svd_m, svd_n) * (6 + 4 * VSMIN(svd_m, svd_n)) + VSMAX(svd_m, svd_n);

    if (d->fast) {
        LAPACKE_set_nancheck(0);
    }

    vsapi->createFilter(in, out, "WNNM", WNNMInit, WNNMGetFrame, WNNMFree, fmParallel, 0, d.release(), core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit(VSConfigPlugin configFunc, VSRegisterFunction registerFunc, VSPlugin *plugin) {
    configFunc(
        "com.WolframRhodium.WNNM", 
        "wnnm", "Weighted nuclear norm minimization denoiser", 
        VAPOURSYNTH_API_VERSION, 1, plugin
    );

    registerFunc("WNNM",
        "clip:clip;"
        "sigma:float[]:opt;"
        "block_size:int:opt;"
        "block_step:int:opt;"
        "group_size:int:opt;"
        "bm_range:int:opt;"
        "residual:int:opt;"
        "adaptive_aggregation:int:opt;"
        "fast:int:opt;",
        WNNMCreate, nullptr, plugin
    );

    auto getVersion = [](const VSMap *, VSMap * out, void *, VSCore *, const VSAPI *vsapi) {
        vsapi->propSetData(out, "version", VERSION, -1, paReplace);
    };
    registerFunc("Version", "", getVersion, nullptr, plugin);
}
