# VapourSynth-WNNM
[Weighted Nuclear Norm Minimization](https://ieeexplore.ieee.org/document/6909762) Denoiser for VapourSynth.

## Description
`WNNM` is a denoising algorithm based on block-matching and weighted nuclear norm minimization.

Block matching, which is popularized by `BM3D`, finds similar blocks and then stacks together in a 3-D group. The similarity between these blocks allows details to be preserved during denoising.

In contrast to `BM3D`, which denoises the 3-D group based on frequency domain filtering, `WNNM` utilizes weighted nuclear norm minimization, a kind of low rank matrix approximation. Because of this, `WNNM` exhibits less blocking and ringing artifact compared to `BM3D`, but the computational complexity is much higher. This stage is called collaborative filtering in `BM3D`.

## Usage
Prototype:

`core.wnnm.WNNM(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, bool residual = false, bool adaptive_aggregation = true, clip rclip = None])`

- clip:

    The input clip. Must be of 32 bit float format. Each plane is denoised separately.

- sigma:

    Denoising strength of each plane.

- block_size, block_step, group_size, bm_range, radius, ps_num, ps_range:

    Same as those in [VapourSynth-BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D).

- residual:

    Whether to center blocks before collaborative filtering. Default: `False`.

- adaptive_aggregation:

    Whether to aggregate blocks adaptively. Default: `True`.

- rclip:

    Reference clip for block matching. Must be of the same dimensions and format as `clip`.

## Implementation
Default values of `block_size`, `block_step`, `group_size` are modified for acceleration.

For spatial denoising, the block-matching implemented is the same as the official implementation, which is similar to that of `BM3D` without setting a threshold on whether dissimilar blocks should be included in the 3-D group. This is the same strategy implemented in [VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA) but not in [VapourSynth-BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D).

For temporal denoising, this implementation utilizes the same predictive search proposed by `V-BM3D`, which is closer to [VapourSynth-BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D) (without dissimilar block thresholding) than [VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA). The later one implemented a modified temporal predictive search that may finds multiple instances of the same similar block for acceleration.

During collaborative filtering, the official WNNM implementation centers blocks in the 3-D group. This is controlled by the `residual` parameter and is off by default. The major singular value is untouched when `residual` is off.

**Note**: Because of WNNM and the modification, the maximum denoising effect achieved is the best rank-one approximation of the 3-D group when `residual` is off, or the mean of the group when `residual` is on, which may not be enough for strong noises. The official implementation uses iterative regularization, which can be easily implemented as
```python
for i in range(num_iterations):
    if i == 0:
        previous = source
    elif i == 1:
        previous = denoised
    else:
        previous = core.std.Expr([source, previous, denoised], "x y - {factor} * z +".format(factor=0.1))
    denoised = WNNM(previous)
# output: `denoised`
```

The similar blocks are weightedly aggregated by the inverse of the number of non-zero singular values after WNNM, inspired by `BM3D`. This is controlled by the `adaptive_aggregation` parameter and is on by default.

The block-matching can be guided by an oracle reference clip `rclip` in the same manner as `ref` for `BM3D`. The collaborative filtering is not guided, unlike `BM3D`.

## Compilation
[oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) is required. [Vector class library](https://github.com/vectorclass/version2) is also required when compiling with AVX2.

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release \
-D MKL_LINK=static -D MKL_THREADING=sequential -D MKL_INTERFACE=lp64

cmake --build build

cmake --install build
```

Example build process can be found in [workflows](https://github.com/WolframRhodium/VapourSynth-WNNM/tree/master/.github/workflows).

## Reference
1. S. Gu, L. Zhang, W. Zuo and X. Feng, "[Weighted Nuclear Norm Minimization with Application to Image Denoising](https://ieeexplore.ieee.org/document/6909762)," 2014 IEEE Conference on Computer Vision and Pattern Recognition, 2014, pp. 2862-2869.

2. K. Dabov, A. Foi, V. Katkovnik and K. Egiazarian, "[Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering](https://ieeexplore.ieee.org/document/4271520)," in IEEE Transactions on Image Processing, vol. 16, no. 8, pp. 2080-2095, Aug. 2007.

3. K. Dabov, A. Foi and K. Egiazarian, "[Video denoising by sparse 3D transform-domain collaborative filtering](https://ieeexplore.ieee.org/document/7098781)," 2007 15th European Signal Processing Conference, 2007, pp. 145-149.

4. [Official implementation](https://www4.comp.polyu.edu.hk/~cslzhang/code/WNNM_code.zip)

5. [VapourSynth-BM3D](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D)

6. [VapourSynth-BM3DCUDA](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA)
