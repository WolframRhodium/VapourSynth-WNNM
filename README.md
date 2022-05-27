# VapourSynth-WNNM
[Weighted Nuclear Norm Minimization](https://ieeexplore.ieee.org/document/6909762) Denoiser for VapourSynth.

## Usage
Prototype:

`core.wnnm.WNNM(clip clip[, float[] sigma = 3.0, int block_size = 8, int block_step = 8, int group_size = 8, int bm_range = 7, int radius = 0, int ps_num = 2, int ps_range = 4, bool residual = false, bool adaptive_aggregation = true])`

Requires floating point input. Each plane is denoised separately.

## Compilation
[oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) is required. [Vector class library](https://github.com/vectorclass/version2) is also required when compiling with AVX2.

```bash
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release \
-D MKL_LINK=static -D MKL_THREADING=sequential -D MKL_INTERFACE=lp64

cmake --build build

cmake --install build
```

