# clpeak

[![Google Play](https://upload.wikimedia.org/wikipedia/commons/7/78/Google_Play_Store_badge_EN.svg)](https://play.google.com/store/apps/details?id=kr.clpeak)
[![Build](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml)

**clpeak &mdash; "Compute Latency PEAK".** A synthetic micro-benchmark that measures the peak achievable performance of GPU compute devices. It exercises tight vector / MAD / MMA loops and vendor-SDK GEMM libraries (cuBLASLt on NVIDIA, MPS on Apple) to expose what the hardware is capable of &mdash; from raw ALU peaks to near-vendor-advertised matrix throughput.

clpeak began as an OpenCL-only tool. It has since been repurposed into a multi-backend benchmark, shipping multiple interchangeable backends &mdash; OpenCL, Vulkan, CUDA, ROCm/HIP, Metal, and oneAPI/SYCL &mdash; running back-to-back on the same hardware, so cross-stack differences (driver lowering, instruction scheduling, extension exposure) become visible alongside the raw peak numbers.

## Sample output

Condensed, peak-revealing lines straight from the tool. Numbers below are real runs.

Apple M1 Pro, Metal backend:

```text
Backend: Metal
  Device 0: Apple M1 Pro

    Single-precision compute (GFLOPS)
      float    : 4487.56
      half     : 4989.62

    simdgroup_matrix fp16xfp16+fp32 8x8x8 (TFLOPS)
      simdgroup_fp16 : 15.84

    MPS GEMM peak (TFLOPS)
      fp32     : 4.09
      fp16     : 3.97

    Global memory bandwidth (GBPS)
      float    : 184.49

    Local memory bandwidth (GBPS)
      float    : 2938.50

    Image memory bandwidth (GBPS)
      rgba32f  : 162.67
```

NVIDIA RTX 5060, CUDA backend:

```text
Backend: CUDA
  Device 0: NVIDIA GeForce RTX 5060

    Single-precision compute (GFLOPS)
      float    : 21100.20
      half     : 21077.21
      bf16     : 20042.78

    WMMA fp16xfp16+fp32 16x16x16 (TFLOPS)
      wmma_fp16 : 165.90

    WMMA int8xint8+int32 16x16x16 (TOPS)
      wmma_int8 : 325.18

    FP8(E4M3) mma.sync m16n8k32+fp32 (TFLOPS)
      fp8_e4m3 : 85.08

    MXFP4(E2M1) mma.sync m16n8k64+fp32 (TFLOPS)
      mxf4_e2m1 : 324.54
    NVFP4(E2M1) mma.sync m16n8k64+fp32 (TFLOPS)
      nvf4_e2m1 : 327.00

    MXFP4 mma.sp 2:4 sparsity m16n8k128+fp32 (TFLOPS)
      mxf4_sparse : 630.37
    NVFP4 mma.sp 2:4 sparsity m16n8k128+fp32 (TFLOPS)
      nvf4_sparse : 630.45

    INT8 dot-product compute (__dp4a) (GOPS)
      int8_dp8 : 33587.22

    cuBLASLt GEMM peak (TFLOPS)
      fp16     : 77.54
      bf16     : 41.14
      fp8_e4m3 : 143.89
      nvf4_e2m1 : 298.99
    cuBLASLt GEMM peak (TOPS)
      int8     : 149.18

    Global memory bandwidth (GBPS)
      float4   : 418.82

    Local memory bandwidth (GBPS)
      float4   : 9118.74

    Kernel launch latency (US)
      roundtrip : 6.24
```

AMD Instinct MI300X, ROCm backend:

```text
Backend: ROCm
  Device 0: AMD Instinct MI300X

    Single-precision compute (GFLOPS)
      float    : 134624.47
      half     : 151388.55
      double   : 62886.77
      bf16     : 117266.34

    MFMA fp16xfp16+fp32 16x16x16 (TFLOPS)
      mfma_fp16 : 1128.18
    MFMA bf16xbf16+fp32 16x16x16 (TFLOPS)
      mfma_bf16 : 1124.29
    MFMA fp8xfp8+fp32 16x16x32 (TFLOPS)
      mfma_fp8 : 2166.78
    MFMA int8xint8+int32 16x16x32 (TOPS)
      mfma_int8 : 2339.26

    Sparse MFMA fp16 2:4 16x16x32 (TFLOPS)
      smfmac_fp16 : 2154.45
    Sparse MFMA fp8 2:4 16x16x64 (TFLOPS)
      smfmac_fp8 : 4138.86
    Sparse MFMA int8 2:4 16x16x64 (TOPS)
      smfmac_int8 : 4499.68

    rocBLAS GEMM peak (TFLOPS)
      fp32     : 129.70
      fp64     : 100.48
      fp16     : 840.05
    hipBLASLt FP8 GEMM peak (TFLOPS)
      fp8_e4m3 : 1588.02

    Global memory bandwidth (GBPS)
      float4   : 3577.33

    Local memory bandwidth (GBPS)
      float4   : 65024.37

    Kernel launch latency (US)
      roundtrip : 8.66
```

## Building

```console
git submodule update --init --recursive --remote
cmake -S . -B build
cmake --build build -j
./build/clpeak
```

Optional backends are auto-detected and enabled when their SDK is found. To opt out of a backend at configure time:

```console
cmake -S . -B build -DCLPEAK_ENABLE_CUDA=OFF
cmake -S . -B build -DCLPEAK_ENABLE_VULKAN=OFF -DCLPEAK_ENABLE_METAL=OFF
cmake -S . -B build -DCLPEAK_ENABLE_ONEAPI=ON -DCMAKE_CXX_COMPILER=icpx
```

> **oneAPI/SYCL note:** the oneAPI backend must be configured with `-DCMAKE_CXX_COMPILER=icpx` (the DPC++ compiler). SYCL kernels are compiled inline by the host compiler, so configuring with a non-`icpx` compiler silently skips the backend.

| CMake option | Default | Effect when `OFF` |
|---|---|---|
| `CLPEAK_ENABLE_OPENCL` | `ON` | Skip OpenCL backend |
| `CLPEAK_ENABLE_VULKAN` | `ON` | Skip Vulkan even if Vulkan SDK is present |
| `CLPEAK_ENABLE_CUDA` | `ON` | Skip CUDA even if CUDA Toolkit is present |
| `CLPEAK_ENABLE_ROCM` | `ON` | Skip ROCm/HIP even if ROCm SDK is present |
| `CLPEAK_ENABLE_METAL` | `ON` | Skip Metal/MPS even on Apple silicon |
| `CLPEAK_ENABLE_ONEAPI` | `ON` | Skip oneAPI/SYCL |

## What it measures

| Test | OpenCL | Vulkan | CUDA | ROCm/HIP | Metal | oneAPI/SYCL |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Global memory bandwidth | &check; | &check; | &check; | &check; | &check; | &check; |
| Local / shared memory bandwidth | &check; | &check; | &check; | &check; | &check; | &check; |
| Image / texture bandwidth | &check; | &check; | &check; | &check; | &check; | &check; |
| Transfer bandwidth (host&harr;device) | &check; | &check; | &check; | &check; | &mdash; | &check; |
| Compute SP / HP / DP / MP / BF16 | &check; | &check; | &check; | &check; | &check; | &check; |
| Compute INT (int32) | &check; | &check; | &check; | &check; | &mdash; | &check; |
| Compute INT24 / INT8 / INT16 | &check; | &mdash; | &mdash; | &mdash; | &mdash; | &mdash; |
| INT8 dot-product (DP4a) | &check; | &check; | &check; | &check; | &mdash; | &check; |
| Tensor / matrix-engine MMA (`--wmma`, `--simdgroup-matrix`, `--coopmat`, `--rocwmma`, `--mfma`, `--joint-matrix`) | &mdash; | coopmat fp32/fp16/bf16/int8/fp8 | WMMA fp16/bf16/int8 + FP8 mma.sync + FP4/MXFP4/NVFP4 mma.sync | rocWMMA fp16/int8 + raw MFMA fp16/bf16/int8/fp8/mxfp4 | simdgroup_matrix fp16/bf16 | joint_matrix bf16/int8 (XMX) |
| Vendor-SDK GEMM peak (`--cublas`, `--rocblas`, `--mps-gemm`, `--onemkl`) | &mdash; | &mdash; | cuBLASLt: fp32/tf32/fp16/bf16/fp8&#x2011;e4m3/fp8&#x2011;e5m2/int8/int4 | rocBLAS: fp32/fp64/fp16 + hipBLASLt: fp8&#x2011;e4m3/fp8&#x2011;e5m2 | MPS: fp32/fp16/bf16 | oneMKL: fp32/fp64/fp16 |
| Kernel launch latency | &check; | &check; | &check; | &check; | &check; | &check; |

## CLI

`./clpeak --help` prints the full flag list. The CLI is uniform across backends: the same global, test-selection, and output flags work whether OpenCL, Vulkan, CUDA, ROCm/HIP, Metal, or oneAPI/SYCL is doing the work.

```console
./clpeak                              # run every test on every available backend
./clpeak --single-precision-compute   # run only single-precision compute, on every backend
./clpeak --metal                      # run only one backend
./clpeak --cuda --vulkan              # combine multiple --<backend> flags
./clpeak --rocm                       # run only the ROCm/HIP backend
./clpeak --oneapi                     # run only the oneAPI/SYCL backend
./clpeak --no-opencl --no-cuda        # or skip the ones you don't want
./clpeak --wmma                       # CUDA tensor-core tests (hand-rolled WMMA)
./clpeak --cublas                     # CUDA vendor-SDK GEMM peak (cuBLASLt, all dtypes)
./clpeak --rocwmma                    # AMD matrix-engine tests (hand-rolled rocWMMA)
./clpeak --mfma                       # AMD raw MFMA matrix-core peak (fp16/bf16/int8/fp8/mxfp4) + 2:4 sparse (smfmac)
./clpeak --rocblas                    # AMD vendor-SDK GEMM peak (rocBLAS fp32/fp64/fp16 + hipBLASLt fp8)
./clpeak --simdgroup-matrix           # Apple matrix-engine tests (hand-rolled simdgroup_matrix)
./clpeak --mps-gemm                   # Apple vendor-SDK GEMM peak (MPS / MPSGraph)
./clpeak --joint-matrix               # Intel XMX matrix-engine tests (hand-rolled joint_matrix)
./clpeak --onemkl                     # Intel vendor-SDK GEMM peak (oneMKL)
./clpeak --coopmat                    # Vulkan tensor-core tests
./clpeak --xml-file out.xml           # save results (also --json-file / --csv-file)
./clpeak --compare baseline.json      # diff this run against a saved baseline JSON
./clpeak --list-devices               # enumerate devices for every backend, no benchmarks
```

`--compare baseline.json` re-runs the selected tests and prints each result next to the value recorded in `baseline.json` (saved earlier with `--json-file`), so regressions or driver/SDK upgrades show up as a per-test delta instead of two outputs you have to eyeball side by side.

### Selecting a specific device

Multi-GPU machines pick devices per-backend. Each index flag takes one index or
a comma-separated list; omitting it runs every device in that backend:

```console
./clpeak --cl-platform 0 --cl-device 1   # OpenCL platform/device pair
./clpeak --vk-device 0,1                 # Vulkan physical-device indices (subset)
./clpeak --cuda-device 0,2               # CUDA device ordinals
./clpeak --rocm-device 0                 # ROCm/HIP device ordinal
./clpeak --mtl-device 0                  # Metal device index
./clpeak --oneapi-device 0               # oneAPI/SYCL device index
```

## For AI agents

This tree is documented with `AGENTS.md` files. Start at the
[root `AGENTS.md`](AGENTS.md) for architecture, directory map, build
instructions, and the self-maintaining documentation conventions.
Every subdirectory has its own `AGENTS.md` with local details — open
the one closest to the code you're touching.
