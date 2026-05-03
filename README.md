# clpeak

[![Build](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml)

A synthetic micro-benchmark that measures the peak achievable performance of GPU compute devices. It exercises tight vector / MAD / MMA loops and vendor-SDK GEMM libraries (cuBLASLt on NVIDIA, MPS on Apple) to expose what the hardware is capable of &mdash; from raw ALU peaks to near-vendor-advertised matrix throughput.

clpeak began as an OpenCL-only tool. It now ships four interchangeable backends &mdash; OpenCL, Vulkan, CUDA, and Metal &mdash; running back-to-back on the same hardware, so cross-stack differences (driver lowering, instruction scheduling, extension exposure) become visible alongside the raw peak numbers.

## Sample output

NVIDIA RTX 5060, condensed:

```text
=== CUDA backend ===

  Single-precision compute (GFLOPS)
    float : 17830.03

  BF16 compute bf16xbf16+fp32 (GFLOPS)
    bf16 : 19653.56

  INT8 dot-product compute (__dp4a) (GOPS)
    int8_dp4 : 33758.77

  WMMA fp16xfp16+fp32 16x16x16 (TFLOPS)
    wmma_fp16 : 165.93

  WMMA int8xint8+int32 16x16x16 (TOPS)
    wmma_int8 : 327.36

  FP8(E4M3) mma.sync m16n8k32+fp32 (TFLOPS)
    fp8_e4m3 : 85.10

  cuBLASLt GEMM peak (TFLOPS)
    fp32     : 15.35
    tf32     : 20.94
    fp16     : 79.94
    bf16     : 42.20
    fp8_e4m3 : 161.61
    fp8_e5m2 : 161.60

  cuBLASLt GEMM peak (TOPS)
    int8     : 161.33
    int4     : unsupported on sm_120

  Global memory bandwidth (GBPS)
    float   : 390.90

  Local memory bandwidth (GBPS)
    float4 : 9139.50

  Atomic throughput (GOPS)
    global : 170.90
    local  : 1322.27

  Kernel launch latency (us)
    noop : 2.35
```

Apple M1 Pro, condensed:

```text
=== Metal backend ===

  Single-precision compute (GFLOPS)
    float : 4487.11

  simdgroup_matrix fp16xfp16+fp32 8x8x8 (TFLOPS)
    simdgroup_fp16 : 15.84

  MPS GEMM peak (TFLOPS)
    fp32 : 4.31
    fp16 : 3.97
    bf16 : unsupported on this device

  Global memory bandwidth (GBPS)
    float   : 180.87

  Local memory bandwidth (GBPS)
    float4 : 2705.21

  Image memory bandwidth (GBPS)
    float4 : 499.71

  Atomic throughput (GOPS)
    global : 24.64
    local  : 256.45
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
```

| CMake option | Default | Effect when `OFF` |
|---|---|---|
| `CLPEAK_ENABLE_VULKAN` | `ON` | Skip Vulkan even if Vulkan SDK is present |
| `CLPEAK_ENABLE_CUDA` | `ON` | Skip CUDA even if CUDA Toolkit is present |
| `CLPEAK_ENABLE_METAL` | `ON` | Skip Metal/MPS even on Apple silicon |

## Backends

| Backend | Default | Compile path | Targets |
|---|---|---|---|
| **OpenCL** | always built | C++ host + .cl strings | OpenCL 1.2 baseline; 3.0 features when headers expose them |
| **Vulkan** | on, if Vulkan SDK present | GLSL .comp &rarr; SPIR-V at configure time | Vulkan 1.1+ |
| **CUDA** | on, if CUDA Toolkit present | .cu source embedded as raw strings, NVRTC at runtime; cuBLASLt for GEMM peak | CUDA driver API + NVRTC + cuBLASLt (all part of CUDA Toolkit) |
| **Metal** | on, on Apple silicon | .metal source embedded as raw strings, runtime compile; MPS / MPSGraph for GEMM peak | Apple7 (M1) and newer (MPSGraph bf16 requires Apple9 / M3+) |

A backend is silently skipped at runtime if its loader / driver / device is missing, so a single binary stays portable across boxes. Force-disable at runtime with `--no-opencl`, `--no-vulkan`, `--no-cuda`, `--no-metal`.

## What it measures

| Test | Unit | OpenCL | Vulkan | CUDA | Metal |
|---|---|:---:|:---:|:---:|:---:|
| Global memory bandwidth | GB/s | &check; | &check; | &check; | &check; |
| Local / shared memory bandwidth | GB/s | &check; | &check; | &check; | &check; |
| Image / texture bandwidth | GB/s | &check; | &check; | &check; | &check; |
| Transfer bandwidth (host&harr;device) | GB/s | &check; | &check; | &check; | &mdash; |
| Compute SP / HP / DP / MP / BF16 | GFLOPS | &check; | &check; | &check; | &check; |
| Compute INT / INT24 / INT8 / INT16 | GOPS | &check; | &mdash; | &mdash; | &mdash; |
| INT8 dot-product (DP4a) | GOPS | &check; | &check; | &check; | &check; (emul) |
| Packed INT4 (emulated) | GOPS | &check; | &check; | &check; | &check; |
| Tensor / matrix-engine MMA (`--wmma`, `--simdgroup-matrix`, `--coop-matrix`) | TFLOPS / TOPS | &mdash; | coopmat fp16/bf16/int8/fp8 | WMMA fp16/bf16/int8 + FP8 mma.sync | simdgroup_matrix fp16/bf16 |
| Vendor-SDK GEMM peak (`--cublas`, `--mps-gemm`) | TFLOPS / TOPS | &mdash; | &mdash; | cuBLASLt: fp32/tf32/fp16/bf16/fp8&#x2011;e4m3/fp8&#x2011;e5m2/int8/int4 | MPS: fp32/fp16/bf16 |
| Atomic throughput (global + local) | GOPS | &check; | &check; | &check; | &check; |
| Kernel launch latency | &mu;s | &check; | &check; | &check; | &check; |

The vendor-SDK GEMM tests (`--cublas`, `--mps-gemm`) measure a different point than the hand-rolled MMA kernels above: they use cuBLASLt / MPS internally, which contain the same hand-tuned tiling and swizzling that NVIDIA / Apple use to publish their own peak numbers. The hand-rolled WMMA / simdgroup_matrix tests benchmark the raw instruction throughput; the vendor-SDK tests benchmark the achievable system GEMM peak including occupancy, memory staging, and algorithm selection.

## Cross-backend comparison

Running multiple backends on the same device exposes driver- and lowering-quality deltas that a single-stack benchmark cannot:

- NVIDIA RTX 5060: OpenCL image bandwidth comes in at ~1/10 the Vulkan or CUDA equivalent &mdash; driver-side image-fetch lowering issue, not a hardware limit.
- NVIDIA RTX 5060: Vulkan local-atomic throughput is ~1/2 the OpenCL or CUDA rate &mdash; NVIDIA's Vulkan SPIR-V atomic lowering takes a heavier-ordering path.
- NVIDIA RTX 5060: CUDA WMMA INT8 (327 TOPS) is almost exactly 2&times; the Vulkan coopmat INT8 (166 TOPS), reflecting the K=16 vs K=32 tile difference and ptxas's cross-chain ILP.
- NVIDIA RTX 5060: cuBLASLt fp16 (80 TFLOPS) is roughly half of WMMA fp16 (166 TFLOPS) &mdash; WMMA exercises raw instruction throughput; cuBLASLt is bounded by memory traffic and occupancy at realistic GEMM sizes. The cuBLASLt number is the practical achievable peak.
- NVIDIA RTX 5060: cuBLASLt fp8 (161 TFLOPS) more than doubles cuBLASLt fp16 (80 TFLOPS), consistent with the 2&times; arithmetic density and efficient memory reuse at the same matrix dimension.
- Apple M1 Pro: simdgroup_matrix fp16 (~16 TFLOPS) is ~4&times; the MPS GEMM fp16 (~4 TFLOPS) &mdash; the hand-rolled kernel saturates the matrix-engine in a register-resident loop; MPS GEMM is memory-bound at M1 VRAM bandwidth.
- Apple M1 Pro: all three backends agree on atomic throughput &mdash; MoltenVK and native Metal both reach the hardware path.

## CLI

`./clpeak --help` prints the full flag list. The CLI is uniform across backends: the same global, test-selection, and output flags work whether OpenCL, Vulkan, CUDA, or Metal is doing the work.

```console
./clpeak                              # run every test on every available backend
./clpeak --compute-sp                 # run only single-precision compute, on every backend
./clpeak --metal                      # run only one backend
./clpeak --cuda --vulkan              # combine multiple --<backend> flags
./clpeak --no-opencl --no-cuda        # or skip the ones you don't want
./clpeak --wmma                       # CUDA tensor-core tests (hand-rolled WMMA)
./clpeak --cublas                     # CUDA vendor-SDK GEMM peak (cuBLASLt, all dtypes)
./clpeak --simdgroup-matrix           # Apple matrix-engine tests (hand-rolled simdgroup_matrix)
./clpeak --mps-gemm                   # Apple vendor-SDK GEMM peak (MPS / MPSGraph)
./clpeak --coop-matrix                # Vulkan tensor-core tests
./clpeak --xml-file out.xml           # save results (also --json-file / --csv-file)
./clpeak --compare baseline.json      # diff against a previous run
./clpeak --list-devices               # enumerate devices for every backend, no benchmarks
```

### Selecting a specific device

Multi-GPU machines pick devices per-backend:

```console
./clpeak --cl-platform 0 --cl-device 1   # OpenCL platform/device pair
./clpeak --vk-device 1                   # Vulkan physical-device index
./clpeak --cuda-device 0                 # CUDA device ordinal
./clpeak --mtl-device 0                  # Metal device index
```

`--cl-platform` / `--cl-device` / `--cl-platform-name` / `--cl-device-name` are the new spellings of `-p` / `-d` / `-pn` / `-dn`; the short forms still work and emit a deprecation warning.

## License

See [LICENSE](LICENSE).
