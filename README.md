# clpeak

[![Build](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml)

A synthetic micro-benchmark that measures the peak achievable performance of GPU compute devices. It exercises tight vector / MAD / MMA loops to expose what the hardware is capable of in isolation, not real-world workload performance.

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

## Backends

| Backend | Default | Compile path | Targets |
|---|---|---|---|
| **OpenCL** | always built | C++ host + .cl strings | OpenCL 1.2 baseline; 3.0 features when headers expose them |
| **Vulkan** | on, if Vulkan SDK present | GLSL .comp &rarr; SPIR-V at configure time | Vulkan 1.1+ |
| **CUDA** | on, if CUDA Toolkit present | .cu source embedded as raw strings, NVRTC at runtime | CUDA driver API + NVRTC |
| **Metal** | on, on Apple silicon | .metal source embedded as raw strings, runtime compile | Apple7 (M1) and newer |

A backend is silently skipped at runtime if its loader / driver / device is missing, so a single binary stays portable across boxes. Force-disable at runtime with `--no-opencl`, `--no-vulkan`, `--no-cuda`, `--no-metal`.

## What it measures

| Test | Unit | OpenCL | Vulkan | CUDA | Metal |
|---|---|:---:|:---:|:---:|:---:|
| Global memory bandwidth | GB/s | &check; | &check; | &check; | &check; |
| Local / shared memory bandwidth | GB/s | &check; | &check; | &check; | &check; |
| Image / texture bandwidth | GB/s | &check; | &check; | &check; | &check; |
| Transfer bandwidth (host&harr;device) | GB/s | &check; | &mdash; | &check; | &mdash; |
| Compute SP / HP / DP / MP / BF16 | GFLOPS | &check; | &check; | &check; | &check; |
| Compute INT / INT24 / INT8 / INT16 | GOPS | &check; | &mdash; | &mdash; | &mdash; |
| INT8 dot-product (DP4a) | GOPS | &check;\* | &check; | &check; | &check; (emul) |
| Packed INT4 (emulated) | GOPS | &check; | &check; | &check; | &check; |
| Tensor / matrix-engine MMA | TFLOPS / TOPS | &mdash; | coopmat fp16/bf16/int8/fp8 | WMMA fp16/bf16/int8 + FP8 mma.sync | simdgroup_matrix fp16/bf16 |
| Atomic throughput (global + local) | GOPS | &check; | &check; | &check; | &check; |
| Kernel launch latency | &mu;s | &check; | &check; | &check; | &check; |

\* needs `cl_khr_integer_dot_product`; not exposed on every OpenCL driver.

Tensor / matrix-engine paths are gated by hardware capability and skipped cleanly when unavailable (e.g. FP8 needs sm_89+ / Hopper+; bf16 simdgroup_matrix needs Apple9 / M3+).

## Cross-backend comparison

Running multiple backends on the same device exposes driver- and lowering-quality deltas that a single-stack benchmark cannot:

- NVIDIA RTX 5060: OpenCL image bandwidth comes in at ~1/10 the Vulkan or CUDA equivalent &mdash; driver-side image-fetch lowering issue, not a hardware limit.
- NVIDIA RTX 5060: Vulkan local-atomic throughput is ~1/2 the OpenCL or CUDA rate &mdash; NVIDIA's Vulkan SPIR-V atomic lowering takes a heavier-ordering path.
- NVIDIA RTX 5060: CUDA WMMA INT8 (327 TOPS) is almost exactly 2&times; the Vulkan coopmat INT8 (166 TOPS), reflecting the K=16 vs K=32 tile difference and ptxas's cross-chain ILP.
- Apple M1 Pro: all three backends agree on atomic throughput &mdash; MoltenVK and native Metal both reach the hardware path.

## CLI

```console
./clpeak --help                       # full flag list
./clpeak --compute-sp                 # run a single test on every backend that supports it
./clpeak --wmma                       # CUDA tensor-core tests
./clpeak --simdgroup-matrix           # Apple matrix-engine tests
./clpeak --coop-matrix                # Vulkan tensor-core tests
./clpeak --xml-file out.xml           # save results
./clpeak --json-file out.json
./clpeak --csv-file out.csv
./clpeak --compare baseline.json      # diff against a previous run
./clpeak --list-devices               # enumerate without running
```

## License

See [LICENSE](LICENSE).
