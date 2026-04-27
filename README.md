# clpeak

[![Build](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml)
[![Snap Status](https://snapcraft.io/clpeak/badge.svg)](https://snapcraft.io/clpeak)

A synthetic micro-benchmark that measures the **peak achievable performance** of GPU compute devices across multiple frameworks. Numbers reflect what the hardware is capable of in tight vector/MAD/MMA loops &mdash; not real-world workload performance.

clpeak began as an OpenCL-only tool. It now ships **four interchangeable backends** so the same kernels can run on the same device through different driver stacks &mdash; making cross-stack quality differences (driver lowering, instruction scheduling, extension exposure) visible at a glance.

## Backends

| Backend | Status | Compile path | Runtime |
|---|---|---|---|
| **OpenCL** | always built | C++ host + .cl strings | OpenCL 1.2 baseline; OpenCL 3.0 features enabled when headers expose them |
| **Vulkan** | optional, default-on | GLSL .comp &rarr; SPIR-V at configure time, embedded as `uint32_t[]` | Vulkan 1.1+ |
| **CUDA** | optional, default-on | .cu source embedded as raw strings, NVRTC-compiled at runtime | CUDA driver API + NVRTC |
| **Metal** | optional, default-on (Apple silicon) | .metal source embedded as raw strings, `MTLDevice newLibraryWithSource:` at runtime | Apple7 (M1) and newer |

The CMake configure prints a per-backend status block:

```text
===============================================================
clpeak backend summary

  OpenCL : ENABLED  (target 300, runtime 3.0)
  Vulkan : ENABLED  (1.4.329)
  CUDA   : ENABLED  (Toolkit 13.2)
  Metal  : disabled (non-Apple host)
===============================================================
```

A backend is silently skipped at runtime if its loader / driver / device is missing, so a single `clpeak` binary built with everything enabled stays portable.

## What it measures

| Test | Unit | OpenCL | Vulkan | CUDA | Metal |
|---|---|:---:|:---:|:---:|:---:|
| Global memory bandwidth | GB/s | &check; | &check; | &check; | &check; |
| Local / shared memory bandwidth | GB/s | &check; | &check; | &check; | &check; |
| Image / texture bandwidth | GB/s | &check; | &check; | &check; | &check; |
| Transfer bandwidth (host&harr;device) | GB/s | &check; | &mdash; | &check; | &mdash; |
| Compute SP / HP / DP / MP / BF16 | GFLOPS | &check; | &check; | &check; | &check; |
| Compute INT / INT24 / INT8 / INT16 | GIOPS | &check; | &mdash; | &mdash; | &mdash; |
| INT8 dot-product (DP4a) | GIOPS | &check;\* | &check; | &check; | &check; (emul) |
| Packed INT4 (emulated) | GIOPS | &check; | &check; | &check; | &check; |
| Tensor-core / matrix-engine MMA | TFLOPS / TOPS | &mdash; | coopmat fp16/bf16/int8/fp8 | WMMA fp16/bf16/int8 + FP8 mma.sync | simdgroup_matrix fp16/bf16 |
| Atomic throughput (global + local) | GIOPS | &check; | &check; | &check; | &check; |
| Kernel launch latency | &mu;s | &check; | &check; | &check; | &check; |

\* `cl_khr_integer_dot_product` is required; not exposed on every OpenCL driver.

Tensor / matrix-engine paths are gated by hardware capability and skipped cleanly when unavailable (e.g. FP8 needs sm_89+ / Hopper+; bf16 simdgroup_matrix needs Apple9 / M3+).

## Building

```console
git submodule update --init --recursive --remote
cmake -S . -B build
cmake --build build -j
```

Optional backends are auto-detected. To force-disable at runtime:

```console
./build/clpeak --no-opencl   # skip OpenCL
./build/clpeak --no-vulkan   # skip Vulkan
./build/clpeak --no-cuda     # skip CUDA
./build/clpeak --no-metal    # skip Metal
```

`./build/clpeak --help` lists all per-test selectors (`--compute-sp`, `--coop-matrix`, `--wmma`, `--simdgroup-matrix`, &hellip;).

## Sample output

Apple M1 Pro &mdash; OpenCL + Vulkan (via MoltenVK) + Metal in a single run, condensed:

```text
=== OpenCL backend ===

Platform: Apple
  Device: Apple M1 Pro
    Single-precision compute (GFLOPS)
      float16 : 2315.21
    Local memory bandwidth (GBPS)
      float4  : 2235.85
    Image memory bandwidth (GBPS)
      float4  : 300.25
    Atomic throughput (GIOPS)
      global  : 24.43
      local   : 250.69

=== Vulkan backend ===

Vulkan Device: Apple M1 Pro
  Single-precision compute (GFLOPS)
    float : 4114.70
  Local memory bandwidth (GBPS)
    float4 : 2679.17
  Image memory bandwidth (GBPS)
    float4 : 452.57
  Atomic throughput (GIOPS)
    global : 24.74
    local  : 255.02

=== Metal backend ===

Metal Device: Apple M1 Pro
  Single-precision compute (GFLOPS)
    float : 4344.09
  simdgroup_matrix fp16xfp16+fp32 8x8x8 (GFLOPS)
    simdgroup_fp16 : 15046.75
  Local memory bandwidth (GBPS)
    float4 : 2698.03
  Image memory bandwidth (GBPS)
    float4 : 498.00
  Atomic throughput (GIOPS)
    global : 24.44
    local  : 256.24
```

NVIDIA RTX 5060 &mdash; OpenCL + Vulkan + CUDA, tensor-core highlights:

```text
=== Vulkan backend ===
  Cooperative-matrix int8xint8+int32 16x16x32 (GIOPS)
    coopmat_int8 : 165041.45

=== CUDA backend ===
  WMMA fp16xfp16+fp32 16x16x16 (GFLOPS)
    wmma_fp16 : 165847.38
  WMMA int8xint8+int32 16x16x16 (GIOPS)
    wmma_int8 : 327761.69
  FP8(E4M3) mma.sync m16n8k32+fp32 (GFLOPS)
    fp8_e4m3 : 85069.15
```

The same hardware through three frameworks &mdash; e.g. CUDA WMMA INT8 (328 TIOPS) is exactly 2&times; the Vulkan coopmat INT8 number (165 TIOPS), reflecting the K=16 vs K=32 tile difference and ptxas's cross-chain ILP.

## Cross-backend diagnostic value

Running multiple backends on the same device exposes driver- and lowering-quality deltas that a single-stack benchmark cannot:

- **NVIDIA RTX 5060**: OpenCL image bandwidth comes in at &sim;1/10 the Vulkan / CUDA equivalent (driver-side image-fetch lowering issue, not a hardware limit).
- **NVIDIA RTX 5060**: Vulkan local-atomic throughput is &sim;1/2 the OpenCL / CUDA equivalent (NVIDIA's Vulkan SPIR-V atomic lowering takes a heavier-ordering path).
- **Apple M1 Pro**: all three backends agree on atomic throughput &mdash; MoltenVK and native Metal both reach the hardware path.

## Output formats

```console
./clpeak --xml-file results.xml
./clpeak --json-file results.json
./clpeak --csv-file results.csv
./clpeak --compare baseline.json   # diff a previous run
```

Each backend emits results under its own section (`<opencl>`, `<vulkan>`, `<cuda>`, `<metal>`).

## License

See [LICENSE](LICENSE).
