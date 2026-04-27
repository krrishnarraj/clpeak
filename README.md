# clpeak

[![Build](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml)
[![Snap Status](https://snapcraft.io/clpeak/badge.svg)](https://snapcraft.io/clpeak)

**How fast is your GPU, really?** clpeak measures the **peak achievable performance** of GPU compute devices &mdash; through OpenCL, Vulkan, CUDA, and Metal &mdash; and runs them all back-to-back so you can see what the hardware can actually do, *and* how much each driver stack leaves on the table.

```text
NVIDIA GeForce RTX 5060

  CUDA  WMMA fp16xfp16+fp32 16x16x16 (GFLOPS)     wmma_fp16  : 165847.38
  CUDA  WMMA int8xint8+int32 16x16x16 (GIOPS)     wmma_int8  : 327761.69
  CUDA  FP8(E4M3) mma.sync m16n8k32+fp32 (GFLOPS) fp8_e4m3   :  85069.15
  VK    Cooperative-matrix int8xint8+int32 16x16x32 (GIOPS)  : 165041.45
  VK    BF16 compute bf16xbf16+fp32 (GFLOPS)      bf16       :  17369.78
  CL    Single-precision compute (GFLOPS)         float8     :  20213.30
```

```text
Apple M1 Pro

  Metal simdgroup_matrix fp16xfp16+fp32 8x8x8 (GFLOPS)  : 15046.75
  Metal Local memory bandwidth (GBPS)            float4 :  2698.03
  VK    Local memory bandwidth (GBPS)            float4 :  2679.17
  CL    Local memory bandwidth (GBPS)            float4 :  2235.85
  Metal Global memory bandwidth (GBPS)           float  :   179.94
```

## Why run it

Same hardware, different framework, **different numbers**. clpeak makes those gaps visible:

- **NVIDIA RTX 5060**: OpenCL image bandwidth comes in at *1/10* the Vulkan or CUDA equivalent &mdash; not a hardware limit, a driver-side image-fetch lowering issue.
- **NVIDIA RTX 5060**: Vulkan local-atomic throughput is *1/2* the OpenCL or CUDA rate &mdash; NVIDIA's Vulkan SPIR-V atomic lowering takes a heavier-ordering path.
- **NVIDIA RTX 5060**: CUDA WMMA INT8 (328 TIOPS) is exactly *2&times;* the Vulkan coopmat INT8 (165 TIOPS) &mdash; K=16 vs K=32 tile + ptxas's cross-chain ILP.
- **Apple M1 Pro**: all three backends agree on atomic throughput &mdash; MoltenVK *and* native Metal both reach the hardware path equally.

If you're picking a framework for a new project, validating a driver upgrade, or just want to know whether the manufacturer's TFLOPS number is real, this tool answers it in one run.

## Quick start

```console
git clone --recursive https://github.com/krrishnarraj/clpeak
cmake -S clpeak -B clpeak/build
cmake --build clpeak/build -j
./clpeak/build/clpeak
```

That's it. Optional backends (Vulkan, CUDA, Metal) are auto-detected at configure time and silently skipped at runtime if the driver isn't there, so a single binary stays portable across boxes.

## Backends

| Backend | Default | Compile path | Targets |
|---|---|---|---|
| **OpenCL** | always built | C++ host + .cl strings | OpenCL 1.2 baseline; 3.0 features when headers expose them |
| **Vulkan** | on, if Vulkan SDK present | GLSL .comp &rarr; SPIR-V at configure time | Vulkan 1.1+ |
| **CUDA** | on, if CUDA Toolkit present | .cu source embedded as raw strings, NVRTC at runtime | CUDA driver API + NVRTC |
| **Metal** | on, on Apple silicon | .metal source embedded as raw strings, runtime compile | Apple7 (M1) and newer |

To skip a backend at runtime: `--no-opencl`, `--no-vulkan`, `--no-cuda`, `--no-metal`.

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
| Tensor / matrix-engine MMA | TFLOPS / TOPS | &mdash; | coopmat fp16/bf16/int8/fp8 | WMMA fp16/bf16/int8 + FP8 mma.sync | simdgroup_matrix fp16/bf16 |
| Atomic throughput (global + local) | GIOPS | &check; | &check; | &check; | &check; |
| Kernel launch latency | &mu;s | &check; | &check; | &check; | &check; |

\* needs `cl_khr_integer_dot_product`; not exposed on every OpenCL driver.

Tensor-core / matrix-engine paths are gated by hardware capability and skipped cleanly when unavailable (e.g. FP8 needs sm_89+ / Hopper+; bf16 simdgroup_matrix needs Apple9 / M3+).

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
./clpeak --compare baseline.json      # diff a previous run
./clpeak --list-devices               # enumerate without running
```

## Background

clpeak began as an OpenCL-only benchmark. As LLM-era workloads pushed GPU evaluation toward INT8 dot-product, BF16, FP8, and tensor-core throughput &mdash; and as those features landed unevenly across driver stacks &mdash; the tool grew Vulkan, CUDA, and Metal backends. Same kernel intent, expressed in each framework's native idiom; same op counts; same scaffolding pattern.

## Building from source

```console
git submodule update --init --recursive --remote
cmake -S . -B build
cmake --build build -j
```

CMake auto-detects the OpenCL / Vulkan / CUDA / Metal SDKs and prints a status block at the end of configure showing what got included:

```text
===============================================================
clpeak backend summary

  OpenCL : ENABLED  (target 300, runtime 3.0)
  Vulkan : ENABLED  (1.4.329)
  CUDA   : ENABLED  (Toolkit 13.2)
  Metal  : disabled (non-Apple host)
===============================================================
```

To force-disable a backend at build time, simply uninstall its SDK or unset the relevant environment variable; CMake will report it as `disabled (...)` with the reason.

## License

See [LICENSE](LICENSE).
