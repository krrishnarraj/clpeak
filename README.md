# clpeak

<a href="https://play.google.com/store/apps/details?id=kr.clpeak"><img alt="Get clpeak on Google Play" height="52" src="https://upload.wikimedia.org/wikipedia/commons/7/78/Google_Play_Store_badge_EN.svg"></a>
<a href="https://snapcraft.io/clpeak"><img alt="Get clpeak from the Snap Store" height="52" src="https://snapcraft.io/static/images/badges/en/snap-store-black.svg"></a>

[![Build](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/krrishnarraj/clpeak/actions/workflows/build.yml)

**clpeak &mdash; "Compute Latency PEAK".** A synthetic micro-benchmark for measuring the peak achievable compute performance of CPUs, GPUs and NPUs. It exercises tight vector, MAD, and MMA kernels, together with vendor-optimized GEMM libraries, to expose peak hardware throughput.

Originally an OpenCL benchmark, clpeak now supports OpenCL, Vulkan, CUDA, ROCm/HIP, Metal, oneAPI/SYCL, ONNX and native CPU execution, enabling direct cross-backend comparisons on the same hardware.

[![clpeak desktop app showing Metal results on an Apple M1 Pro](docs/assets/img/results-dark.png)](https://krrishnarraj.github.io/clpeak/)

## Sample output

Peak lines from real runs, condensed (see `results/` for full baselines).

NVIDIA RTX 5060, CUDA backend:

```text
Backend: CUDA
  Device 0: NVIDIA GeForce RTX 5060
    FP16 mma.sync m16n8k16+fp16
      fp16_f16acc : 83.4 TFLOPS
    NVFP4 mma.sync m16n8k64+fp32
      nvf4_e2m1 : 327 TFLOPS
    cuBLASLt GEMM peak
      fp16     : 77.5 TFLOPS
      nvf4_e2m1 : 299 TFLOPS
    Global memory bandwidth
      float4   : 419 GB/s
    Kernel launch latency
      roundtrip : 6.24 µs
```

Apple M1 Pro and RTX 5060, ONNX backend (one execution provider = one device):

```text
Backend: ONNX
  Device 0: Apple CoreML (Neural Engine) [NPU]
    ONNX MatMul peak
      fp32     : 2.37 TFLOPS
      fp16     : 8.80 TFLOPS
    ONNX convolution peak
      fp16_conv3x3 : 9.30 TFLOPS
    Transformer block, prefill
      fp16_s512 : 4.86 TFLOPS
    Transformer block, decode
      fp16_kv2048 : 49.6 GB/s

  Device 0: NVIDIA TensorRT [GPU]
    ONNX MatMul peak
      fp16     : 63.3 TFLOPS
      int8_qdq : 112 TOPS
      nvfp4    : 217 TFLOPS
    ONNX MatMul numeric error
      fp16     : 1185 ppm
```

## Desktop app

Same engine as the CLI, with device detection, live results and run history — one Flutter app for **macOS, Linux and Windows** (plus Android/iOS from the same codebase, over the `clpeak_ffi` C ABI). Grab it from the [latest release](https://github.com/krrishnarraj/clpeak/releases/latest), press **Run**, and export from History as JSON.

## Building

```console
git submodule update --init --recursive --remote
cmake -S . -B build
cmake --build build -j
./build/clpeak
```

Backends auto-enable when their SDK is found; opt out with `-DCLPEAK_ENABLE_<X>=OFF`. The ONNX backend needs no SDK (vendored header only). oneAPI needs `-DCMAKE_CXX_COMPILER=icpx`.

| CMake option | Default | Effect when `OFF` |
|---|---|---|
| `CLPEAK_ENABLE_OPENCL` | `ON` | Skip OpenCL backend |
| `CLPEAK_ENABLE_VULKAN` | `ON` | Skip Vulkan even if SDK present |
| `CLPEAK_ENABLE_CUDA` | `ON` | Skip CUDA even if Toolkit present |
| `CLPEAK_ENABLE_ROCM` | `ON` | Skip ROCm/HIP even if SDK present |
| `CLPEAK_ENABLE_METAL` | `ON` | Skip Metal/MPS even on Apple silicon |
| `CLPEAK_ENABLE_ONEAPI` | `ON` | Skip oneAPI/SYCL |
| `CLPEAK_ENABLE_CPU` | `ON` | Skip native CPU backend (otherwise always available) |
| `CLPEAK_ENABLE_ONNX` | `ON` | Skip ONNX Runtime backend (otherwise always built; runtime loaded at run time) |
| `CLPEAK_ENABLE_GUI` | `ON` | Skip the `clpeak-gui` desktop app (also skipped when no Flutter SDK is found) |

The app bundle lands in `build/clpeak-gui/` whenever Flutter is on `PATH` (`cmake --build build --target clpeak-gui`).

## CLI

`./clpeak --help` prints all flags. Selection is uniform: `--<backend>` runs only that backend, `--<test>` runs only that test, `--no-<x>` always subtracts.

```console
./clpeak                              # everything, everywhere
./clpeak --cuda --vulkan              # one or more backends (--onnx, --metal, --rocm, --oneapi, --cpu, …)
./clpeak --single-precision-compute   # one test, on every backend
./clpeak --onnx-gemm --onnx-block     # ONNX tests (--onnx-conv, --onnx-numeric-error, --onnx-tensor-bandwidth, …)
./clpeak --onnx --onnx-device 0       # one ONNX provider; --onnx-lib PATH picks the runtime
./clpeak --describe                   # what each test and reading measures
./clpeak -o out.clpeak.json           # save results (one JSON document)
./clpeak --compare baseline.clpeak.json   # diff against a saved baseline
./clpeak --list-devices               # enumerate devices, no benchmarks
```

`--compare` re-runs and prints each result beside the saved value, flagging regressions as regressions. Device indices (`--cl-platform/--cl-device`, `--vk-device`, `--cuda-device`, `--rocm-device`, `--mtl-device`, `--oneapi-device`) take one index or a comma-separated list; the CPU backend has no index (`--no-cpu` skips it).

## For AI agents

This tree is documented with `AGENTS.md` files. Start at the
[root `AGENTS.md`](AGENTS.md) for architecture, directory map, build
instructions, and the self-maintaining documentation conventions.
Every subdirectory has its own `AGENTS.md` with local details — open
the one closest to the code you're touching.
