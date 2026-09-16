# src/litert — LiteRT Backend Implementation

`LitertPeak` class implementation: benchmarks run through LiteRT (the
runtime formerly TensorFlow Lite) on each of its accelerators.  Built as
`peak_litert` static library.

This backend exists to reach **Android's NPUs**.  ONNX Runtime is the
Windows-native route to an NPU and Core ML the Apple-native one; on Android
the vendor AI runtimes -- Qualcomm's QNN, MediaTek's NeuroPilot, Google's
Tensor TPU driver, Samsung's AI LiteCore -- are fronted by exactly one
first-party API with on-device compilation for all of them, LiteRT's
`CompiledModel`.  The same runtime carries a GPU accelerator on every OS
(OpenCL on Android, Metal on Apple, WebGPU on Linux and Windows) and XNNPACK
on the CPU, so it also answers what the mainstream mobile ML runtime gets
out of silicon clpeak already measures raw.  Each accelerator slot -- NPU,
GPU, CPU -- is one device, exactly as a Core ML compute unit is.

## Quick Lookups

- Looking for the main class (`LitertPeak`, `runAll`, device probing)? → `litert_peak.cpp`
- Looking for how the runtime library is found/loaded, or how `--litert-lib` / `--litert-npu-dir` pick one? → `litert_runtime.cpp` + `litert_runtime.h`
- Looking for environments, options payloads, compilation, buffers, the fallback guard, the profiler, the log capture? → `litert_session.cpp`
- Looking for how `.tflite` models are written without flatc? → `tflite_model.cpp` + `tflite_model.h`
- Looking for the formats, per-accelerator plans and model recipes? → `litert_model.cpp` + `litert_model.h`
- Looking for the measurement shape every test follows? → `litert_bench.h`
- Looking for the matmul ladder? → `gemm.cpp`; the accuracy rows? → `numeric_error.cpp`
- Looking for convolution? → `conv.cpp`; the transformer block? → `block.cpp`
- Looking for the vendored C headers? → `third_party/litert/`

## Key Files

| File | Purpose |
|------|---------|
| `litert_peak.cpp` | `LitertPeak`: `runAll()`, `enumerate()`; `litertUsableDevices()` probes each accelerator with one tiny model; the NPU vendor table (dispatch-library name → display name) |
| `litert_runtime.{h,cpp}` | `litertRuntime()` — dlopens libLiteRt and resolves every entry point by name into `LitertApi` (one X-macro, required and optional lists); `litertSetLibraryOverride()`, `litertSetNpuDirOverride()`, `litertLoadDiagnostic()` |
| `litert_session.{h,cpp}` | `LitertSession`: one environment per accelerator (with the recreation the Metal accelerator needs), the options payloads, model load + compile, tensor buffers, `run()`, the sink logger and console capture, the profiler |
| `tflite_model.{h,cpp}` | A minimal back-to-front FlatBuffer builder and `TfliteModel`, which serializes tensors, buffers, operators and their options tables to the `.tflite` wire format |
| `litert_model.{h,cpp}` | `LitertFormat` / `LitertPlan` (what a format is on each accelerator), scalar conversions, the operand generator and quantization scales, and every recipe: matmul, plain matmul, GEMV, activations, transfer, trivial, conv, transformer block |
| `litert_bench.h` | `litertMeasure()` (warmup / probe / timed), `litertBindScalar()`, `litertConfigFor()`, `litertFailureStatus()` |
| `gemm.cpp` | `runGemm` (`--gemm`) — `litert_gemm`: FULLY_CONNECTED peak per format over a doubling size ladder, in flops or ops, naming the kernel that ran |
| `numeric_error.cpp` | `runNumericError` (`--numeric-error`) — relative RMS error per format vs a double-precision host reference, in ppm |
| `conv.cpp` | `runConv` (`--convolution`) — 3×3 / 1×1 / depthwise 3×3 at 256 channels in fp32, fp16 and full-integer int8, swept over feature-map size |
| `block.cpp` | `runBlock` (`--transformer-block`) — the ONNX backend's decoder block: `litert_block_prefill` (flops, `ops` for int8_qdq), `litert_block_decode` (bps), `litert_block_latency` (s) |
| `activation.cpp` | `runActivation` (`--activation`) — SiLU / softmax / layer norm as GB/s at 8/32/128 MB, net of a reference graph |
| `tensor_bandwidth.cpp` | `runTensorBandwidth` (`--tensor-bandwidth`) — GEMV against a resident fp16 weight, 8 MB to 2 GB, net of the dispatch floor |
| `transfer.cpp` | `runTransferBandwidth` (`--transfer-bandwidth`) — h2d / round trip / d2h through LiteRT's tensor buffers |
| `dispatch_latency.cpp` | `runDispatchLatency` (`--kernel-launch-latency`) — a one-operator graph, a 256-cube matmul, and compiled-model creation |

Test ids are lower_snake and keep the `litert_` prefix (`litert_gemm`,
`litert_block_decode`); the CLI flags that gate them are backend-neutral
(`--gemm`, `--transformer-block`, …) and shared with the ONNX and Core ML
backends through the same `Benchmark` values.  Every test is heterogeneous:
each reading is a different format, shape, size or context length.

## The runtime is dlopen'd, never linked

Only the platform's conventional names are searched (`libLiteRt.so` on
Android, where the app packages the `com.google.ai.edge.litert` AAR and the
bare soname resolves out of the APK; `libLiteRt.dylib` / `.dll` on desktops,
where the pip `ai-edge-litert` wheel or `--litert-lib PATH` supplies one).
Every C entry point is resolved by name -- LiteRT has no single
`OrtGetApiBase`-style getter on its public surface -- through the X-macro
lists in `litert_runtime.h`: a REQUIRED symbol missing fails the load with
its name, an OPTIONAL one (the profiler, the sink logger, error messages)
leaves a null the callers check.  The directory the library loaded from is
remembered because LiteRT looks for its GPU accelerator library and the NPU
dispatch libraries there unless told otherwise; `--litert-npu-dir` is that
telling.  **Never `dlclose` the runtime**: it keeps accelerator contexts and
worker threads alive.

**Accelerator and vendor options are TOML strings, not linked helpers.**
The `Lrt*Options` builders in `litert/c/options/*.h` are client-side
sources in LiteRT's C++ SDK, not exports of libLiteRt (they need Abseil).
What the runtime receives is an opaque payload under an identifier --
`"xnnpack"` with `num_threads = 10`, `"gpu_options"` with `precision = 2`,
`"qualcomm"` with `htp_performance_mode = 2`, `"runtime_options_string"`
with `enable_profiling = true` -- and `litert_session.cpp` writes those
strings directly.  The vendored option headers are kept for their enums.

## Models are emitted as FlatBuffer bytes, not built with a converter

`tflite_model.cpp` writes the `.tflite` wire format by hand, as the ONNX
and Core ML backends write protobuf: no flatc, no FlatBuffers library, no
1.1 MB `schema_generated.h`, and byte-identical models on every platform.
The builder is the reference implementation reduced to what a model needs
(back-to-front, vtables, forward offsets, 16-byte-aligned weight buffers,
the `TFL3` identifier), and every field id and union position it writes is
transcribed from `schema.fbs` with the source noted beside it.  Weights are
generated straight into the model bytes (`addBufferFill`), so a rung's
operands exist once in memory rather than twice -- on a phone the
difference between a 1 GB rung and a killed process.  Models are loaded
from that buffer in place (`LiteRtCreateModelFromBuffer`), and the session
owns the bytes for its lifetime.

**Every activation tensor carries a leading batch dimension of one.**  The
GPU accelerator maps a 2-D tensor's first dimension onto its batch axis and
reduces over it wrongly: `REDUCE_MAX` over axis 0 of a `[D, D]` matmul
result returned one row of the input on an M1 Pro, while the CPU was right
to the last bit.  With `[1, M, K]` activations every accelerator agrees.

## A format is one graph, run as each accelerator can

`LitertFormat` is what a shipped model *is* -- its storage types and
quantization scheme -- and `litertPlanFor()` says how one accelerator
expresses it: the graph's tensor types plus the accelerator policy that goes
with them.  The rows are then honest about arithmetic the way the ONNX rows
are: the rate row says how fast, the accuracy row how accurately, and the
kernel name from one profiled run says which kernel it was.

- **XNNPACK names its GEMM by the types it packed** -- `Fully Connected
  (NC, F16) GEMM`, `(NC, QS8, QC8W)`, `(NC, QP8, F32, QC8W)`, `(NC, QD8,
  F32, QB4W)`.  The last two are the finding: the weight-only formats
  (`int8_weight`, `int4_weight`) run as *int8 arithmetic on dynamically
  quantized activations*, not as float multiplies of unpacked weights.  On
  an M1 Pro that is why `int8_weight` reads 2.57 TFLOPS against fp32's 468
  GFLOPS, and why its accuracy row reads 3915 ppm where the GPU's -- which
  unpacks the same weights to half and multiplies in float -- reads 4696
  because it accumulates in fp16.
- **The GPU accelerator computes an fp32 graph in fp16 unless told
  otherwise**, so `fp32` asks for its fp32 policy, `fp16` is its fp16 policy
  over the same fp32 graph (it refuses half-typed tensors outright), and
  `fp16_acc32` is its third policy, fp16 storage with fp32 accumulation.  On
  an M1 Pro: 4.10 / 4.71 / 3.44 TFLOPS and 0.57 / 4690 / 388 ppm -- the GPU
  accumulates fp16 in fp16 by default.  `fp16_acc32` applies nowhere else.
- **Its quantized graphs are float kernels between quantize and dequantize
  passes** (`convolution1x1(conv_wave_matrix) -> quantize_and_dequantize`),
  so its `int8_qdq` "TOPS" equals its fp16 rate; the row says so when the
  kernel tag carries `quantize_and_dequantize`.
- `int16x8` has no XNNPACK kernel: the CPU row is the runtime's reference
  kernel, three orders of magnitude slower, which is the honest number for
  a format that only an NPU implements.
- `bf16` and `fp8_weight` are in the schema (fp8 since 2.2.0) and no kernel
  takes them on the CPU; the rows record the runtime's refusal.

**Labels are the ONNX and Core ML backends' where the format is the same
one** (`fp16`, `int8_qdq`, `int8_weight`, `int4_weight`), so a block
reading here divides by a GEMM reading there.  `int16x8` and `fp16_acc32`
are this backend's own.

## The runtime's own answer is the guard

`LiteRtCompiledModelIsFullyAccelerated()` says whether every operation
landed on the accelerator selected; LiteRT otherwise keeps the CPU as a
silent fallback and a matmul it handed back would be timed on the CPU under
the accelerator's name.  Every GPU and NPU session checks it and a row it
fails reports unsupported with the reason.  Sessions select *only* the
target accelerator in their options, so partial delegation is what the
check catches rather than being masked by a second accelerator.

The profiler (`enable_profiling = true`, `LiteRtCompiledModelGetProfiler`)
is the second witness: one profiled run per variant at the smallest size
names the kernel that did the multiply, and that name goes into the row.
**Profiling never touches a timed session** -- on the Metal GPU it waits on
every kernel and cut a 4 TFLOPS matmul to 0.4.

## What the accelerator libraries do wrong, and how the backend stays up

Four faults in LiteRT 2.2.0's GPU accelerator would take the whole run
down, and a benchmark that dies to prove a point has proved the wrong one.
Each is fenced in `litertPlanFor()` or `litert_session.cpp` with the fault
recorded in the row's reason, so lifting the fence when a release fixes it
is a one-line change:

- **Unsupported tensor types abort the process.**  The graph reader
  `CHECK`s that a tensor is fp32/fp16/int8/uint8/int4/int2/bool/int32
  (`object_reader.cc`) and glog aborts on anything else -- int16, bfloat16,
  float8 -- rather than declining.  Those formats are never sent to the GPU.
- **Blockwise int4 corrupts the heap.**  The weight conversion transposes
  the packed nibbles with an 8-bit 16×16 transpose microkernel
  (`xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon`) that runs past the
  buffer; guard malloc catches it on the first model, a normal run dies some
  rungs later.  Per-row int4 goes through a different path and is safe, but
  it is not the format language models ship in, so the blocked format is not
  sent to the GPU rather than measured as something else.
- **The OpenCL accelerator crashes without an OpenCL library.**  On a
  machine with no `libOpenCL` at all (an emulator, a box without a driver)
  `libLiteRtClGlAccelerator` dereferences a null inside `strlen` when it
  compiles its first model instead of declining.  `litertUsableDevices()`
  brings the GPU environment up on its own first, reads which accelerator
  library registered, and when it is the OpenCL one requires an OpenCL
  library to be loadable before any model is sent.
- **The Metal accelerator leaks a residency set per compiled model** and
  `IOGPUMetalCommandQueue` asserts on the 33rd ("command queue residency
  set limit of 32 exceeded").  The `enable_metal_residency_set` option does
  not stop it and neither does an autorelease pool; only a new command
  queue does, so on Apple the GPU environment is torn down and rebuilt
  every `kGpuModelsPerEnvironment` (24) compiled models, between sessions,
  for ~300 ms of Metal initialisation each time.

Verify a new format or accelerator path under guard malloc
(`DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib`) before trusting a run
that merely completes: the int4 overrun completed several runs first.

## What the runtime says reaches the row

LiteRT logs through a process-wide logger, replaced once per runtime with
its sink logger so the console stays clean and every line reaches the run
log at LiteRT's own severity under `--verbose`.  TFLite's kernel errors take
a different road -- its error reporter, which is stderr -- and they arrive at
the *first inference* on the CPU, not at compile time ("input->type !=
kTfLiteFloat32 (BFLOAT16 != FLOAT32)").  So creation, environment setup and
the first `run()` of every session are wrapped in `ScopedConsoleMute
(Capture::Always)`, and `lastLines()` distils whatever came out into the
sentence the row quotes.  Later runs are bare: a mute is two `dup2` calls a
timed loop must not pay.  A refusal that arrives as "failed to prepare" is
reported as unsupported, not as an error: it is the runtime saying it has
no kernel for the format.

## Reference readings, M1 Pro (LiteRT 2.2.0, macOS 26)

| row | GPU (Metal) | CPU (XNNPACK, 10 threads) |
|---|---|---|
| gemm fp32 / fp16 / fp16_acc32 | 4.10 / 4.71 / 3.44 TFLOPS | 468 GFLOPS / 1.00 TFLOPS / — |
| gemm int8_qdq / int16x8 | 4.73 "TOPS" (float kernel) / aborts | 2.51 TOPS / 1.23 GOPS |
| gemm int8_weight / int4_weight | 4.65 TFLOPS / heap overrun | 2.57 / 1.67 TFLOPS |
| error fp32 / fp16 / fp16_acc32 | 0.57 / 4690 / 388 ppm | 0.57 / 4690 / — |
| error int8_qdq / int8_weight / int4_weight | 10486 / 4696 / — | 9383 / 3915 / 4370 |
| conv3x3 fp32 / fp16 / int8 | 8.88 / 12.7 T (Winograd-counted) / 12.7 TOPS | 594 G / 1.07 T / 2.64 TOPS |
| block prefill fp16 s2048 / decode fp16 kv2048 | 4.37 TFLOPS / 203 GB/s | ~0.95 TFLOPS / 102 GB/s |
| tensor_bw 8mb / 128mb | 183 / 140 GB/s | 214 / 117 GB/s |
| dispatch trivial / matmul_256 / create | 252 µs / 331 µs / 1.23 ms | 551 ns / 142 µs / 173 µs |

## Packaging

- **Android**: `app/android/app/build.gradle.kts` packages the
  `com.google.ai.edge.litert` AAR (arm64: `libLiteRt.so` 5.5 MB +
  `libLiteRtClGlAccelerator.so` 3.1 MB); its manifest's `uses-native-library`
  entries merge into ours.  NPU dispatch and compiler-plugin shims come from
  the release's `litert_npu_runtime_libraries_jit.zip` via
  `tool/fetch_litert_npu.sh` into `src/main/jniLibs/` (git-ignored).  MediaTek
  and Google Tensor runtimes are system libraries; Qualcomm's QAIRT
  libraries must be bundled per Hexagon generation (the zip's own
  `fetch_qualcomm_library.sh`), which for a Play release means the dynamic
  feature modules the zip is laid out as.  NPU needs API 31+ and arm64.
- **Desktop**: `--litert-lib` at a pip wheel's `libLiteRt.{so,dylib,dll}`;
  the GPU accelerator and the Intel OpenVINO NPU dispatch sit beside it.
- **Verified on Android** with the CLI built against the NDK (root
  CMakeLists with `android.toolchain.cmake`, every other backend off) and
  pushed to an arm64 emulator beside the AAR's two `.so` files: every test
  runs on XNNPACK, the numeric-error rows match macOS bit for bit, the GPU
  is declined for lack of OpenCL, and the fp32 block is skipped by the
  memory gate on 4 GB.  The release APK builds with the AAR's `litert-api`
  excluded (it repeats the AAR's namespace, which AGP 9 rejects) and R8
  told not to chase the AAR's unused Java classes.
- **iOS**: not yet wired; LiteRT's dylibs can be dlopen'd from an embedded
  framework there (MLPerf Mobile does), unlike ONNX Runtime.

## When You Change This Directory

- A new format: a `LitertFormat` value, its plan in `litertPlanFor()` (with
  any GPU fence), a row in `gemm.cpp`'s and `numeric_error.cpp`'s tables, a
  `storedValue` case if it stores codes.
- A new entry point: the X-macro in `litert_runtime.h`, checked against
  `nm -gU libLiteRt.dylib` -- and REQUIRED only if every path uses it.
- A new recipe: `litert_model.h` declares, `litert_model.cpp` builds
  through `Recipe`; keep the leading batch dimension and the runtime scalar.
- Update `third_party/litert` with `tool/update_litert_headers.sh <tag>`.
