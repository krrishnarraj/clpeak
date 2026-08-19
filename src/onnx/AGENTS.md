# src/onnx — ONNX Runtime Backend Implementation

`OnnxPeak` class implementation: benchmarks run through ONNX Runtime
execution providers.  Built as `peak_onnx` static library.

This backend exists to reach **NPUs**.  Vendor neural accelerators (Qualcomm
Hexagon, Intel AI Boost, AMD XDNA, Apple Neural Engine) expose no ISA, no
kernel language, and no dispatch API — the vendor's AI runtime *is* the
lowest reachable level.  ONNX Runtime is the one layer that fronts all of
them on all five OSes clpeak targets, so an execution provider (EP) is
treated as a device here, exactly as a `cl_device_id` is in the OpenCL
backend.

## Quick Lookups

- Looking for the main class (`OnnxPeak` ctor, `runAll`, inventory, EP table)? → `onnx_peak.cpp`
- Looking for how the runtime library is found/loaded? → `onnx_runtime.cpp` + `onnx_runtime.h`
- Looking for session creation / per-EP options / the CPU-fallback guard? → `onnx_session.cpp`
- Looking for how models are built without protobuf? → `onnx_model.cpp` + `onnx_model.h`
- Looking for the MatMul benchmark? → `gemm.cpp`
- Looking for the dtype-accuracy benchmark? → `numeric_error.cpp`
- Looking for the transformer-block benchmark? → `block.cpp`
- Looking for the vendored C API header? → `third_party/onnxruntime/`

## Key Files

| File | Purpose |
|------|---------|
| `onnx_peak.cpp` | `OnnxPeak` class: `applyOptions()`, `runAll()`, `enumerate()`, `printInventory()`, plus `kEpTable` — the EP → display-name/type map and `onnxAvailableEps()` |
| `onnx_runtime.cpp` | `ortRuntime()` — dlopens the runtime once per process and resolves the `OrtApi` table |
| `onnx_session.cpp` | `onnxEnv()`, `onnxCreateSession()`, `onnxStatusText()` — per-EP registration options and the CPU-fallback guard |
| `onnx_model.cpp` | `OnnxGraph` — emits ONNX protobuf wire format directly; `onnxMatMulModel()` / `onnxQdqMatMulModel()` recipes; fp16/bf16 scalar conversions |
| `gemm.cpp` | `runGemm` (`--onnx-gemm`) — single-node MatMul peak. Two scopes, because a test carries one unit: `onnx-gemm-fp` (tflops, fp32 + fp16) and `onnx-gemm-int` (tops, int8 QDQ) |
| `numeric_error.cpp` | `runNumericError` (`--onnx-numeric-error`) — relative RMS error per dtype vs an fp32 CPU-EP reference, in ppm |
| `block.cpp` | `runBlock` (`--onnx-block`) — one fixed transformer decoder block in both regimes. Three scopes off two timings: `onnx-block-prefill` (tflops), `onnx-block-decode` (gbps), `onnx-block-latency` (us) |

## The runtime is dlopen'd, never linked

Only `OrtGetApiBase` is resolved by name; everything else comes through the
`OrtApi` function-pointer table it returns.  A machine without ONNX Runtime
gets a one-line "library not found" note and no rows — the same shape as a
missing GPU driver.  `CLPEAK_ONNXRUNTIME_LIB` overrides the search.

The build needs only the vendored header (`third_party/onnxruntime/`,
pinned to the ORT release in its `ORT_API_VERSION`), so no ONNX Runtime
installation is required to compile the backend.  `onnx_runtime.cpp` requests
`ORT_API_VERSION` and walks *down* to `kMinApiVersion` so a binary built
against a new header still runs on an older installed runtime.

**Never `dlclose` the runtime.** ONNX Runtime keeps worker threads alive; the
handle is deliberately leaked at exit.

## Models are emitted as protobuf bytes, not files

`onnx_model.cpp` writes the ONNX wire format by hand (varint + length-
delimited fields are all a single-op `ModelProto` needs).  This keeps clpeak
free of a protobuf dependency and of any `.onnx` asset to ship or embed, and
guarantees byte-identical models on every platform — which is what makes the
cross-vendor comparison meaningful.  Weights are embedded as an initializer
so the EP sees them as constant model weights it may pre-pack, matching how
real inference runs.

Weight/input values are small deterministic floats in `[-0.5, 0.5)`, not raw
random bits: fp16 accumulation over a 2048-deep dot product overflows with
larger magnitudes, and random bit patterns hit NaN/denormal slow paths that
would understate the hardware.

## The CPU-fallback guard

The failure mode that makes most NPU benchmarks worthless is silent
fallback: ORT partitions the graph, the EP declines a node, the node runs on
the CPU, and a CPU number gets reported under an NPU heading.  Every non-CPU
session is therefore created with `session.disable_cpu_ep_fallback=1`, so a
graph the EP cannot take **fails session creation** and the row reports
`Unsupported` with the runtime's own message instead of a wrong number.

Limits worth knowing: the guard is enforced at the ORT partitioning level.
An EP that accepts a node and then falls back *internally* (CoreML choosing
CPU over the ANE; QNN dropping from HTP to the DSP) is invisible to ORT and
to clpeak. Two cross-checks close most of that gap: the CPU EP row — always
enumerated, always last — should be far below any accelerator row, and the
`onnx-numeric-error` fp32 row exposes an EP that took an fp32 graph and
computed it at lower precision (see below).

## What the numeric-error rows are for

A TOPS figure without an accuracy figure is half a number: int8 is fast
because it discarded precision, and the speed row cannot say how much. Each
dtype is therefore also measured as relative RMS error against an fp32
reference built from *the same values the reduced-precision run saw* —
widened, not re-generated — and computed on the CPU EP.

The fp32 row doubles as a precision-downgrade detector, and on Apple silicon
it resolves which engine actually ran the work. Reference readings, M1 Pro:

| Row | fp32 | fp16 | int8 QDQ |
|-----|------|------|----------|
| CPU EP | 0.0 ppm | 207 ppm | 9463 ppm |
| CoreML EP | 0.4 ppm | 214 ppm | (unsupported) |

CPU-EP fp32 at exactly 0.0 is the methodology validating itself. CoreML's
fp16 error matching the CPU's says both accumulate in fp16 — combined with
6.2 TFLOPS (1.5x this machine's MPS GEMM fp16 peak) that row is the ANE.
CoreML's fp32 row is the interesting one: 0.4 ppm rules out fp16 arithmetic,
and its ~2.0 TFLOPS sits in the Accelerate sgemm band (2.18 best / 1.67 at
8k, `results/Apple/M1_Pro.xml`) — so CoreML serves fp32 from the CPU matrix
coprocessor, not the Neural Engine, an internal routing decision ORT never
reports.

**int8 QDQ is unsupported on the CoreML EP** — it declines the
DequantizeLinear/MatMul/QuantizeLinear graph outright, which is what makes
the guard fire. The graph shape is not the problem: the CPU EP runs the same
bytes at 1.7 TOPS, 3.7x its own fp32.

## The AI scope: what `Category::Ai` is for

`onnx-block` is the rung above a raw GEMM peak and below tokens/second: one
fixed decoder block, run in the two regimes that bound all LLM inference.
Prefill (512 tokens at once) is compute-bound and reports effective TFLOPS;
decode (1 token, 2048 of context) is memory-bound and reports the GB/s of
weights plus KV cache that must move per token. Both come out of the whole
stack — attention, softmax, SwiGLU, the layout shuffles between them — so
they are what a device delivers, not what its silicon could do in principle.
The latency scope restates the same two timings so they can be multiplied by
a model's layer count to check a tokens/second claim.

Geometry is fixed in `block.cpp` (2048-wide, 16 heads, SwiGLU 5504, 50.6M
parameters) and deliberately *not* 7B-shaped: the weights must overflow every
cache while still compiling in seconds on NPU toolchains, which build graphs
ahead of time. A 7B block is 4x the size for no extra insight and minutes of
AOT compile. fp16 only — nobody serves an LLM in fp32, so a full-precision
block would measure a configuration that does not exist.

M1 Pro reference: prefill 4.8 TFLOPS against a 6.2 TFLOPS raw fp16 MatMul
peak, i.e. a complete layer retains ~77% of the pure-matmul rate; decode
48 GB/s; 2.4 ms per layer per token.

## ORT's own graph rewrites can cost NPU placement

`MatMulAddFusion` turns `MatMul` + `Add` into `Gemm`, and several NPU
providers implement MatMul but not Gemm. The CoreML EP accepts 20 of the
block's 22 nodes and refuses exactly the two fused ones — which, under the
CPU-fallback guard, fails the entire session. `onnx_session.cpp` therefore
sets `optimization.disable_specified_optimizers=MatMulAddFusion` on every
session, so each provider runs the graph as authored. That is both what
makes the numbers comparable and what an app targeting the NPU would do.

The lesson generalizes: when a provider declines a graph, check what ORT
rewrote it into before assuming the provider lacks the operator. `--verbose`
opens the runtime's own log, which names the unsupported ops.

## Per-EP session options live in one place

`onnx_session.cpp` holds every provider's registration in `appendProvider()`.
These options are part of what gets measured (CoreML's `MLComputeUnits`,
QNN's `backend_path` pointing at HTP rather than the DSP, OpenVINO's
`device_type`), so they belong in one visible place rather than scattered
across benchmarks.

Two registration shapes exist in the ORT C API and both are needed:
`genericEpOptions()` covers the providers taking a name plus string
key/values (CoreML, QNN, OpenVINO, VitisAI, NvTensorRtRtx, XNNPACK, DML,
WebGPU), while CUDA, TensorRT, ROCm and MIGraphX have typed options structs
and dedicated append calls. A provider matching neither is reported as
unsupported rather than run with defaults — silently measuring something
unintended is the thing this backend must not do. The CPU EP is implicit in
every session and is the one provider that registers nothing and keeps its
fallback.

## When You Change This Directory

- Adding a benchmark → new `.cpp` here, entry in `src/onnx/CMakeLists.txt`,
  a `Benchmark` enum value + CLI flag (`include/common/benchmark_enums.h`,
  `src/common/options.cpp`), a call in `runAll()`, and a row in Key Files above.
- Adding EP support → `kEpTable` (`onnx_peak.cpp`) **and** `epOptionsFor()`
  (`onnx_session.cpp`). Both, or the EP enumerates but refuses to run.
- Bumping the vendored header → `tool/update_onnx_headers.sh <tag>` (never by
  hand: it refetches all three files from one release tag and rewrites the
  recorded pin). Then check `kMinApiVersion` still names the oldest runtime
  worth supporting.
