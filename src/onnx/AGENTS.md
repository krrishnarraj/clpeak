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
- Looking for the bandwidth / dispatch-overhead benchmarks? → `tensor_bandwidth.cpp`, `dispatch_latency.cpp`
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
| `tensor_bandwidth.cpp` | `runTensorBandwidth` (`--onnx-tensor-bandwidth`) — GEMV against a resident fp16 weight matrix at three sizes (gbps) |
| `dispatch_latency.cpp` | `runDispatchLatency` (`--onnx-dispatch-latency`) — per-submission overhead and session-creation cost (us) |

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

**Ask for the right API version in one call.** ORT numbers its API after its
own minor version (1.23.x serves API 23), so `onnx_runtime.cpp` parses
`GetVersionString()` and requests that directly. Counting down from
`ORT_API_VERSION` instead makes ORT print `The requested API version [N] is
not available` once per failed attempt, straight to the console and below any
log level — six lines of it against a 1.23 runtime, before any test runs.

## Vendor console spam is muted, not tolerated

Registering a provider can pull in a second copy of the ONNX schema registry:
the XNNPACK EP emits hundreds of `Schema error: ... already registered` lines
from the bundled ONNX library, direct to the console, below any ORT log
level. Session creation therefore runs inside `clpeak::ScopedConsoleMute`
(`common/console_mute.h`, shared with the ROCm backend's hipBLASLt query).
The mute is a no-op under `--verbose`.

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
| CUDA EP (RTX 5060) | **261 ppm** | 207 ppm | 9467 ppm |

The CUDA fp32 row is the design paying off on a second vendor: 261 ppm is
worse than fp16's, which is not what fp32 arithmetic looks like. It is TF32 —
ten mantissa bits, the same as fp16 — which cuBLAS selects by default. The
"fp32" throughput row on NVIDIA is therefore a TF32 number, and only the
error row says so.

CPU-EP fp32 at exactly 0.0 is the methodology validating itself. CoreML's
fp16 error matching the CPU's says both accumulate in fp16 — combined with
6.2 TFLOPS (1.5x this machine's MPS GEMM fp16 peak) that row is the ANE.
CoreML's fp32 row is the interesting one: 0.4 ppm rules out fp16 arithmetic,
and its 2.3 TFLOPS sits in the Accelerate sgemm band (2.18 best over a size
sweep, `results/Apple/M1_Pro.xml`) — so CoreML serves fp32 from the CPU
matrix coprocessor, not the Neural Engine, an internal routing decision ORT
never reports. The size curves above agree: the fp32 row behaves like AMX
(faster the bigger it gets) and the fp16 row like the ANE (a cliff once it
outgrows on-chip memory).

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

M1 Pro reference: prefill 4.8 TFLOPS against an 8.8 TFLOPS raw fp16 MatMul
peak, so a complete layer retains ~55% of the pure-matmul rate; decode
48 GB/s; 2.4 ms per layer per token. The block still passes its activations
across the host boundary each run (2 MB in, 2 MB out), so on a discrete GPU
its numbers carry a transfer component the GEMM rows no longer do — worth
fixing the same way if the two are ever compared closely.

## Throughput graphs keep both operands resident

`onnx-gemm` makes **A and B both initializers** and reduces the result to one
row, so nothing large crosses the host boundary per run. The obvious shape —
A as a graph input, C returned to the host — measures a discrete GPU through
its PCIe bus. On an RTX 5060 it reported 15 TFLOPS for fp16 while
`onnx-block`, whose weights are resident, reached 28 on the same device: a
composite layer beating the peak of the operation it is built from, which is
impossible and was the tell. Even on the ANE, where memory is unified, going
resident moved fp16 from 6.0 to 8.8 TFLOPS.

Three details are load-bearing:

- **Constant folding must be off** (`keepConstantsUnfolded` on
  `onnxCreateSession`). Two constant operands are otherwise multiplied once at
  load time and every timed run measures an empty graph. `gemm.cpp` guards
  this: real work grows with the cube of the size, 64x across the ladder, so
  timings that stay flat are reported as an error rather than as a
  spectacular number.
- **Reduce with `ReduceMax`, never `ReduceSum`.** Summing the rows of `A*B`
  equals multiplying the summed rows of `A` — a rewrite an optimiser is free
  to make, and it would quietly turn the matrix multiply into a matrix-vector
  one. Max does not distribute over the product. (At opset 17 `ReduceMax`
  takes its axes as an attribute while `ReduceSum` takes them as an input.)
- **Nothing may sit between the dequantize and the matmul** in the QDQ form,
  or ORT stops recognising a quantized matmul and silently measures float
  arithmetic. The scaling input therefore hangs off the far end, and the
  reduction runs on the int8 result — ORT reduces int8 directly, so the tail
  costs one pass instead of dequantizing the whole product first.

`onnx-block` is built the same way — activations are a constant scaled by a
runtime scalar, result reduced to one row. Scaling by a runtime value keeps
every node downstream non-constant, so unlike the GEMM graphs the block does
not depend on constant folding being disabled.

`numeric_error.cpp` keeps using the plain, non-resident models: it compares
actual output values, so it needs the real result rather than a reduction.

## Sizes are swept, never chosen by a probe

`gemm.cpp` runs a fixed 1024/2048/4096 ladder and reports each datatype's
best. Picking one size from a timing probe — the pattern `mps-gemm` uses —
was tried first and is unstable: the estimate comes out of a cube root and
is then bucketed, so a 2% wobble in the probe can push it across a bucket
edge, and the size changes the answer. On the M1 Pro the fp16 row alternated
between 5.8 and 6.2 TFLOPS run to run purely on which side it landed.

No single size is right anyway, which the ladder makes visible:

| CoreML EP | 1024³ | 2048³ | 4096³ |
|-----------|-------|-------|-------|
| fp32 | 1.76 | 1.94 | **2.32** |
| fp16 | 3.13 | **5.98** | 5.57 |

fp32 climbs to 4096 while fp16 peaks at 2048 and *falls* — the two rows are
served by different engines (see the numeric-error section) with different
fast-memory limits, and the fp16 drop is the Neural Engine spilling its
on-chip memory, the same cliff Apple silicon is independently reported to
have between those two sizes. The CPU EP is flat across the ladder, as a
cache-blocked CPU should be.

A fixed ladder also means two devices are always compared on identical work,
which an adaptive probe cannot promise. The per-size budget is 2 s rather
than the 5 s a single-size test would use, so sweeping three costs about
what measuring one did.

**`mps-gemm` in the Metal backend still uses the probe pattern** and carries
the same latent instability; whether it bites depends on where that device's
estimate lands relative to a bucket edge.

## Measuring bandwidth needs a matmul, not something simpler

`onnx-tensor-bw` streams a resident fp16 weight matrix through a GEMV,
because that is the operation generating a token performs and the one every
provider tunes hardest. An elementwise-plus-reduction graph reads ~22 GB/s
on the M1 Pro against a machine that does roughly ten times that — it
measures the reduction, not memory. Two operations per weight also puts the
arithmetic far enough below the memory cost that only memory is left.

One matmul per dispatch, deliberately. Chaining several against the same
weights would amortise submission overhead, but **the ANE refuses a chained
program outright** (`ANEProgramProcessRequestDirect` fails) while running the
single-op form happily.

Submission cost is subtracted instead: the same graph is timed once with a
256×256 weight matrix, whose 128 KB cannot matter, and that floor comes off
every rung. Without it the ladder reads backwards wherever dispatch is
expensive — an RTX 5060 charges ~17 µs, half of what eight megabytes takes to
move, and reported 219 / 265 / 392 GB/s across a ladder whose small end
should be fastest. The floor probe carries a little transfer of its own on
slow providers, but only ~128 KB worth, so it over-corrects by under 2% of
the smallest rung.

The 128 MB rung reads ~50 GB/s on the M1 Pro and `onnx-block-decode` reads
49 GB/s from a completely different graph. Two independent tests landing
together is the best evidence available here that both measure what they
claim.

## Dispatch overhead is why NPUs lose small work

M1 Pro, CoreML EP vs CPU EP: an empty dispatch costs **46 µs against 2 µs**,
and session creation **32 ms against 0.5 ms** — the Core ML figure is a
compiler run, not bookkeeping. The consequence is visible in the same test:
a 256-cube matmul takes 64 µs on the ANE, of which ~46 µs is the ask, so it
lands at ~0.5 TFLOPS against a 6.2 TFLOPS peak. The CPU EP needs 430 µs for
the same matmul despite its 20x cheaper dispatch. That crossover — cheap to
ask but slow to compute, versus expensive to ask but fast once asked — is
the whole reason a device advertising tens of TOPS can still lose, and no
throughput row in this backend can show it.

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

## A stock onnxruntime is CPU-only

The GPU and NPU providers only exist in a runtime built for them. A default
install enumerates something like Dnnl / XNNPACK / CPU and nothing else, even
on a machine with an obvious GPU in it — which makes the backend look broken
when it is the runtime that cannot reach the hardware. `runAll()` emits a
one-line note when no accelerator provider is present, pointing at
`CLPEAK_ONNXRUNTIME_LIB`, which selects a different runtime library.

## Graphs must avoid optional operator behaviour

Providers implement the common shapes and decline the rest, and under the
CPU-fallback guard one declined node fails the whole session. Two cases found
on real hardware:

- **Broadcasting.** `onnx-dispatch-latency`'s trivial graph multiplied by a
  scalar; the XNNPACK EP implements fixed-shape elementwise ops only and
  refused it, losing the row. Both operands now carry the same shape.
- **Chained ops against one weight.** See the bandwidth section — the ANE
  refuses the program outright.

When a graph works on one provider and not another, suspect the optional
behaviour before the operator.

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
