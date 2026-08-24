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
- Looking for the convolution benchmark? → `conv.cpp`
- Looking for the softmax / norm / gate benchmark? → `activation.cpp`
- Looking for the host-transfer benchmark? → `transfer.cpp`
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
| `transfer.cpp` | `runTransferBandwidth` (`--onnx-transfer-bandwidth`) — host→device bandwidth, swept, plus the full offload round trip |
| `activation.cpp` | `runActivation` (`--onnx-activation`) — SiLU, softmax and LayerNorm throughput in GB/s, each net of a reference graph that reads and reduces the same tensor with no operation applied |
| `conv.cpp` | `runConv` (`--onnx-conv`) — fp16 convolution peak: 3×3, 1×1 and depthwise 3×3, each swept over feature-map size |
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

## Verifying a row measured what its name says

The CPU-fallback guard proves the work ran on the right device. It cannot
prove it ran as the right *operation* — ONNX Runtime rewrites graphs before
executing them, and a provider that will not fuse
DequantizeLinear/MatMul/QuantizeLinear into a quantized kernel will instead
dequantize the operands and multiply in floating point. That produces a
perfectly good number which is not an int8 number at all, and publishing it
as one would flatter or penalise the wrong hardware in any comparison.

`onnxCollectExecutedOps()` answers it directly: one profiled run, then the
kernel names ONNX Runtime recorded. The judgement is **inverted** — it looks
for the failure, not the success. A provider that executes ONNX operators one
at a time names each kernel and a fused quantized matmul appears as something
like `QLinearMatMul`, but a provider that compiles whole subgraphs reports one
opaque kernel of its own: TensorRT builds a working int8 engine and calls it
`TRTKernel_graph_clpeak_7216741020808563463_0`. Matching known good names
rejected that. What a *failed* fusion leaves behind is unmistakable, though —
a bare floating-point `MatMul` beside the dequantize nodes — so that is what
is tested for. Anything else ran as the provider's own quantized kernel, and
the fallback guard rules out its having run on the CPU.

That kernel name carries a hash of the graph, so it never reaches a result
row; the description says "a kernel it compiled itself" instead. `gemm.cpp` runs it once for the
quantized variant at the smallest size (the fusion decision does not depend
on size) and reports the row as unsupported, naming what actually ran, when
no integer matmul kernel appears. When one does, the row's description names
it — the M1 Pro CPU EP reports `QLinearMatMul`, so its 1.6 TOPS is real.

**No single quantization scheme fuses everywhere**, which is why the check
also chooses one. A Threadripper reported no fusion where an M1 running the
same code fused to `QLinearMatMul`: the graph used signed activations, and
while ARM's MLAS does S8S8 natively via SDOT, x86 without VNNI implements
only U8S8 and declines the signed form. Switching to U8S8 then broke
TensorRT, which rejects unsigned activations outright (`Found unsupported
input type of UINT8`) and requires a zero point of zero.

So `gemm.cpp` tries signed first, then unsigned, and keeps whichever actually
fuses — the fusion check is the selector, and the row names the scheme it
settled on. Probing happens *before* the sweep, so a provider that fuses
neither costs two small sessions instead of a full ladder. The activation
scale arrives as a runtime input in both, which keeps the dequantize out of
reach of constant folding.

Two details cost time to find and are easy to get wrong again:

- The profile is JSON with `"op_name" : "QLinearMatMul"` — **spaces around
  the colon**. Searching for `"op_name":"` matches nothing, silently, and the
  check then passes everything.
- The profile is written to a file, so the prefix points at the system
  temp directory. A benchmark should not leave files where it was run from.

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
| TensorRT EP (RTX 5060) | 261 ppm | **1185 ppm** | 9463 ppm |
| CPU EP (Zen 2) | 0.0 ppm | 207 ppm | **178926 ppm** |

The CUDA fp32 row is the design paying off on a second vendor: 261 ppm is
worse than fp16's, which is not what fp32 arithmetic looks like. It is TF32 —
ten mantissa bits, the same as fp16 — which cuBLAS selects by default. The
"fp32" throughput row on NVIDIA is therefore a TF32 number, and only the
error row says so.

TensorRT is the clearest case of the row earning its place. It runs fp16 at
66.2 TFLOPS against the CUDA EP's 40.2 on the same card — and its fp16 error
is 1185 ppm against 207. It is 65% faster and nearly six times less accurate,
because it accumulates in fp16 where cuBLAS accumulates in fp32. Neither
number alone is the story; the pair is.

The same two providers on int8 make the other half of the argument. TensorRT
reaches **125 TOPS**, 1.9x its own fp16 rate, which is what int8 tensor cores
are for. The CUDA EP reports nothing at all, because it never fuses a
quantized matmul — it dequantizes and multiplies in floating point. One card,
one graph, and a five-fold difference that depends entirely on which runtime
was asked.

The Zen 2 int8 row is the other half of the argument for measuring accuracy
at all. That CPU fuses the quantized matmul happily and runs it at 1.3 TOPS —
a perfectly respectable throughput row — while losing **18%** of the answer,
twenty times the quantization noise the same recipe costs everywhere else.
The cause is the x86 U8S8 kernel: without VNNI, MLAS multiplies uint8 by int8
into int16 accumulators, which saturate when the weights use the full int8
range. It is the reason ONNX Runtime's quantizer offers `reduce_range` for
pre-VNNI x86. clpeak deliberately does **not** reduce the range — the point
is to report what full-range int8 actually costs on that hardware, and a
throughput row alone would have called it a win.

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
- **Every quantization scale is a build-time constant.** Supplying the
  activation scale as a runtime input is tidier — it keeps the dequantize out
  of constant folding's reach with no optimizer disabled — and ONNX Runtime
  accepts it, since `QLinearMatMul` takes scales as inputs. TensorRT does not:
  it bakes quantization into the engine when it builds, and a scale it cannot
  see until the run leaves it unable to commit to integer arithmetic. It
  compiles, reports an opaque kernel, and quietly delivers 20 TOPS —
  indistinguishable from the same card's fp32 and a third of its fp16.
- **Nothing may sit between the dequantize and the matmul** in the QDQ form,
  or ORT stops recognising a quantized matmul and silently measures float
  arithmetic. The scaling input therefore hangs off the far end.
- **The output quantize must be followed by a dequantize.** Reducing the
  quantized result directly saves a full-width pass and ORT accepts it, but
  TensorRT refuses to build: *"Node n4 cannot be quantized by n3. You might
  want to add a DQ node before n4."* The standard Q-then-DQ shape costs about
  a fifth of the measured rate and is what a real quantized layer does, so it
  is what the model uses.

`onnx-block` is built the same way — activations are a constant scaled by a
runtime scalar, result reduced to one row. Scaling by a runtime value keeps
every node downstream non-constant, so unlike the GEMM graphs the block does
not depend on constant folding being disabled.

`numeric_error.cpp` keeps using the plain, non-resident models: it compares
actual output values, so it needs the real result rather than a reduction.

## Report asymptotes, not readings at a size someone picked

A number tied to a fixed problem size has an expiry date. Whatever size looks
generous today will one day be too small to saturate anything, it gets raised,
and every result recorded before that day silently becomes a different
measurement under the same name. Two tests here would have hit that, and both
now search instead:

- **`onnx-gemm`** doubles from 1024 until the rate stops improving, and reports
  the peak with the size that produced it. "The best this device can do at any
  size" means the same thing in ten years as it does now; "the rate at 4096"
  does not. The search is bounded by a predicted per-iteration time and an
  operand-memory ceiling, both of which scale themselves — hardware fast enough
  to make a bigger size cheap is exactly the hardware that should try it. A
  slow provider stops after two or three rungs; the M1 Pro's fp16 curve peaks
  at 2048 and collapses to 0.3 TFLOPS by 8192, which the strikes rule catches.
- **`onnx-tensor-bw`** measures three fixed rungs, then climbs while the rate
  is still falling, which is where the working set has left the last level of
  cache. A fixed top rung would have needed raising already: 128 MB fits
  inside a single AMD Infinity Cache. Rungs are named for their working-set
  size, so adding more never invalidates the ones below.

  The base three always run. A flat curve means "main memory reached" only
  when the operation is limited by memory, and a provider limited by its own
  arithmetic is flat from the first rung — ONNX Runtime 1.30's CPU build
  streams this at 2.2 GB/s at every size, where 1.28 gave a clean
  232 / 53 / 20 ladder on the same machine. Stopping on flatness dropped the
  only reading taken at a size no cache could hold.

**Every byte ceiling comes from `clpeak::memoryBudget()`** (`common.h`), a
fraction of physical RAM rather than a constant. The difference between a
workstation and a cheap phone is two orders of magnitude, and a ceiling that
merely wastes time on one is an out-of-memory kill on the other — Android
kills the process outright rather than failing an allocation. The constants
passed to it are the ceilings for a *large* machine; the fraction is what
protects a small one.

Where a size *is* fixed, it is fixed because it defines the workload rather
than because it was convenient, and it is meant to stay fixed forever:

- **`onnx-block`'s geometry** is a workload definition, in the way a standard
  benchmark names a model. Being fixed, it is the one test that cannot shrink
  to fit, so it checks memory up front and reports every scope as unsupported
  on a device too small — a smaller layer would not be the same test. It is sized to match a real model's layer, not to
  saturate a particular device, and changing it later would break exactly the
  comparability the fixed choice buys. Its output is a rate, which stays
  meaningful as hardware grows; the guard against it becoming *too* small to
  measure is the `onnx-dispatch-latency` row printed beside it, which says
  what a submission costs on the same device.
- **`onnx-numeric-error`'s 1024** is fixed because accumulation depth is part
  of what the error means. Comparing accuracy across devices requires the same
  depth on each.

## Measuring the cable, and what it cost to try

`onnx-transfer-bw` measures the thing every other test here is built to avoid:
the cost of getting data to the provider and back. It decides whether
offloading is worth doing at all, and no vendor quotes it.

Two rows, both honest, where three were attempted:

- `h2d` sends a tensor and returns one value, so nearly all of it is the trip
  out. Swept, since the rate climbs with size until the link saturates.
- `roundtrip` sends a tensor, squares it and returns the result — the real
  cost of offloading a trivial operation. Measured once at the smallest rung,
  never swept: a link saturates rather than improving, and the compile cost
  does not scale gently (see below).

The return trip alone is **not** reported. Subtracting `h2d` from `roundtrip`
leaves the elementwise pass in the answer, and that pass is not always cheap —
the ANE applies a pointwise operation at roughly a seventh of the rate it
reads memory, which made a naive difference read four times too slow. The
correct isolation needs a third graph doing everything the round trip does
except shipping the result back; that graph took Core ML over twenty minutes
to compile, so it was dropped. Where the return trip matters it can be
recovered from `roundtrip` and the provider's own `onnx-activation` rate, on
the same hardware, for nothing.

**Ahead-of-time compilation is the hidden cost in this backend.** Core ML
compiles a 16 MB elementwise graph in seconds and a 64 MB one in more than
ten minutes. Any new test that builds several large graphs needs to count
sessions, not just iterations, and prefer one measurement at a small size
over a sweep.

TensorRT makes the point at the other extreme: building its engines costs
**454 seconds** of session creation on an RTX 5060, against 7.7 s for the
CUDA EP on the same card and the same graphs. Whatever a provider's
throughput, the compile is a separate cost and can dwarf everything else.

Reference wall times, M1 Pro CoreML EP with a warm compile cache: conv 64 s,
block 62 s, gemm 46 s, activation 32 s, bandwidth 11 s, transfer 2 s,
dispatch 1 s, numeric error under 1 s — 218 s for the provider.

A wall-clock deadline on the sweeps was tried and removed. It would have had
to be an invented number — there is no measurement that says how long a
provider *ought* to take — and an invented number that silently truncates a
sweep produces a quiet, incomparable result, which is worse than a slow
honest one. The bounds that remain are all derived from something: predicted
iteration time, measured improvement, and available memory. If a provider
ever hangs again, fix the graph that provokes it, as the transfer test's
round trip was fixed, rather than putting a clock on the symptom.

Those figures are warm. Core ML caches compiled models on disk, so the first
run of a new graph shape on a given machine is far slower than the second,
and a timing taken after a session of development flatters itself.

## What the transfer rows can and cannot say

`h2d` reports the **largest** size measured, not the fastest. A provider may
hand small tensors over by pointer and copy only large ones — Core ML costs
nothing up to 64 MB and then copies at 19 GB/s — so a peak across sizes can
be a bandwidth figure for a transfer that never happened.

Whether a transfer happens at all is decided structurally: if the time does
not grow with the tensor, nothing is being copied, and the row says so rather
than reporting a number. The CPU EP returns the same microsecond for 16 MB
and 128 MB, which divided out to 133 TB/s before this check existed. Testing
the *shape* of the curve rather than comparing against some plausible
bandwidth ceiling means nothing needs revising as links get faster.

**Summarising a tensor with a reduction is the recurring trap in this
backend.** A reduction looks free next to a transfer or a matmul and is not:
it reads the whole tensor on the device, and that pass lands in whatever the
test was trying to isolate. It has bitten three times —

- `h2d` measured the CPU EP's fp16 reduction rate, 4 GB/s, and called it a
  transfer that never happens;
- the `d2h` isolation subtracted a read of the result along with the return
  journey, flattering it by about 40% on the ANE;
- the activation reference graph carries the same pass, which is why those
  rows lean on their subtraction.

Where a graph needs a small output and the tensor itself is the measurement,
**`Gather` one element** instead. A graph input is materialised in full before
any kernel sees it, so the transfer still happens; only the reading stops.

## How decode cost grows with context

`onnx-block-kv-scaling` runs the decode block at 512, 2048 and 8192 tokens of
context. Everything but attention costs the same at every length — the
weights are the weights — so what the row adds as context grows is attention,
and it is why a long conversation answers more slowly than a short one. M1
Pro CPU EP: 14.0 ms, 16.8 ms, 25.8 ms per layer, so attention goes from
roughly a tenth of the layer to half of it.

`onnx-block-prefill-knee` is the compute-bound counterpart: the same layer
given 64, 512 and 2048 tokens at once. A short prompt cannot keep wide
hardware busy, so the rate climbs until the device saturates and then
flattens, and where it flattens says how much text must arrive together
before batching stops helping. The 512 rung reuses the prefill timing already
taken. A CPU is nearly flat across the ladder — ten cores need very little
work in flight — where wide hardware should climb steeply.

The KV ladder stops at 8192 because each rung is a separate graph with its own
cache baked in and therefore its own session, and the ahead-of-time providers
charge dearly for the larger ones. Rows are named by length, so a longer rung
can be appended later without changing what any existing one means.

## The cheap operations are not cheap

A transformer layer is mostly matmul by arithmetic and mostly everything else
by op count. `onnx-activation` measures those others — softmax,
LayerNorm, the feed-forward gate. None does meaningful arithmetic, so their
ceiling is memory bandwidth and their rate is reported as the bandwidth they
achieve, directly comparable with `onnx-tensor-bw` above.

M1 Pro, CoreML EP, against 50–90 GB/s of resident-tensor streaming:

| | rate | share of streaming |
|---|---|---|
| softmax | 30 GB/s | ~40% |
| SiLU | 12 GB/s | ~14% |
| LayerNorm | 13 GB/s | ~15% |

The ANE reads weights four to seven times faster than it can apply a
pointwise function to them. Hardware shaped for matrix multiplication has no
particular path for these, which is why a layer can spend a surprising share
of its time on the parts that look free in a FLOP count — and why these are
the operations a provider most often hands back to the CPU.

Each rate is net of a reference graph that reads and reduces the same
constant with no operation applied, so what is left is the operation rather
than the scaffolding. On providers where the reduction is itself expensive
the subtraction does heavy lifting (on the CPU EP the reference is most of
the measured time), which makes those rows noisier than the accelerator ones.

## Why convolution is measured separately

Accelerators were built for convolution before they were asked to do anything
else, and the gap between `onnx-conv` and `onnx-gemm` is an architectural
number in its own right. M1 Pro, CoreML EP:

| | rate | vs its own fp16 matmul peak (8.7) |
|---|---|---|
| conv 3×3 | 9.5 TFLOPS | **109%** |
| conv 1×1 | 5.1 TFLOPS | 58% |
| depthwise 3×3 | 0.17 TFLOPS | 2% |

The ANE convolves faster than it multiplies, which is what "built for
convolution" looks like in a measurement. The 1×1 row is arithmetically a
matmul applied per pixel and still runs at half the 3×3 rate, so the two
shapes clearly reach different machinery. And depthwise collapses by **56×**
against the dense 3×3 of identical shape — it loads the same data for a
fraction of the arithmetic, so it is bandwidth-bound, which is exactly why
mobile-efficient networks so often run slower than their FLOP counts promise.
A general-purpose CPU shows none of this spread (0.36 / 0.43 / 0.11).

## Sizes are swept, never chosen by a probe

`gemm.cpp` runs a fixed 1024/2048/4096 ladder and reports each datatype's
best. Picking one size from a timing probe — the pattern `mps-gemm` uses —
was tried first and is unstable: the estimate comes out of a cube root and
is then bucketed, so a 2% wobble in the probe can push it across a bucket
edge, and the size changes the answer. On the M1 Pro the fp16 row alternated
between 5.8 and 6.2 TFLOPS run to run purely on which side it landed.

No single size is right anyway, which the ladder makes visible:

| CoreML EP | 1024³ | 2048³ | 4096³ | 8192³ |
|-----------|-------|-------|-------|-------|
| fp32 | 2.11 | 2.09 | **2.30** | 1.84 |
| fp16 | 4.53 | **8.56** | 6.04 | 0.33 |

fp32 climbs to 4096 while fp16 peaks at 2048 and *falls* — the two rows are
served by different engines (see the numeric-error section) with different
fast-memory limits, and the fp16 drop is the Neural Engine spilling its
on-chip memory, the same cliff Apple silicon is independently reported to
have between those two sizes. The CPU EP is flat across the ladder, as a
cache-blocked CPU should be.

The per-size budget is 2 s rather than the 5 s a single-size test would use,
since the ladder measures several.

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
