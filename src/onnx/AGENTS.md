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
| `onnx_model.cpp` | `OnnxGraph` — emits ONNX protobuf wire format directly; `onnxMatMulModel()` / `onnxQdqMatMulModel()` recipes; fp16/bf16 scalar conversions; `onnxOpsetForDtype()` / `onnxMinOrtApiForOpset()` |
| `gemm.cpp` | `runGemm` (`--onnx-gemm`) — single-node MatMul peak. Two scopes, because a test carries one unit: `onnx-gemm-fp` (tflops, fp32 + fp16 + bf16 + fp8 e4m3/e5m2 + fp4 e2m1 + fp4/int4 weight-only) and `onnx-gemm-int` (tops, int8 QDQ) |
| `transfer.cpp` | `runTransferBandwidth` (`--onnx-transfer-bandwidth`) — host→device bandwidth, swept, plus the full offload round trip |
| `activation.cpp` | `runActivation` (`--onnx-activation`) — SiLU, softmax and LayerNorm throughput in GB/s at `onnx-tensor-bw`'s three working-set sizes, each net of a reference graph that reads and reduces the same tensor with no operation applied; the reference is measured once per size and shared by all three |
| `conv.cpp` | `runConv` (`--onnx-conv`) — fp16 convolution peak: 3×3, 1×1 and depthwise 3×3, each swept over feature-map size |
| `numeric_error.cpp` | `runNumericError` (`--onnx-numeric-error`) — relative RMS error per dtype vs an fp32 CPU-EP reference, in ppm |
| `block.cpp` | `runBlock` (`--onnx-block`) — one fixed transformer decoder block in both regimes. Three scopes off six timings, one unit each: `onnx-block-prefill` (tflops, prompt ladder), `onnx-block-decode` (gbps), `onnx-block-latency` (us, prefill pass + context ladder) |
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

**A guard that is too tight is as bad as no guard.** Three of them shipped
too tight and each destroyed or invented a number on first contact with a new
provider: the folding check threw away a correct 124 TOPS int8 reading because
TensorRT's rate improves 9.4x across the ladder and it only allowed 8x; the
activation floor accepted a one-microsecond difference between two ~248 µs
measurements and published 17 TB/s; the transfer check called a real copy no
copy. Set the threshold against what the failure looks like — folding leaves
the time flat, an absent copy leaves it identical — not against what success
is assumed to look like.

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

## TensorRT will hunt for a subgraph it can claim; stop it

When TensorRT cannot parse a graph it does not give up — it splits it and
retries, up to `trt_max_partition_iterations` (**1000** by default), looking for
the largest subgraph it can take. Every attempt logs the same import failure,
which is where a flood of `ONNX initializer A_q cannot be imported into
TensorRT` comes from: dozens of identical errors and four seconds of wall time
for one float8 E5M2 graph it was never going to accept.

None of that search can help here. Every session in this backend disables CPU
fallback, so a partition TensorRT only partly claims fails the session exactly
as a rejected one does. `onnx_session.cpp` sets the limit to **1**, which
answers the only question clpeak asks — all of it, or none. Measured on the
same E5M2 graph that produced the flood: **four seconds and dozens of identical
errors became one millisecond and two lines.**

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

## Opsets are per model, and never bumped globally

Every recipe declares its own opset, defaulting to **17**. That is not a
historical accident: 17 is what ONNX Runtime 1.17 understands, and
`kMinApiVersion` says 1.17 is the oldest runtime this backend loads at all. A
model declaring a newer opset fails to *load* on an older runtime, so a global
bump would take down every row on a machine whose runtime is merely old,
rather than the one row whose datatype is genuinely newer than the install.

Datatypes arrived in the standard in waves, and the newer ones cannot be named
below their opset: float8 needs 19, int4/uint4 and the blocked quantization
scales need 21, float4e2m1 and the MX scale types need 23. `onnxOpsetForDtype()`
holds that mapping and `onnxMinOrtApiForOpset()` turns an opset into the oldest
ORT that parses it — comparable directly against `OrtRuntime::apiVersion`,
since ORT numbers its API after its own minor version. A recipe taking a
datatype parameter calls `setOpset(onnxOpsetForDtype(dtype))` before adding
anything, and the IR version follows the opset in `build()`: the IR version is
what actually gates which `TensorProto` datatypes may appear.

**Never emit `ReduceMax` directly — call `OnnxGraph::reduceMax()`.** Opset 18
moved its `axes` from an attribute to an input. Both spellings mean the same
thing and which one is legal depends only on the declared opset, so a recipe
that raised its opset to reach a new datatype would otherwise break on a
reduction node that has nothing to do with that datatype. Putting the choice
in one helper is what makes raising an opset a one-line change. `setOpset()`
must be called before any node is added, because `reduceMax()` reads the opset
at the moment it emits.

## What the datatype rows cover

`onnx-gemm-fp` measures fp32, fp16, bf16 and both float8 formats;
`onnx-gemm-int` measures int8 QDQ. The split is by **unit**, not by whether a
row is quantized — float8 is a floating-point format and reports TFLOPS, so it
belongs beside fp16 even though its graph is the QDQ shape. Every one of them
also has an `onnx-numeric-error` row, and they are meant to stay in step: a
rate without its accuracy is half a number here.

bf16 is the row that separates hardware with a real bf16 path from hardware
emulating one. It is expressible at opset 17, so it needs no version gate; what
it does need is a provider with the kernel, and there are three distinct ways
that goes wrong, all of them findings rather than defects in the graph:

- **No matmul at all.** The ARM and x86 CPU EPs answer `Could not find an
  implementation for MatMul(13)`.
- **No matmul the provider will own.** The CoreML EP fails the CPU-fallback
  guard: it would have handed the node back to the CPU.
- **A matmul but no reduction.** The CUDA EP, which is why `gemm.cpp` retries
  with the product cast to fp32 (see above).

**`widen()` in `numeric_error.cpp` must know every dtype the rows use.** Its
switch used to treat the integer path as `default:`, so the first float type it
had not been taught — bf16 — would have been reinterpreted as int8 and scaled,
yielding a confident error figure for a tensor that was never widened at all.
It is exhaustive now and returns empty for anything unknown, which the caller
reports. Any new dtype needs a case there in the same change that adds its row.

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

**That choice decides what the number means, so it is worth stating plainly.**
Because the reference multiplies the operands the device was actually handed,
already rounded, the operand rounding is identical on both sides and cancels.
What each row reports is therefore the *arithmetic* plus the width the answer
is kept in — the part that belongs to the hardware — and not the cost of
quantizing the inputs, which is a property of the format, the same on every
device, and would swamp the comparison this test exists for. It is why fp16
reads 207 ppm rather than something far larger, and why the ratios below fall
on exact powers of two: they are ratios of how many mantissa bits the *result*
was stored in.

The fp32 row doubles as a precision-downgrade detector, and on Apple silicon
it resolves which engine actually ran the work. Reference readings, M1 Pro:

Float8, M1 Pro CPU EP (quantized correctly, multiplied in float): e4m3
26546 ppm, e5m2 52824 ppm.

| Row | fp32 | fp16 | bf16 | int8 QDQ |
|-----|------|------|------|----------|
| CPU EP (M1 Pro) | 0.0 ppm | 207 ppm | (no kernel) | 9443 ppm |
| CoreML EP | 0.4 ppm | 214 ppm | (unsupported) | (unsupported) |
| CUDA EP (RTX 5060) | **261 ppm** | 207 ppm | 1659 ppm | 9447 ppm |
| TensorRT EP (RTX 5060) | 261 ppm | **1185 ppm** | 1659 ppm | 9463 ppm |
| TensorRT float8 (RTX 5060) | — | — | e4m3 **26546 ppm** | — |
| CPU EP (Zen 2) | 0.0 ppm | 207 ppm | (no kernel) | **178926 ppm** |
| DirectML (RTX 5060) | **0.6 ppm** | 1185 ppm | — | 9443 ppm |

**bf16 is the row that validates the arithmetic of the whole test.** It reads
1659.32 ppm on both NVIDIA providers, and 1659.32 / 207.39 = **8.000**. bf16
has exactly three fewer mantissa bits than fp16, so if what is being measured
is input rounding and nothing else, the ratio has to be 2³ — and it is, to four
figures, across two providers that agree on nothing else. Alongside CPU-EP fp32
reading exactly 0.0, that is the second place this test proves itself rather
than merely reporting.

The CUDA EP makes the same point from the other direction, and more
usefully. There bf16 and fp16 run at **the same speed** — 40.14 against
40.01 TFLOPS, once the cast fallback lets the bf16 row report at all — while
bf16's error is 1659 ppm against fp16's 207. Identical throughput, eight times
the error: on that provider there is no reason to prefer bf16 unless the values
need its exponent range, and only the pair of rows says so.

TensorRT's float8 accuracy settles the last question about it. It reads
**26545.94 ppm**, the same figure the CUDA and CPU providers report from a
float multiply, so TensorRT accumulates float8 in fp32 — unlike fp16, which it
accumulates in fp16 for the extra rate. Its 75.5 TFLOPS is therefore roughly
1.9x its own fp32-accumulated fp16 rate, which is exactly what a doubled-rate
float8 path should look like. Rate and accuracy agreeing on the same
explanation is the strongest form this test's evidence takes.

The float8 pair says it a third time. On the M1 Pro CPU EP, which quantizes
correctly and then multiplies in floating point (the fusion check says so), the
two formats read 26546 and 52824 ppm — a ratio of **1.990**, and E5M2 has
exactly one fewer mantissa bit than E4M3FN. Three independent confirmations
now, on three different pairs of formats, that what this test measures is
rounding and nothing else.

That same pair carries the finding worth reading twice: **int8 beats both float8
formats on this data**, 9443 ppm against 26546. That is not a defect in float8,
it is what the operands are. `fillTensor` draws uniformly over a fixed range,
which is int8's best case and float8's worst — float8 buys dynamic range, and
uniform data has none to spend. A model's real activations do, which is why
float8 wins in practice and loses here. The row descriptions say so, and anyone
quoting these numbers as "int8 is more accurate than fp8" is quoting the data
distribution rather than the formats.

It also settles what TensorRT does with each format. Its fp16 error is 1185 and
its bf16 error is 1659 — the same as the CUDA EP's bf16 — so it accumulates
fp16 in fp16 and bf16 in fp32, two different choices for two formats of the
same width. The throughput rows agree without being asked: TensorRT reaches
66.5 TFLOPS in fp16 and 38.1 in bf16, and that 38.1 sits beside the CUDA EP's
40.0 for fp32-accumulated fp16. What looks like bf16 being a second-class
format on this card is the well-known half-rate fp32 accumulate of a consumer
GeForce part, and it took a rate row and an accuracy row together to say so.

DirectML is the control that shows the fp32 rows are reading something real:
0.6 ppm on the same card where CUDA and TensorRT both report 261. It computes
fp32 in fp32 where they substitute TF32, and it accumulates fp16 in fp16 like
TensorRT does. Three providers, one GPU, three different arithmetic choices —
none of them stated anywhere but here.

The CUDA fp32 row is the design paying off on a second vendor: 261 ppm is
worse than fp16's, which is not what fp32 arithmetic looks like. It is TF32 —
ten mantissa bits, the same as fp16 — which cuBLAS selects by default. The
"fp32" throughput row on NVIDIA is therefore a TF32 number, and only the
error row says so.

The clearest single demonstration is one card, one graph, two operating
systems. TensorRT reaches 66.4 TFLOPS in fp16 on Linux and 20.4 on Windows —
a threefold gap that looks like a broken installation until the error rows are
read beside it: **1185 ppm on Linux, 207 on Windows**. The fast one
accumulates in fp16; the slow one accumulates in fp32, exactly as the CUDA
provider does, and lands at its fp32 rate of 19.8. The throughput difference
is not a defect, it is an arithmetic choice, and nothing but the accuracy row
distinguishes the two. Their int8 rates match to within 1% (125.1 against
124.0), which is what makes the fp16 gap interpretable rather than mysterious.

TensorRT is also the clearest case of the row earning its place. It runs fp16 at
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

The error row does not vanish when the fusion does, and it must not be read as
if it had. `numeric_error.cpp` profiles its quantized run the same way
`gemm.cpp` does and says in the row's own description when the provider
dequantized and multiplied in floating point, because the number is then what
the quantization scheme costs rather than what this hardware's integer unit
costs. The two tests can legitimately disagree on one provider: the throughput
row picks the activation signedness by which scheme *fuses* and reports
nothing when neither does, while the error row picks by which scheme *runs*,
deliberately, so that x86's unsigned kernel is the one measured (see the Zen 2
row below).

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

## Float4, and the two shapes it is worth trying in

E2M1 is one sign bit, two of exponent and one of mantissa: **eight magnitudes
in total** — 0, 0.5, 1, 1.5, 2, 3, 4, 6 — no infinity and no NaN, every bit
pattern a number. Opset 23, packed two to a byte exactly like int4.

Two rows, because there are two questions:

- **`fp4_e2m1`** puts four bits on *both* operands with a per-tensor scale.
  Unlike int4, which ONNX can never express as a multiply, this one has a real
  chance of fusing: current tensor cores implement E2M1 and a provider that
  builds engines from QDQ patterns may reach them. The fusion check says which
  happened.
- **`fp4_weight`** is the blocked weight-only form, identical to `int4_weight`
  in every respect — same geometry, same block of 32, same fp16 activations —
  except whether the four bits are spent on a float or an integer. That is the
  comparison worth having, and it is only fair because nothing else differs.

**The output scale has to know the format's largest magnitude.** `qdqOutputScale`
maps four sigma of a K-deep dot product onto the widest code the output type
has. For the 8-bit types that is 127; float8 reaches further, but its precision
is scale-invariant so the choice does not matter and the established figures
stay comparable. Float4 tops out at **6**, and dividing by 127 there would
saturate nearly every value it was handed — a row measuring a badly chosen
scale rather than a format.

For the same reason the operands are stored spending float4's whole range:
`onnxQuantScaleFor` returns 1/6, so [-1, 1] uses all eight magnitudes rather
than the five a scale of one would reach.

Measured, RTX 5060 and M1 Pro:

| provider | `fp4_e2m1` | `fp4_weight` | `int4_weight` | fp16 |
|---|---|---|---|---|
| TensorRT | engine build fails | **65.75** | 65.61 | 66.24 |
| CUDA EP | no fp4 dequantize | no fp4 kernel | 39.66 | 39.99 |
| CPU EP | no fp4 kernel | no fp4 kernel | 1.19 | 0.03 |

**The two weight-only rows land on top of each other and on fp16.** 65.75
against 65.61 against 66.24 — whether the four bits are a float or an integer
makes no difference at all, because neither is doing four-bit arithmetic and
the shape is compute-bound either way. That is the clearest confirmation
available that these rows measure the unpack and not the format, and it is why
the pair only earns its place on providers where the two *disagree*: the CUDA
EP fuses int4 into `MatMulNBits` and refuses float4 outright, because
`MatMulNBits` is an integer kernel.

**`fp4_e2m1`'s failure on TensorRT is the most useful result of the phase.**
The graph parses; the engine will not build:

```
MyelinCheckException: cask_op.h:1926: CHECK(output_quantize_axis_.has_value()) failed.
Could not find any implementation for node {ForeignNode[...]}
```

TensorRT is asking for a quantization **axis** that a per-tensor scale does not
have. Its float4 path wants block scaling, not one scale for the whole tensor —
which is to say it wants NVFP4, and a symmetric per-tensor E2M1 graph is not a
shape it implements. That is worth knowing before attempting NVFP4 rather than
after.

**And NVFP4 is closer than MXFP4 for a reason worth recording.** Its block scale
is `FLOAT8E4M3FN`, which is opset 19 and already implemented here, so the graph
needs nothing newer than float4's own opset 23. MXFP4's scale is `FLOAT8E8M0`,
which the vendored header dates to ONNX **1.21** — a later opset than anything
this backend emits. The two are not one step, and NVFP4 is the near one.

ONNX Runtime 1.30's CUDA EP has no float4 `DequantizeLinear` and 1.29's CPU EP
has no float4 kernel at all — `Could not find an implementation for
DequantizeLinear(23)`. Both are the correct answer. Float4 is very new.

**NVFP4 is here; MXFP4 is not**, and the difference is the scale type. NVFP4's
block scale is `FLOAT8E4M3FN` — opset 19, already implemented — so an NVFP4
graph needs nothing newer than float4's own opset 23. MXFP4's is `FLOAT8E8M0`,
which the vendored header dates to ONNX **1.21**, a later opset than anything
this backend emits. Park MXFP4 until a runtime carries that type.

The second, global scale turned out not to be the obstacle it looked like.
NVFP4 scales twice — an E4M3 scale per block of 16 along the reduction axis,
and one fp32 scale for the whole tensor — and the worry was that the second
would have to become a `Mul` sitting exactly where nothing may sit. It does not.
ONNX expresses it as **a dequantize feeding a dequantize**: the block scales are
themselves dequantized by the global one *before* they scale the data. That puts
the second level on the scale path, leaving the data's dequantize adjacent to
the matmul, which is what ORT needs in order to keep recognising a quantized
matmul at all.

`onnxResidentNvfp4MatMulModel` builds it, on **both** operands — blocked along
the reduction axis, which is axis 1 for the activations and axis 0 for the
weights, the same axis under two different numbers. Both narrow is the point:
every other four-bit row here unpacks into something wider, and this is the only
shape in which a number would be genuine four-bit arithmetic.

The block scale is rounded to E4M3 *before* the data is quantized against it,
not after. Recovering a slightly different scale than the one intended is part
of what the format costs, and quantizing against the value that will actually
come back is what keeps the row measuring NVFP4 rather than an idealisation of
it. The global scale is a power of two so that factoring it out is exact.

**The NVFP4 accuracy row exists, and two things about its shape are
load-bearing.**

*The activations arrive at run time.* An accuracy model has to return real
values, so it cannot reduce them away — and with both operands left as
initializers TensorRT would be free to fold the product while building its
engine and answer in something wider. That reads as **better** accuracy rather
than as a failure, and unlike the throughput row, whose guard watches time scale
with problem size, an accuracy row has no tell at all. Handing the activations
in as fp32 and quantizing them on device against supplied block scales removes
the possibility rather than detecting it. It costs nothing: the values passed in
are exactly what the format recovers, so the device's own quantize round-trips
them.

*The result is quantized back to float4 before it returns.* Without that the row
would measure accumulation alone and could not be read beside the eight-bit
rows, which all include the width their answer is kept in. It is blocked along
the result's own second axis — the reduction axis of whatever consumes it next —
both because that is what a real four-bit pipeline hands on, and because a
per-tensor float4 quantize is precisely the shape TensorRT already refused. Its
scale is one calibrated value per block, four sigma of a K-deep dot product over
float4's widest magnitude, which is how a real pipeline gets them too.

Expect roughly **four times `fp8_e4m3`'s figure**: E2M1 holds the answer in one
mantissa bit against E4M3's three.

NVFP4 is also the first variant narrow enough to reach 32768 on the size ladder:
its operands come to 1.125 bytes per element against fp16's four, so a size the
wider rows cannot afford is within both the memory budget and the protobuf
ceiling.

## What four bits are worth, measured

TensorRT fuses NVFP4 into an engine of its own and reads **238.66 TFLOPS** on an
RTX 5060. The whole card, in one place:

| row | rate | what actually ran |
|---|---|---|
| fp32 | 19.18 | TF32, per the 261 ppm error row |
| bf16 | 38.08 | fp32 accumulate |
| fp16 | 66.20 | fp16 accumulate |
| fp8_e4m3 | 75.40 | fused, 8-bit |
| int8_qdq | 124.77 TOPS | fused, 8-bit |
| **nvfp4** | **238.66** | **fused, 4-bit** |
| fp4_weight | 65.88 | fp16 arithmetic, 4-bit storage |
| int4_weight | 65.87 | fp16 arithmetic, 4-bit storage |

**The last three rows are the argument for having built all of them.** `nvfp4`
and `fp4_weight` use the same element type on the same card and are **3.6x
apart**, because one is four-bit arithmetic and the other is four bits unpacked
into something wider. No single "fp4" row could have said that, and either one
alone would have been quoted as though it were the format's rate.

Against the eight-bit rows, `nvfp4` is **1.91x** int8 and 3.17x float8 — close
enough to the doubling a four-bit tensor core is supposed to deliver over an
eight-bit one that the number needs no defending.

**It is a floor, not a peak.** The ladder was still climbing 10.5% at 32768 and
stopped because the next rung exceeds the operand budget, not because the rate
flattened. Every other row on this card reaches its asymptote; this one runs out
of memory first, which is the size search behaving exactly as designed and worth
knowing when the figure is compared against a vendor's.

**And the per-tensor row stays refused**, with the same
`CHECK(output_quantize_axis_.has_value())` it gave before. That is the pair
working: TensorRT wanted a quantization axis, `nvfp4` has one, `fp4_e2m1` does
not, and the two rows together say what the hardware requires rather than
leaving it to be guessed.

## int4 is a memory format here, not a compute one

**ONNX has no 4-bit matmul.** `MatMulInteger` and `QLinearMatMul` are both
8-bit, and opset 21 added int4 to `DequantizeLinear` and nowhere else. A
symmetric int4 graph — both operands narrow — could therefore only ever
dequantize into a floating-point multiply, which the fusion check would refuse
on every provider forever. There would be no row, only a permanent refusal.

What ONNX *can* express is the form quantized language models actually ship in,
and which AWQ, GPTQ and ORT's own `MatMulNBits` all have: **16-bit activations
against blocked-quantized weights**, one scale per 32 weights along the
reduction axis, no zero point. `onnxResidentWeightOnlyMatMulModel` builds it and
`int4_weight` reports it.

**Its unit is TFLOPS, not TOPS, and that is the whole point.** The weights are
unpacked on the way into the multiply, so the arithmetic is 16-bit and four bits
buys a quarter of the weight traffic rather than a faster multiply. Reporting it
in TOPS beside `int8_qdq` would invite a comparison between a compute rate and a
memory saving. On a square problem it mostly reads near the fp16 row; a provider
well below it is unpacking badly, and a provider well above it — as the M1 Pro
CPU EP is, at 0.52 against fp16's 0.09 — has a fused narrow-weight kernel where
its plain fp16 matmul has nothing.

The fusion check runs here for the same reason it runs on int8, and matters
more: a provider without a fused kernel dequantizes the entire weight matrix
into floats on **every run**, which is a real measurement of a shape nobody
deploys. `MatMulNBits` joins the recognised kernel names.

Measured, RTX 5060 and M1 Pro, ONNX Runtime 1.29/1.30:

| provider | fp16 | `int4_weight` | fused as |
|---|---|---|---|
| TensorRT | 66.4 | **65.9** | `TRTKernel_...` |
| CUDA EP | 40.0 | **39.7** | `MatMulNBits` |
| CPU EP (x86) | 0.03 | **1.25** | `MatMulNBits` + `Cast` |
| CPU EP (M1 Pro) | 0.09 | **0.52** | `MatMulNBits` + `Cast` |

**On hardware with a real fp16 path the two rows agree to within 1%, and that
is the finding.** Unpacking four-bit weights on the way into a 16-bit multiply
costs nothing measurable on either NVIDIA provider — but it also gains nothing
here, because a square GEMM at this size is thoroughly compute-bound: 16384
cubed is 8.8 TFLOP against 128 MB of weights, so the 384 MB of traffic that
four bits saves is lost in the noise. **The row proves the unpack is free; it
does not and cannot show what four-bit weights are for.** That lives in a
decode-shaped workload, where one token reads every weight and nothing else
happens — `onnx-tensor-bw` and `onnx-block-decode` are where a four-bit model
would actually pull ahead, and neither measures a narrow weight type yet.

The CPU providers invert it. Their fp16 matmul has no optimised kernel at all,
so `MatMulNBits` beats it by 42x on x86 and 5.7x on the M1 — and beats plain
fp32 as well. A narrow-weight row above its own fp16 row is not a paradox, it
is a provider that tuned one path and not the other.

**Watch for an inserted `Cast`.** Both CPU providers fuse to `MatMulNBits` and
also cast the fp16 activations to fp32 first, because that is the width their
kernel wants — neither NVIDIA provider does — a full pass over the activations on every run, inside the
figure. The row says so when it happens, which is the same discipline the
throughput models already needed: adding anything to a half-precision graph
risks a silent conversion, so read the executed kernels before believing a
number.

**There is deliberately no `int4_weight` accuracy row.** Under this test's
methodology the reference multiplies the operands the device was handed, already
rounded — so weight rounding cancels, and an int4 row would report the fp16
arithmetic's error, near 207 ppm, sitting next to int8's 9443 and reading as
though four bits were more accurate than eight. What four-bit weights actually
cost is the rounding that cancels: a property of the format, identical on every
device, and outside what these rows measure. An absent row with a reason beats a
present one that misleads.

**TensorRT reaches float8 and clpeak's own numbers say what it costs.** RTX 5060,
ONNX Runtime 1.30: fp8_e4m3 fuses into a `TRTKernel_...` at **75.4 TFLOPS**,
against 66.5 for fp16, 38.1 for bf16, 19.2 for fp32 and 124.6 TOPS for int8.
Read against the accuracy rows that is a coherent picture of one card: fp16 at
66.5 accumulates in fp16, everything accumulating in fp32 sits near 40, and
fp8 at 75.4 is the doubled-rate path. The row worth pausing on is int8, which
beats fp8 by **1.65x** on the same silicon — narrower is not automatically
faster, and on this part the integer tensor cores are simply wider than the
float8 ones.

Two TensorRT limits came out of the same run, both reported as unsupported
rather than guessed at:

- **E5M2 weights are refused outright** — *"ONNX initializer A_q cannot be
  imported into TensorRT"*. It implements E4M3 and not the range-favouring
  format, so the pair of rows is doing exactly its job: one number, one
  refusal, and the refusal is the finding.
- **A float8 tensor cannot cross the EP boundary** — *"TensorRT EP input onnx
  tensor data type: 17 not supported"*. Initializers are fine, which is why
  `onnx-gemm` works: both its operands are baked into the model.
  That is why `onnxQdqMatMulModel` takes `floatIo`: the accuracy model hands
  its input over as the fp32 values it dequantizes to and quantizes them on
  device, then dequantizes the result before it leaves, so the quantized type
  never touches the boundary. It costs nothing in accuracy — every value passed
  in is already exactly representable in the target type, so the added
  `QuantizeLinear` round-trips it rather than rounding twice — and the numbers
  prove it: int8 still reads 9443.09 ppm and float8 26545.94 / 52823.74, to the
  last decimal, before and after the change.

**A float8 QDQ graph must not be fused into `QLinearMatMul`.** ORT's QDQ
selector rewrites DequantizeLinear/MatMul/QuantizeLinear into `QLinearMatMul`,
which is what makes an int8 row an int8 row — but `QLinearMatMul` is an integer
operator with no float8 type constraint, so on a float8 graph the rewrite turns
a valid model into one that fails its own type check: *"Type
'tensor(float8e4m3fn)' of input parameter (A_q) of operator (QLinearMatMul) in
node (n2) is invalid"*. The message reads like clpeak emitted a broken graph and
it did not; the graph was correct until ORT changed it. `onnxCreateSession`
takes `keepQdqUnfused` for exactly this, adding
`QDQSelectorActionTransformer` to the disabled list, and only float8 graphs ask
for it — int8 still fuses and still reports 1.50 TOPS on the M1 Pro CPU EP.
Hardware with real float8 matmul consumes the QDQ nodes itself and never wanted
the rewrite. This is the general lesson again: check what ORT rewrote the graph
into before believing the provider lacks the operator.

**A provider can have a matmul for a datatype and no reduction for it.** The
CUDA EP multiplies bf16 perfectly well — the error row above was measured on it
— but has no bf16 `ReduceMax`, which is the node `onnx-gemm` uses to keep the
result on the device. ORT does not refuse that cleanly: it throws out of
`transformer_memcpy.cc` saying the ReduceMax node has no provider set, rather
than reporting the node as unassigned. `gemm.cpp` retries once with the product
cast to fp32 before the reduction, lazily, so a provider that needs nothing
pays nothing, and the row's description says when the cast was used. The cast
sits after the multiply and cannot change the arithmetic being timed; the
numeric-error row is the independent check, since a matmul quietly promoted to
fp32 would report fp32's rate *and* fp32's error. The quantized form never
needs this — its reduction already runs on a dequantized fp32 result.

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
The latency scope carries both ladders in microseconds, so any of them can be
multiplied by a model's layer count to check a tokens/second claim.

Geometry is fixed in `block.cpp` (2048-wide, 16 heads, SwiGLU 5504, 50.6M
parameters) and deliberately *not* 7B-shaped: the weights must overflow every
cache while still compiling in seconds on NPU toolchains, which build graphs
ahead of time. A 7B block is 4x the size for no extra insight and minutes of
AOT compile. fp16 only — nobody serves an LLM in fp32, so a full-precision
block would measure a configuration that does not exist.

M1 Pro reference: prefill 4.8 TFLOPS against an 8.8 TFLOPS raw fp16 MatMul
peak, so a complete layer retains ~55% of the pure-matmul rate; decode
48 GB/s; 2.5 ms per layer per token. The block still passes its activations
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
- **The runtime scalar scales the result, not an operand.** That leaves the
  matmul a product of two constants, so the graph is correct only while
  constant folding stays disabled — and ONNX Runtime before about 1.18
  accepts `optimization.disable_specified_optimizers` and ignores it. It then
  folds the multiply at load time and the timed phase measures an empty
  graph: 6789 TFLOPS on an RTX 5060 through DirectML, 183000 on its CPU. The
  scaling guard in `gemm.cpp` catches exactly this and reports an error.

  Scaling an *operand* instead would make the graph unfoldable outright, and
  it was tried. It cannot be used: **the CPU provider has no fp16 kernel for
  the multiply**, so it inserts a `Cast` and runs the whole matmul in fp32 —
  the half-precision row came back equal to the single-precision one, 0.41
  against 0.40, measuring the wrong arithmetic entirely. A guarded fold beats
  a silent upcast. Adding any elementwise op to a half-precision graph risks
  this; check the executed kernels for a `Cast` before believing an
  improvement.
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
measurement under the same name. Three tests here would have hit that. Two
search for the asymptote; the third publishes its whole ladder and names each
rung for its working set, which does not expire either:

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

  Its "did anything get copied" test asks only that the time *grew*, not that
  it grew in proportion: a transfer amortises its fixed cost as it gets
  bigger, and TensorRT takes 3437 µs for 16 MB against 11169 µs for 128 MB.
  Demanding proportionality declared that real copy to be no copy at all.

  The base three always run. A flat curve means "main memory reached" only
  when the operation is limited by memory, and a provider limited by its own
  arithmetic is flat from the first rung — ONNX Runtime 1.30's CPU build
  streams this at 2.2 GB/s at every size, where 1.28 gave a clean
  232 / 53 / 20 ladder on the same machine. Stopping on flatness dropped the
  only reading taken at a size no cache could hold.

- **`onnx-activation`** publishes every rung, at `onnx-tensor-bw`'s three
  working-set sizes, and picks nothing. Both single-number rules were tried and
  neither is honest. Its reference graph cannot be made cheap — something has to
  consume the result or the optimiser deletes the operation under test — and on
  TensorRT the reference *scales with the tensor* (175 / 321 / 612 µs at 8 / 16 /
  32 MB), so the **fastest** rung is whichever one the subtraction over-credited
  most: that card's SiLU ladder reads 361 / 202 / 128 GB/s and the 361 comes from
  the rung where 79% of the time was subtracted away. The **largest** rung is no
  better, because the strike rule climbs one rung past the peak on purpose to
  detect the fall, so the last rung measured is usually the collapsed one — and
  whether it is reached at all turns on jitter at the rung before. On an M1 Pro
  that made SiLU alternate between 12.0 and 4.1 GB/s run to run, off a 0.3%
  difference in whether 4096 rows counted as an improvement over 2048.

  A fixed ladder has no selection to be unstable. It also costs less than the
  sweep it replaced (nine measurements plus three shared references, always) and
  it is what makes the comparison this test exists for a division rather than an
  argument: `silu_32mb` over `onnx-tensor-bw`'s `32mb` is one number about one
  working set.

**One ceiling does not come from memory: an ONNX model is a protobuf message,
and protobuf cannot serialize more than 2 GiB.** Both GEMM operands live inside
the model as initializers, so a size the machine has ample memory for can still
be unbuildable — fp32 at 16384 needs exactly 2 GiB of operands and ORT answers
"Model data size exceeds maximum supported size (2GB)". That is a property of
the file format rather than of the device, so it is a hard cap in `gemm.cpp`
alongside the memory budget rather than folded into it.

**Every byte ceiling comes from `clpeak::memoryBudget()`** (`common.h`), a
fraction of physical RAM rather than a constant. The difference between a
workstation and a cheap phone is two orders of magnitude, and a ceiling that
merely wastes time on one is an out-of-memory kill on the other — Android
kills the process outright rather than failing an allocation. The constants
passed to it are the ceilings for a *large* machine; the fraction is what
protects a small one. **This backend runs on phones, not only on desktops with
an NPU**, which is what makes every one of these checks load-bearing rather
than defensive.

**Count the copies, not the tensor.** A graph's weights exist about three times
over at peak: the raw values and the model embedding them overlap while the
model is built, and the model and ORT's own copy overlap while the session is
created. A check against the tensor alone under-estimates by 3x, and on a phone
that is the difference between a skipped row and a killed process. It matters
most where a rung list is *fixed* rather than swept — `onnx-activation` and
`onnx-tensor-bw` both always attempt 8 / 32 / 128 MB, so their budget check is
the only thing that declines the large one, and both multiply by three before
asking.

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

Three rows:

- `h2d` sends a tensor and gathers one element back, so nearly all of it is
  the trip out. Swept, since the rate climbs with size until the link
  saturates.
- `roundtrip` sends a tensor, squares it and returns the result — the real
  cost of offloading a trivial operation. Measured once at the smallest rung,
  never swept: a link saturates rather than improving, and the compile cost
  does not scale gently (see below).
- `d2h` is the round trip minus a third graph that does everything the round
  trip does except ship the result back.

**The return trip must be isolated against that third graph, not against
`h2d`.** Subtracting `h2d` leaves the elementwise pass in the answer, and that
pass is not always cheap — the ANE applies a pointwise operation at roughly a
seventh of the rate it reads memory, which made a naive difference read four
times too slow. The third graph is only affordable because it gathers one
element rather than reducing: a reduction reads the whole result on the
device, and that extra pass would be subtracted out along with the journey
home.

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

## Three scopes, six timings, nothing said twice

`onnx-block` measures two ladders — prompt lengths 64/512/2048 for prefill,
context lengths 512/2048/8192 for decode — and every row comes out of those
six timings. A scope carries one unit, so the split follows the unit and
nothing else:

| scope | unit | rows |
|---|---|---|
| `onnx-block-prefill` | tflops | `s64`, `s512`, `s2048` |
| `onnx-block-decode` | gbps | `kv2048` |
| `onnx-block-latency` | us | `prefill_s512`, `decode_kv512`, `decode_kv2048`, `decode_kv8192` |

**A ladder's headline rung does not also get a scope of its own.** A scope for
prefill's 512 rung, or for decode's 2048 rung in microseconds, republishes a
row verbatim — same timing, same unit, same value, printed twice per device —
and a reader cannot tell a repetition from an independent confirmation. Where a
second scope is genuinely earned it is because the unit differs and the
conversion carries meaning: `onnx-block-decode` exists in GB/s because that is
what compares against `onnx-tensor-bw`, not because decode deserves a header.

The decode rows are why a long conversation answers more slowly than a short
one. Everything but attention costs the same at every length — the weights are
the weights — so what they add as context grows is attention. M1 Pro CPU EP:
8.3 ms, 10.9 ms, 19.6 ms per layer, so attention goes from roughly a tenth of
the layer to three fifths of it.

The prefill rows are the compute-bound counterpart. A short prompt cannot keep
wide hardware busy, so the rate climbs until the device saturates and then
flattens, and where it flattens says how much text must arrive together before
batching stops helping. A CPU is nearly flat across the ladder — ten cores need
very little work in flight — where wide hardware should climb steeply.

The KV ladder stops at 8192 because each rung is a separate graph with its own
cache baked in and therefore its own session, and the ahead-of-time providers
charge dearly for the larger ones. Rows are named by length, so a longer rung
can be appended later without changing what any existing one means.

## The cheap operations are not cheap

A transformer layer is mostly matmul by arithmetic and mostly everything else
by op count. `onnx-activation` measures those others — softmax,
LayerNorm, the feed-forward gate. None does meaningful arithmetic, so their
ceiling is memory bandwidth and their rate is reported as the bandwidth they
achieve — at `onnx-tensor-bw`'s own three working-set sizes, so the two
ladders divide row for row rather than being compared by eye.

M1 Pro, CoreML EP, each rung beside `onnx-tensor-bw`'s rung of the same name
(GB/s, with the share of streaming in brackets):

| | 8mb | 32mb | 128mb |
|---|---|---|---|
| `onnx-tensor-bw` | 88.8 | 45.3 | 50.7 |
| softmax | 30.8 (35%) | 12.4 (27%) | 15.2 (30%) |
| LayerNorm | 12.6 (14%) | 10.7 (24%) | 11.7 (23%) |
| SiLU | 11.1 (13%) | 12.3 (27%) | **4.1 (8%)** |

The ANE applies a pointwise function at roughly a quarter of the rate it
streams weights, and at 128 MB SiLU collapses to a twelfth of it. Hardware
shaped for matrix multiplication has no particular path for these, which is why
a layer can spend a surprising share of its time on the parts that look free in
a FLOP count — and why these are the operations a provider most often hands back
to the CPU. The SiLU cliff reproduces run to run — 4.1, 4.1, 3.7, 4.3 GB/s over
four runs — where any single reported number would have either hidden it or,
worse, published it as the operation's rate. The 8 MB and 32 MB rungs hold to
a few percent between runs; the 128 MB one moves by up to a fifth on a 16 GB
machine, since it is a large allocation competing with everything else.

How much the subtraction matters is entirely a property of the provider and it
spans the whole range: negligible on CoreML, whose reference costs a flat
44–46 µs at every size (under 1% of the measurement), half to three quarters of
the measurement on the CPU EP, and on TensorRT proportional to the tensor. On
the CPU EP the rows are correspondingly noisier than the accelerator ones.

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

The 128 MB rung reads ~51 GB/s on the M1 Pro and `onnx-block-decode` reads
47 GB/s from a completely different graph. Two independent tests landing
together is the best evidence available here that both measure what they
claim.

## Dispatch overhead is why NPUs lose small work

M1 Pro, CoreML EP vs CPU EP: an empty dispatch costs **53 µs against 2 µs**,
and session creation **64 ms against 0.6 ms** — the Core ML figure is a
compiler run, not bookkeeping. The consequence is visible in the same test:
a 256-cube matmul takes 68 µs on the ANE, of which ~53 µs is the ask, so it
lands at ~0.5 TFLOPS against an 8.5 TFLOPS peak. The CPU EP needs 465 µs for
the same matmul despite its 25x cheaper dispatch. That crossover — cheap to
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
