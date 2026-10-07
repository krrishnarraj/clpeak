# src/coreml — Core ML Backend Implementation

`CoreMLPeak` class implementation: benchmarks run through Apple's Core ML
framework on each of its compute devices.  Built as `peak_coreml` static
library, Apple only (macOS and iOS; the simulator has no Neural Engine).

This backend exists to reach the **Apple Neural Engine** without ONNX
Runtime.  The ANE has no public compute API of any kind — no ISA, no kernel
language, no dispatch call — and Core ML is the only framework that reaches
it (Metal, MPS, MLX and BNNS are all GPU or CPU).  The ONNX backend's CoreML
execution provider is a wrapper on the same framework, with three things it
cannot do that this backend can: prove per operation *which* unit ran the
work, express Core ML's own compressed weight formats, and run on a phone
without an 80 MB static runtime.  Each Core ML compute device (Neural
Engine, GPU, CPU) is enumerated as a device, so one machine gets the same
micro-graphs on all three side by side.

## Quick Lookups

- Looking for the main class (`CoreMLPeak` ctor, `runAll`, device enumeration)? → `coreml_peak.mm`
- Looking for how a model is written, compiled, loaded, and its plan read? → `coreml_session.mm` (+ `coreml_session.h`, the C++ surface the tests use)
- Looking for how ML Program models are emitted without coremltools? → `coreml_model.cpp` + `coreml_model.h`
- Looking for the weight formats and the shared projection recipe? → `coremlEmitWeight` / `coremlEmitProjection` in `coreml_model.cpp`
- Looking for the measurement shape every test follows? → `coreml_bench.h`
- Looking for the matmul ladder? → `gemm.cpp`; the accuracy rows? → `numeric_error.cpp`
- Looking for the transformer block (prefill / decode)? → `block.cpp`
- Looking for why the dispatch and transfer tests are shaped unlike the ONNX ones? → `dispatch_latency.cpp`, `transfer.cpp`, and "The planner" below

## Key Files

| File | Purpose |
|------|---------|
| `coreml_peak.mm` | `CoreMLPeak`: `runAll()`, `enumerate()`; `coremlDevices()` from `MLAllComputeDevices()`; `coremlSpecVersion()` (the newest model spec this OS accepts) |
| `coreml_session.{h,mm}` | `CoremlSession`: writes the `.mlpackage`, compiles it (synchronously, inside `@try`), loads it for the device's `MLModelConfiguration`, loads its `MLComputePlan`, binds inputs to session-owned buffers, times predictions.  `onDevice()` / `offDevice()` / `offDeviceCapable()` are the placement guard |
| `coreml_internal.h` | ObjC helpers shared by the `.mm` files (`coremlKindOf`, `coremlConfigurationFor`); never included from a `.cpp` |
| `coreml_model.{h,cpp}` | `CoremlProgram` — emits `Model.proto` wrapping a `MILSpec.Program` plus the MIL storage-format weight blob; the weight formats (`CoremlWeight`), the projection recipe, and every model recipe (`coremlMatMulChainModel`, `coremlBlockModel`, …); fp16 / bf16 / fp8 conversions |
| `coreml_bench.h` | `coremlMeasure()` (warmup / probe / timed), `coremlBindScalar()`, `coremlOffDeviceReason()`, `coremlNonFiniteReason()` (the readback below), `CoremlLayoutRace` (the weight-layout race below), `coremlHeldBytes()` (what a session holds: every memory gate's count) |
| `gemm.cpp` | `runGemm` (`--gemm`) — `coreml_gemm`: matmul peak per weight format, the ONNX backend's chain (sixteen distinct square layers per prediction from a 64-wide live seed) over a doubling width ladder; fp16, fp32, bf16, int8_weight, int4_weight, int4_lut, fp8_weight in flops, int8_qdq in ops |
| `numeric_error.cpp` | `runNumericError` (`--numeric-error`) — relative RMS error per format vs a double-precision host reference (Accelerate `cblas_dgemm`), in ppm; and the answer checks behind every rate row (`CoreMLPeak::answerCheck` / `wrongAnswer`: per weight layout, and for the convolution rows the product as a 1x1 convolution, `coremlPlainConv1x1Model`) |
| `conv.cpp` | `runConv` (`--convolution`) — 3×3 at stride 2 / 1×1 / depthwise 3×3 at 256 channels, fp16 and fp32, swept over feature-map size |
| `block.cpp` | `runBlock` (`--transformer-block`) — the ONNX backend's decoder block through Core ML: `coreml_block_prefill` (flops, `ops` for int8_qdq), `coreml_block_decode` (bps), `coreml_block_latency` (s) |
| `activation.cpp` | `runActivation` (`--activation`) — SiLU / softmax / layer norm as GB/s at 8/32/128 MB, net of a reference graph |
| `tensor_bandwidth.cpp` | `runTensorBandwidth` (`--tensor-bandwidth`) — GEMV against a resident fp16 weight, 8 MB to 2 GB, net of the dispatch floor |
| `transfer.cpp` | `runTransferBandwidth` (`--transfer-bandwidth`) — h2d / d2h / round trip as differences between three spellings of one matmul |
| `dispatch_latency.cpp` | `runDispatchLatency` (`--kernel-launch-latency`) — the smallest work the planner sends to the unit, and model creation |

Test ids are lower_snake and keep the `coreml_` prefix (`coreml_gemm`,
`coreml_block_decode`); the CLI flags that gate them are backend-neutral
(`--gemm`, `--transformer-block`, …) and shared with the ONNX backend through
the same `Benchmark` values.  Every test is heterogeneous: each reading is a
different format, size or context length.

## Models are emitted as protobuf bytes, not built with coremltools

`coreml_model.cpp` writes the ML Program wire format by hand, exactly as
`src/onnx/onnx_model.cpp` writes ONNX: varints and length-delimited fields,
no protobuf library, no `.mlpackage` asset shipped.  What has to be right,
and was checked against the coremltools sources (`mlmodel/format/MIL.proto`,
`converters/mil/backend/mil/load.py`, `MILBlob/Blob/StorageFormat.hpp`):

- **The package layout.**  `Manifest.json` (`fileFormatVersion` 1.0.0, two
  `itemInfoEntries` authored by `com.apple.CoreML`, a `rootModelIdentifier`),
  `Data/com.apple.CoreML/model.mlmodel`, `Data/com.apple.CoreML/weights/weight.bin`.
  `coreml_session.mm` writes it under `NSTemporaryDirectory()` and removes
  it, and the compiled `.mlmodelc`, when the session dies.
- **The opset name is the spec version minus one**: spec 8 (iOS 17) is
  `CoreML7`, 9 is `CoreML8`, 10 (iOS 26) is `CoreML9`.  It appears twice, as
  `Function.opset` and as the key of `block_specializations`.
- **Every parameter of a generic op is a `const` operation bound by name**,
  including `axes`, `transpose_x` and the dtype string of `quantize`.
- **Compressed-weight ops are serialized two ways.**  The iOS 16/17 ops
  (`constexpr_affine_dequantize`) carry their parameters as *attributes*
  holding Values; the iOS 18 ops (`constexpr_blockwise_shift_scale`,
  `constexpr_lut_to_dense`) take them as *inputs* bound to Values.
- **Which `TensorValue` field a dtype uses**: fp32 in `floats`, int32 in
  `ints`, bool in `bools`, strings in `strings`, and fp16 plus every integer
  narrower than 32 bits — int8, int4 (two per byte, first in the low nibble)
  — as raw `bytes`.
- **The weight blob**: a 64-byte header (count, version 2), then per blob a
  64-byte metadata record at a 64-byte-aligned offset (sentinel `0xDEADBEEF`,
  dtype code, size, data offset) with the data immediately after.
  `BlobFileValue.offset` is the *metadata's* offset and `fileName` is
  `@model_path/weights/weight.bin`.
- **Model I/O** is `ArrayFeatureType` with `FLOAT16` (65552) or `FLOAT32`
  (65568); nothing narrower crosses the boundary.

**Declare the lowest specification version the model needs, never the OS's
newest.**  `coremlSpecNeeded()`: 8 unless the format needs the iOS 18
compression ops (9) or fp8 (10).  A newer opset changes which operation
versions the model carries, and the compute units do not keep up uniformly:
at the macOS 26 opset the M1's GPU has no `ios19.mul`, `quantize` or
`dequantize`, and a graph that ran entirely on the GPU at the iOS 18 opset
had its matmul dragged to the CPU with them.  This is also what coremltools
does with `minimum_deployment_target`.

## The compute plan is the guard

Core ML never refuses a model for lack of a kernel on the unit asked for.
There is no Neural-Engine-only configuration — `cpuAndNeuralEngine` is the
strictest request — and an operation the ANE cannot take is moved to the
CPU without a word.  `MLComputePlan` (macOS 14.4 / iOS 17.4) is what says
so, per operation, and the backend refuses to run on an OS without it.

Every session loads its plan and `onDevice()` judges it by the plan's own
cost estimate: a session passes when the operations placed off its device
carry under `kCoremlOffDeviceShare` (5%) of the estimated cost.  The share
matters because the plan never lists the GPU as able to run a standalone
elementwise op, so the `[1, N]` scalar multiply that closes every
result-scaled graph lands on the CPU on the GPU device — and refusing a
2048-cube matmul over that would be absurd.  A matmul that moved carries
most of the cost and fails whatever else stayed.

Two different things put an operation on the CPU, and the message tells
them apart (`coremlOffDeviceReason`):

- **`supportedComputeDevices` excludes the unit**: it cannot run the
  operation in this form.  fp32 on the ANE; blockwise int4 with live
  activations on an M1's ANE; int4 on the GPU at the macOS 26 opset.
- **The unit is supported but not `preferredComputeDevice`**: the planner
  judged the work too small to send.  Under `cpuAndNeuralEngine` a 64-value
  multiply, a 256-cube matmul and an 8 MB matrix-vector product all stay on
  the CPU; the smallest things the M1 Pro's ANE is ever given are a
  1024-cube matmul with a live input and an 8 MB elementwise op.  The
  timings confirm it is a decision and not a label: a GEMV under the ANE
  configuration takes exactly the CPU configuration's time.

Both are real for an application — no Core ML configuration can force a
unit — so both are reported as unsupported with the reason rather than as
a CPU number under the accelerator's name.  Reading the plan costs as much
as loading the model (it loads the model again); it is paid on every
session because the ANE's acceptance depends on shape and size.

**A plan costed at zero everywhere is judged by count, not weight.**  Core
ML answers that for a model it keeps on the CPU end to end — the ONNX
backend's 64-token transformer block under the same configuration read 26
operations, all `MLCPUComputeDevice`, all `0.000000` — and weighed, such a
plan passes `onDevice()` with every operation off the device (0 ≤ 5%).
`CoremlSession::create` marks every weight unknown when none is positive,
which counts each operation whole and fails the session.  The ONNX backend
reads the same plan through the provider's `ProfileComputePlan` option
(`src/onnx/onnx_coreml_plan.h`) and applies the same rule.

## A ladder reads back what it timed

A timing says nothing about what was computed, and a unit that
miscompiles a graph times it like a correct one.  So once a rung's timed
phase is over, `gemm.cpp`, `conv.cpp` and `block.cpp` read the row the
last prediction reduced to, and `coremlNonFiniteReason()` turns a NaN or
an infinity in it into an `Error` -- the ONNX backend's rule (its
"A ladder reads back what it timed" has the reasoning).  In gemm and conv
that withholds the whole row; in the block, the one point, which is its
own row.  It rests on nothing in these graphs being able to overflow: the
gemm chains keep each layer's magnitude (the reduced row reads 11-48 on an
M1 Pro) and the block's largest value sits 31x under fp16's.  It sees only
the reduced row, so it is a tripwire, not an accuracy test.

## A fast kernel can be a wrong one

The accuracy matmul sees the whole answer, so it doubles as the check every
rate row asks before it publishes (`CoreMLPeak::answerCheck` /
`wrongAnswer`; the line and the words are the three ML backends',
`include/common/answer_check.h`): memoised per compute unit and weight
format, measured by whichever test asks first -- gemm, before each ladder,
since the rate tests run before the accuracy one.

- **Each weight layout is checked on its own**, since a layout can change the
  kernel: the check builds and times both (the accuracy row reads the faster
  of those that answer right) and measures each answer.  A wrong layout
  leaves the gemm and block races, LiteRT's rule for a wrong form, and the
  row says so; with no layout right the rows are withheld.  The block's
  variants ask their projections' format (fp16's for `fp16_explicit` and
  `int8_kv`).
- **The convolution rows ask their precision's product written as a 1x1
  convolution** (`coremlPlainConv1x1Model`, x[1, K, 32, 32]): the conv1x1
  row's operator and the nearest there is to the other shapes'.
- **Wrong is at or past 50% RMS, or NaN or infinity anywhere**
  (`clpeak::kWrongAnswerPpm`; the header says why the line sits there).  A
  withheld rate row is an `Error` naming the figure, and the accuracy row an
  `Error` carrying it, so the pair stands together.  A check that could not
  be measured -- a format the OS lacks, a session refused or sent to another
  unit -- gates nothing.

## Every weight is raced in both layouts

`coremlEmitWeight` stores a weight matrix as `[in, out]`, W[K, N] the way it
multiplies, or as `[out, in]`, its transpose read with `transpose_y` -- the
layout a converted Linear layer carries.  Which one a unit runs faster
depends on the unit, the format and the shape, often by a factor and in
both directions, so no test fixes one: the gemm chain races the two at
every width, the block at every point (its prompt and its cache race
separately, since one unit can take the wide multiply and the one-row one
in opposite layouts), and the accuracy matmul times both and reads the
faster one's answer, because a layout can change the kernel and with it
the arithmetic.  Each row reports the faster and names its layout, with
what the other read beside it.  `CoremlLayoutRace` (`coreml_bench.h`) has
the M1 Pro's cases, and `clpeak::FormRace` (`include/common/form_race.h`)
the rule that closes a race once its readings settle it, so most rows pay
for one extra model; on the Neural Engine, which runs
the two at one rate and compiles `[in, out]` ten times slower, the tie
sends the climb on in `[out, in]`.

## Core ML's compile cache is purged, and free space is checked

E5RT, the runtime behind the GPU and the Neural Engine, caches every model
it specialises under `~/Library/Caches/<process name or bundle id>/com.apple.e5rt.e5bundlecache`
-- the compiled program **with its weights, twice** (a `weights.bin` and an
MPSGraph package) -- and never evicts.  Twelve thousand entries and 292 GB
had accumulated on the development Mac from the ONNX backend's CoreML
provider and this backend together, and one 16384-cube session adds 4.5 GB.
macOS reports that directory as purgeable, so the volume shows hundreds of
gigabytes "available" with a few actually free, and the run died in the
Metal compiler with `LLVM ERROR: IO failure on output stream: No space left
on device`, which nothing can catch.

So (`include/common/coreml_cache.h`, shared with the ONNX backend, whose
`runAll` purges too):

- `clpeak::purgeCoreMLCompileCache()` removes this process's own E5RT cache
  directory, and every `CoremlSession` calls it from its destructor -- every
  session here is a different model, so the cache never repays keeping.  It
  is also called before the first session, which clears what a crashed run
  or an ONNX run left.
- `CoremlSession::create` checks real free space on the temporary volume
  (`statvfs`, which does not count purgeable space) against six times the
  model plus a gigabyte, and refuses the session with the numbers instead.
- The package is deleted as soon as the compiler has copied it, halving
  what sits in `$TMPDIR` during a session, and packages left by dead
  processes (`clpeak-coreml-<pid>-*`) are swept before the first session.

## The planner shapes three tests

The ONNX versions of the dispatch-latency and transfer tests use trivial
graphs, and the resident-bandwidth ladder starts at 8 MB.  All three would
measure the CPU under the Neural Engine's name here, so:

- **`coreml_dispatch_latency`** climbs a ladder of sizes for each row and
  reports the *smallest* the planner sends to the unit, naming it in the
  description.  On the ANE that is an 8 MB elementwise op (~1.2 ms, mostly
  the copies in and out) and a 1024-cube matmul (~0.7 ms); on the CPU the
  ladders end at their first rung (35 µs, 55 µs).  `model_create` is the
  trivial model's compile + load whatever unit it lands on, salted so the
  ANE compile cache cannot answer from a previous run.
- **`coreml_transfer_bw`** builds one 1024-wide fp16 matmul three ways —
  input resident, input handed in, whole result handed back — and reports
  the differences, so the arithmetic cancels and the model is heavy enough
  to be placed.  The two graphs that keep the result on the unit keep one
  *sliced* row of it, not a reduction over all of them: `reduce_max` over
  8192 rows is a pass the ANE runs at ~20 GB/s, it has no counterpart in
  the round-trip graph, and on macOS 27 it outweighed the copy back — the
  whole-result graph ran *faster* than the reduced one (3004 against 3492
  µs at 16 MB), which inverted `d2h` and inflated `roundtrip` (47.6 GB/s;
  33 with the slice).  The trip back must also be a twentieth of the trip
  out at the same size to count as a copy: 3 µs against 51 µs once passed
  the growth test and published 659 GB/s of nothing.  M1 Pro, macOS 27.0:
  ANE h2d 17 GB/s (its ladder ends at 32 MB — the ANE declines a live
  [32768, 1024] input), roundtrip 33, d2h no copy; GPU 28 / 46 / 76 GB/s;
  CPU 15 / 31 / no copy.  On 26.6 the ANE's d2h read 95 GB/s.  `h2d` is
  reported at the largest size and `roundtrip` at the smallest, and each
  row names its size so the two are not read as one subtraction apart.
- **`coreml_tensor_bw`** tries each of its three base rungs even when an
  earlier one was refused: the ANE gets the 128 MB GEMV (45-51 GB/s) and
  not the 8 or 32 MB ones.  The 512 MB rung takes the ANE compiler ~58 s
  and trips the create budget.  **The GPU is never given a matrix-vector
  product at any size** under `cpuAndGPU` — nor the fp16 decode block,
  whose projections are GEMVs — so those rows on the GPU device report the
  planner's refusal, and a Core ML app decoding an fp16 model on a Mac GPU
  is in fact decoding on the CPU.  The GPU does get the int4 / int8 decode
  blocks, whose decompression makes it the cheaper unit in the plan's eyes.

## The activation rows are differences, and are guarded as such

`coreml_activation` reports an operation's cost as the time of a graph with
the operation less the time of the same graph without it.  Two things make
that fragile here and both are handled in `activation.cpp`:

- **Session-to-session variance.**  Two compilations of one and the same
  128 MB graph differ by 15-30% on the M1 Pro's GPU.  Every graph, reference
  and operation alike, is therefore built and timed twice; the minimum of
  each pair is the estimate and the larger spread is a noise floor the
  difference must clear three times over, on top of the ONNX test's rule
  that it be at least a tenth of the whole.  A row that fails says so with
  the numbers.
- **Fusion.**  The GPU absorbs SiLU into the passes around it and applies
  it to 128 MB in ~130 µs of arithmetic; as a bandwidth that is two
  terabytes a second, and a matmul sink after the activation does not make
  it materialise either (measured).  Such a row, when it clears the guards,
  is emitted with a note that it exceeds the rate the unit was seen
  streaming in the reference graph and is the arithmetic's rate, not
  memory's.  There is no physical cap: the reference on the CPU is
  reduction-bound at ~43 GB/s for a tensor the CPU streams at 90-200, so any
  ceiling drawn from it would delete real rows.

## Weight formats, and what each row means

Core ML's arithmetic is fp16 or fp32 and nothing else.  Every narrow format
is *storage*, decompressed into a float multiply by a `constexpr_*` op —
except `int8_qdq`, whose `quantize` / `dequantize` pair around the matmul is
the pattern the compiler fuses into integer arithmetic on Neural Engines
that have it (A17 Pro, M4 and later).  So the narrow rows report flops and
only `int8_qdq` reports ops.  On an M1 the GPU and the CPU read every
format at the fp16 rate, each in the layout it runs that format faster
(above: stored the other way, the GPU's per-channel int8 and the CPU's
blockwise int4 run at a tenth of it or less); the Neural Engine reads every
narrow format faster than fp16, its weight traffic showing even in the gemm
chain.  It shows more where traffic is the whole limit: the block's decode
rows, and the ANE's prefill (below).

| label | stored as | decompression | needs |
|---|---|---|---|
| `fp16` / `fp32` | plain | — | spec 8 |
| `bf16` | bfloat16 constant | none exists | attempted; the parser refuses it at the matmul |
| `int8_weight` | int8, one fp scale per output column | `constexpr_affine_dequantize` | spec 8 |
| `int4_weight` | int4, one fp scale per 32 along K | `constexpr_blockwise_shift_scale` | spec 9 (macOS 15 / iOS 18) |
| `int4_lut` | 4-bit indices into one 16-entry table | `constexpr_lut_to_dense` | spec 9 |
| `fp8_weight` | float8 E4M3, block scales | `constexpr_blockwise_shift_scale` | spec 10; **crashes the parser** (below) |
| `int8_qdq` | int8 weights, int8 activations | `quantize` / `dequantize` + affine weights | spec 8 |

`int8_weight` and `int4_lut` are the two the ANE was designed for.  The same
16-level grid is stored both ways (`int4_weight` blockwise, `int4_lut` one
table) so the two rows differ in nothing but how the levels are found.

**bf16 is in Core ML's type enum and in the blob format and accepted by no
operation.**  The row is attempted so the refusal is Core ML's own sentence
("Param 'y' has incorrect type for operator 'ios17.matmul' … got
tensor<bf16>"), not this backend's assertion.

**fp8 crashes Core ML's model parser** (macOS 26.6): an `NSInvalidArgumentException`
from `insertAdditionalStoragePrecisionForQuantizedWeights` — `-[__NSSetM addObject:]:
object cannot be nil`.  With the asynchronous `compileModelAtURL:completionHandler:`
that exception is raised on a dispatch worker and kills the process; the
session uses the deprecated synchronous `compileModelAtURL:error:` inside
`@try` so it becomes the row's reason.  Keep it that way.  The type exists
so that some OS may take it; the row will say when one does.

**macOS 27.0 crashes compiling the block's decode form for the GPU and
CPU compute units**: a segfault inside `bnns::GraphCompile`, reached
through Espresso's CPU-backend lowering pass (`MILCompilerForBnns`).  On the
CPU unit every weight format's decode crashes (int4_weight's kv2048 point
compiled, the next one did not); on the GPU unit only the fp16-weight
variants (`fp16`, `int8_kv`), whose M=1 projections the planner hands to
BNNS.  Prefill of the same block compiles everywhere, the Neural Engine
runs every form (it takes the whole graph, so BNNS never compiles it), and
26.6 ran everything.  It cannot be caught (no exception, a signal in a
system library), so `block.cpp`'s `decodeFence` keeps those rows off on
macOS >= 27 with the reason in the row; re-test and lift it when a release
fixes it.

## Reference readings, M1 Pro (macOS 26.6)

`coreml_gemm`, peak over the ladder of sixteen-layer chains (macOS 27.0.1):

| row | Neural Engine | GPU | CPU |
|---|---|---|---|
| fp16 | 10.5 TFLOPS (2048-wide; 5.9 at 4096) | 4.7 | 4.5-5.0 |
| fp32 | cannot run (plan: CPU) | 4.4 | 2.0 |
| int8_weight | 11.0 | 4.7 (0.38 stored [out, in]) | 4.7 |
| int4_weight | cannot run: declined with live activations, as in the block | 4.7 (4.4 stored [out, in]) | 5.1 (0.76 stored [in, out]) |
| int4_lut | 11.4 | 4.7 | 4.6 |
| int8_qdq | 11.5 TOPS | 4.6 | 3.6 |

Each cell is the faster of the two layouts; where a cell names the other,
the race was won by a margin, and everywhere else the two tie within 3%.

The ONNX backend's CoreML EP reads 10.5 for its own chain on the same
machine, so the two agree on what they share and this one sees more.  The
single multiply the test timed before read 8.2-8.7 on the Neural Engine,
where one prediction's overhead and the reduction behind it were a fifth of
the time, and 5.0-5.2 fp16 on the CPU, where BNNS ran it with both operands
constant: given a live input the same multiply reads 4.2, under the chain.
The chain costs time instead: the rung that confirms a plateau runs
sixteen multiplies.  Its compiles cost little: on the Neural Engine the
two layouts tie at the first width and the climb goes on `[out, in]`,
whose formats compile in 0.5 / 1.2 / 4 s at 1024 / 2048 / 4096-wide
layers where `[in, out]` takes 4 / 15 / 60 s -- the figures the ONNX
backend's CoreML provider pays.

`coreml_numeric_error` (ppm): fp16 reads **214 on the ANE, 207 on the GPU,
3320 on the CPU** — the CPU's fp16 matmul accumulates in fp16 (it is also
the fastest fp16 unit on the machine at 4.9 TFLOPS), and so does its int4
one, 3218, with the weights [out, in]; stored [in, out] they go to a kernel
seven times slower that accumulates in fp32 and reads 272.  GPU fp32 0.6,
CPU fp32 0.4.  int8_qdq ~9600 everywhere, the cost of keeping the answer in
int8.  int8_weight reads 294 on the ANE against fp16's 214.  The reference uses
the exact scale-times-code values, not fp16-rounded ones: the ANE keeps the
codes and applies the scale after accumulating (`dequantizedTo` in
`coreml_model.cpp` says how that was established).

`coreml_block`, Neural Engine (every row ties between the two layouts and
is measured `[out, in]`):

| row | prefill s512 | decode kv2048 | per token |
|---|---|---|---|
| fp16 | 4.93 TFLOPS | 43.8 GB/s | 2.69 ms |
| fp16_explicit | 4.83 | 48.4 | 2.44 ms (3.40 at 8192, against fp16's 4.74: the cache transpose, see below) |
| int4_lut | **7.85** | 29.1 (of 4-bit bytes) | **1.45 ms** |
| int8_weight | 6.80 | 37.7 | 1.79 ms |
| int8_qdq | 7.97 TOPS | 37.7 | 1.79 ms |
| int4_weight | cannot run: the ANE declines a blockwise int4 matmul with live activations | | |
| int8_kv | | 43.1 | 2.55 ms (3.95 at 8192, against fp16's 4.67) |

**The M1 Pro's Neural Engine is weight-bandwidth-bound even at prefill**: a
palettized block runs 60% faster than fp16 and a per-channel int8 one 38%,
with identical arithmetic.  And **blockwise int4 is accepted only with the
activations resident**: a single matmul over constant activations ran at
8.6 TFLOPS, and neither the layer nor the gemm chain runs, because their
activations are live, as a model's are.

Attention is Core ML's fused `scaled_dot_product_attention` where the OS has
it (spec 9), which is what a converted model carries.  The `fp16_explicit`
row spells it out as matmul, softmax and matmul over a key cache stored
already transposed, `[H, Dh, ctx]` — the ONNX backend's form, so the two
backends' fp16 blocks divide row for row — and sweeps context.  It exists
because the ONNX backend's block, verified on the ANE by its compute plan,
decoded *faster* than this one: 2.21 / 2.45 / 3.41 ms at kv 512 / 2048 /
8192 against the fused op's 2.27 / 2.74 / 4.79 (macOS 27.0), 0.16 µs per
cached token against 0.33.  The explicit row reads 2.20 / 2.44 / 3.40 —
the ONNX figures to the hundredth — and an explicit spelling over the
*untransposed* `[H, ctx, Dh]` cache read 2.26 / 2.73 / 4.53, the fused
op's figures.  So the op is not the cost; the layout is: the fused op
takes keys as `[H, ctx, Dh]`, and on the M1 Pro's Neural Engine that is a
transpose pass over the whole cache every token, which a model storing its
cache transposed does not pay.  At prefill the fused op is slightly ahead
(4.95 against 4.83 TFLOPS at 512 tokens).  The explicit row's 64-token
prompt is one the planner keeps on the CPU (the fused form's goes to the
ANE), which is why `block.cpp` probes each variant at the first point it
reports and lets a point the planner declined for its size stand as that
point's row instead of settling the variant.

## When You Change This Directory

- Adding a benchmark → new `.cpp` here, entry in `CMakeLists.txt`, a call in
  `runAll()` gated on a `Benchmark` value, and a row in Key Files.  Reuse an
  existing value when the measurement already has a name on another backend
  (`include/common/benchmark_enums.h`); a new one needs a flag row in
  `src/common/options.cpp` -- named for what it measures, never for Core ML.
- Adding a weight format → `CoremlWeight` + `coremlEmitWeight` (+ its
  `dequantized` values, which the accuracy reference depends on),
  `coremlWeightBytes`, `coremlSpecForWeight`, `coremlWeightLabel`, and a
  variant row in `gemm.cpp`, `numeric_error.cpp` and `block.cpp` so the three
  tables stay in step.
- Adding a model that multiplies by weight matrices → race their layouts
  (`CoremlLayoutRace`) rather than picking one; which runs faster is the
  unit's and the format's to decide.
- Adding an ObjC helper → declare in `coreml_internal.h`; keep `coreml_session.h`
  free of Objective-C so the tests stay `.cpp`.
- The `.mm` files build with `-fobjc-arc`; the compile call must stay
  synchronous and inside `@try` (fp8).
