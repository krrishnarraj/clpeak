# Crash gates

Every place clpeak withholds work, or does it differently, because a runtime
**crashes the process** (or corrupts its heap) instead of declining -- and
nothing else.  A refusal the runtime speaks is learned by asking and needs no
list here; so does a format a runtime is too old to have.  These are the
cases where asking is itself the fault: the answer arrives as a signal, and
every row after it is never run.

Each entry is a defect in someone else's release.  **Delete it when that
release is fixed**, re-test with the command given, and let the runtime
answer for itself again.

## ONNX Runtime backend

A fault a *format* provokes anywhere goes in `onnxProviderFenceReason()`
(`src/onnx/onnx_session.cpp`): the gemm probe asks it before building a
variant, so every test that consults the probe cache (gemm, conv,
numeric-error, block) inherits it, and the accuracy row asks again itself.
A fault only one test's graph provokes goes in that test instead --
`variantFence()` in `src/onnx/block.cpp` -- so the rows it does not touch
stay measured.  Lifting either is deleting its `if`.

### TensorRT for RTX: `fp4_e2m1`

- **Gate**: `onnxProviderFenceReason()`, the `NvTensorRTRTXExecutionProvider`
  branch.
- **Withheld**: `onnx_gemm` and `onnx_numeric_error` row `fp4_e2m1`, the
  per-tensor float4 QDQ matmul.  The block-scaled float4 rows (`nvfp4`,
  `fp4_weight`) are a different graph and are still put to the provider.
- **Fault**: `GetCapability` takes the whole graph and the engine build ends
  the process with an access violation (exit 0xC0000005).  Classic TensorRT
  declines the same graph (`CHECK(output_quantize_axis_.has_value())
  failed`: its float4 path wants a block scale); the RTX library does not
  survive its own check.
- **Seen**: NvTensorRTRTX EP 0.3.0 from the Windows ML package 2.30.43, ONNX
  Runtime 1.27.1, RTX 5060, 2026-09-18.  fp32, fp16, bf16 and fp8_e4m3 had
  built and run on the same provider first.
- **To lift**: delete the branch and run `clpeak --onnx --onnx-winml --gemm
  --verbose`.  The 32-cube probe is the graph that died; a build, or a
  refusal carrying a message, is a fixed runtime.

### DirectML: the transformer block's `int8_weight`

- **Gate**: `variantFence()` in `src/onnx/block.cpp`, asked at the top of
  `validateVariant` -- before the fusion probe, which is the session that
  dies.
- **Withheld**: on `DmlExecutionProvider`, every `int8_weight` row of the
  block test (`onnx_block_prefill`, `onnx_block_decode`,
  `onnx_block_latency`).  The whole variant goes because one probe session
  gates all of them and it is prefill-shaped whichever row asked.
  `onnx_gemm`, conv and the accuracy rows still measure the format.
- **Fault**: seven `DequantizeLinear`(int8, one fp16 scale per 32 rows: opset
  21, `block_size=32`, `axis=0`) feeding 2048-wide MatMuls ends the process
  with an integer divide by zero (exit 0xC0000094) inside session creation,
  after ORT's own transformers have finished with the graph and before the
  allocation planner speaks: where the DML provider compiles its fused
  partitions.  int4 escapes because ORT's `DQMatMulToMatMulNBits` rewrites it
  first and takes 4-bit weights only (`Is4BitIntType`, `qdq_selectors.cc`);
  the int8 graph reaches DirectML as a raw opset-21 `DequantizeLinear`, which
  the provider registers with no support query (`OperatorRegistration.cpp`)
  and hands to `DML_DEQUANTIZE_OPERATOR_DESC` (`DmlOperatorQuantization21`).
- **Seen**: ONNX Runtime 1.24.4 ("DirectML legacy"), Intel Arc A380, Windows
  11 25H2, 2026-09-21.  Nothing after it ran -- not the rest of the block,
  not the later tests, not the CPU provider's rows.
- **Wider than the fault, and why**: it needs this provider *and* this graph
  *and* that GPU -- the `int8_weight` matmul ladder runs on the same A380,
  and an RTX 4060, a UHD 630, an Adreno X1-45 and the DirectML CPU all run
  this block.  clpeak cannot key on the third: DirectML is registered with no
  device id, ORT picks the adapter, and a built-in provider's
  `onnx_ep_info_t` carries no hardware identity.  Keying on a guess that is
  wrong on a two-GPU box would take the run down, so the row is withheld on
  every DirectML device and says so.  (Narrowing it needs a DirectML device
  per adapter, which ORT 1.22+ makes possible through `GetEpDevices`; vendor
  alone would not do -- the UHD 630 is Intel and runs the graph.)
- **To lift**: delete the `if` and run `clpeak --onnx --transformer-block
  --verbose` on an Arc.  The `int8_weight` fusion probe is the reproducer,
  about 30 s in, after the int4_weight rows.  A newer DirectML may arrive as
  a Windows ML plugin (`--onnx-winml`) rather than another `onnxruntime.dll`;
  same provider key, same gate.

## Core ML backend

### macOS 27: the block's decode form on the CPU and GPU compute units

- **Gate**: `decodeFence` in `src/coreml/block.cpp`, on
  `coremlOsMajorVersion() >= 27` (`src/coreml/coreml_peak.mm`).
- **Withheld**: `coreml_block_decode` and the decode rows of
  `coreml_block_latency` -- on the CPU unit for every weight format, on the
  GPU unit for the fp16-weight variants (`fp16`, `fp16_explicit`,
  `int8_kv`), whose M=1 projections the planner hands to BNNS.  Prefill of
  the same block compiles everywhere; the Neural Engine, which takes the
  whole graph, runs every form.
- **Fault**: a segfault inside `bnns::GraphCompile`, reached through
  Espresso's CPU-backend lowering pass (`MILCompilerForBnns`).  A signal in a
  system library; nothing to catch.
- **Seen**: macOS 27.0 (26A428), M1 Pro; 26.6 ran everything (int4_weight's
  kv2048 point compiled on 27.0, its next did not).  The gate is on the OS
  major version, so iOS 27 gets it too, untested there.
- **To lift**: on a later release, drop the gate (or narrow it to the range
  of releases known to crash) and run `clpeak --coreml --transformer-block`
  on the CPU and GPU compute units (`--devices coreml:1,coreml:2`).

## LiteRT backend, GPU accelerator (LiteRT 2.2.0)

`src/litert/AGENTS.md` § "What the accelerator libraries do wrong" has the
full account of each.

### bf16, int16x8 and fp8_weight are never sent to the GPU

- **Gate**: `litertPlanFor()` in `src/litert/litert_model.cpp`, the
  `gpuAborts` branches.
- **Withheld**: on the GPU device, `litert_gemm` and `litert_numeric_error`
  rows `bf16`, `int16x8`, `fp8_weight`; `litert_block_*` rows `bf16` and
  `fp8_weight`.
- **Fault**: the graph reader `CHECK`s that a tensor is
  fp32/fp16/int8/uint8/int4/int2/bool/int32 (`object_reader.cc`, "Tensor
  type(INT16) is not supported") and glog aborts on anything else.
- **To lift**: delete the `gpuAborts` calls and run `clpeak --litert --gemm`
  on the GPU device: a declined format comes back as a refusal in the
  runtime's own words, an accepted one is measured.

### Blockwise int4_weight is never sent to the GPU

- **Gate**: `litertPlanFor()`, the `Int4Weight` branch.
- **Withheld**: on the GPU device, `litert_gemm`, `litert_numeric_error` and
  `litert_block_*` rows `int4_weight`.
- **Fault**: the weight conversion transposes the packed nibbles with an
  8-bit 16x16 transpose microkernel
  (`xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon`) that runs past the
  buffer; guard malloc catches it on the first model, a normal run dies some
  rungs later (4096-cubed on an M1 Pro).
- **To lift**: delete the branch and verify under guard malloc
  (`DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib`) before trusting a run
  that merely completes: the overrun completed several runs first.

### The composite block variant's decode rows are not sent to the GPU

- **Gate**: `decodeFence()` in `src/litert/block.cpp`.
- **Withheld**: on the GPU device, `litert_block_decode`
  `fp16_composite_kv2048` and `litert_block_latency`
  `fp16_composite_decode_kv*`.  Its prefill rows run.
- **Fault**: the graph reader `CHECK`-fails `CanReadValue(node_input_index)`
  (`object_reader.cc`) on a `STABLEHLO_COMPOSITE` whose operand is a constant
  tensor, which decode's K/V cache is.
- **To lift**: delete `decodeFence` and run `clpeak --litert
  --transformer-block` on the GPU device.

### The GPU device is withheld when no OpenCL library can be loaded

- **Gate**: `litertUsableDevices()` in `src/litert/litert_peak.cpp`: when the
  registered accelerator is the OpenCL one, an OpenCL library must be
  loadable before any model is sent.
- **Withheld**: the whole GPU device, on a machine with no `libOpenCL` (an
  emulator, a box without a driver).
- **Fault**: `libLiteRtClGlAccelerator` dereferences a null inside `strlen`
  when it compiles its first model, instead of declining.
- **To lift**: delete the check and run `clpeak --litert --list-devices`
  followed by any test on such a machine.

### The Metal GPU environment is torn down and rebuilt every 24 models

- **Gate**: `kGpuModelsPerEnvironment` (24) in
  `src/litert/litert_session.cpp`.  Nothing is withheld; it costs ~300 ms of
  Metal initialisation per rebuild.
- **Fault**: the Metal accelerator leaks a residency set per compiled model
  and `IOGPUMetalCommandQueue` asserts on the 33rd ("command queue residency
  set limit of 32 exceeded").  `enable_metal_residency_set` does not stop it
  and neither does an autorelease pool; only a new command queue does.
- **To lift**: set the constant to 0 (no limit) and run a full
  `clpeak --litert` on Apple silicon: more than 32 compiled models in one
  environment without the assert is a fixed release.
