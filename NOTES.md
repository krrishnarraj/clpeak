# Fenced tests

Rows clpeak deliberately does not run on a particular runtime because that
runtime crashes the process (or corrupts its heap) instead of declining the
graph.  Every other refusal in the tree is learned by asking -- the provider
builds the graph or says why not, and the row reports its words -- and needs
no list.  These are the exceptions: each is a defect in someone else's
release, each row it withholds carries the reason below as its `unsupported`
reason, and each is to be re-tested when that vendor ships a fix and then
removed.  Nothing here is a capability fact about the hardware.

Not listed: the device-loss latches (`onnxDeviceLost()`; `deviceLost` in
`src/litert/litert_peak.cpp`), which abandon a device only after its GPU has
been reset under it; the version gates on formats a runtime does not have
yet; and the iOS Simulator's MPSGraph gate (`mpsGraphSupported`,
`src/metal/mtl_device.mm`), which is a limitation of the simulator rather
than a defect awaiting a fix.

## ONNX Runtime backend

Both entries live in `onnxProviderFenceReason()` (`src/onnx/onnx_session.cpp`).
The gemm probe asks it before building a variant, so every test that consults
the probe cache (gemm, conv, numeric-error, block) inherits the fence; the
accuracy row asks again itself.  Lifting one is deleting its `if`.

### TensorRT for RTX: `fp4_e2m1`

- **Withheld**: `onnx_gemm` and `onnx_numeric_error` row `fp4_e2m1` -- the
  per-tensor float4 QDQ matmul -- on `NvTensorRTRTXExecutionProvider`.
- **Fault**: `GetCapability` takes the whole graph and the engine build ends
  the process with an access violation (exit 0xC0000005).  Classic TensorRT
  declines the same graph (`CHECK(output_quantize_axis_.has_value())
  failed`: its float4 path wants a block scale); the RTX library does not
  survive its own check.
- **Seen**: NvTensorRTRTX EP 0.3.0 from the Windows ML package 2.30.43, ONNX
  Runtime 1.27.1, RTX 5060, Windows, 2026-09-18.  fp32, fp16, bf16 and
  fp8_e4m3 had built and run on the same provider first.
- **Not fenced**: the block-scaled float4 rows `nvfp4` and `fp4_weight`, the
  shape that check asks for.  They are still put to the provider.
- **To lift**: with a newer RTX provider, delete the entry and run
  `clpeak --onnx --onnx-winml --gemm --verbose` on Windows.  The 32-cube
  probe is the graph that died; a build, or a refusal with a message, is a
  fixed runtime.

### DirectML: `int8_weight`

- **Withheld**: on `DmlExecutionProvider`, `onnx_gemm` row `int8_weight`;
  `onnx_block_prefill` `int8_weight_s512`; `onnx_block_decode`
  `int8_weight_kv2048`; `onnx_block_latency` `int8_weight_prefill_s512` and
  `int8_weight_decode_kv2048`.
- **Fault**: the transformer block's `int8_weight` session -- seven
  `DequantizeLinear`(int8, one fp16 scale per 32 rows: opset 21,
  `block_size=32`, `axis=0`) feeding 2048-wide MatMuls -- ended the process
  with an integer divide by zero (exit 0xC0000094) inside session creation,
  after ORT's own transformers had finished and before the allocation planner
  spoke: where the DML provider compiles its fused partitions.  int4 survives
  because ORT's `DQMatMulToMatMulNBits` rewrites it first and takes 4-bit
  weights only (`Is4BitIntType`, `qdq_selectors.cc`); the int8 graph reaches
  DirectML as a raw opset-21 `DequantizeLinear`, which the provider registers
  with no support query (`OperatorRegistration.cpp`) and hands straight to
  `DML_DEQUANTIZE_OPERATOR_DESC`.
- **Seen**: ONNX Runtime 1.24.4 with its DirectML provider, adapter 0 = RTX
  4060 (an Arc A380 was also present), Windows 11 25H2, 2026-09-21, from
  `clpeak --onnx --transformer-block --verbose`.  The same block's fp16 and
  int4_weight forms had built and run first.
- **Scope, and why it is wider than the graph seen to die**: the 32-cube gemm
  probe of the same graph built and ran, but its scale is a single row, which
  DirectML can read as a plain per-column broadcast; every rung the gemm
  ladder times (1024 and up) carries the block's real layout, on the same
  unchecked path.  So the format is withheld at every size.  `int8_qdq` is
  per-tensor and unaffected; `int4_weight` is fused away before DirectML sees
  it.
- **To narrow**: delete the entry and run `clpeak --onnx --onnx-lib
  <onnxruntime.dll with DirectML> --gemm --verbose` on Windows.  If the
  1024-cube `int8_weight` rung builds and runs, the gemm row is fine and the
  fence belongs in the block alone: `validateVariant` in
  `src/onnx/block.cpp`, keyed on `ep.providerKey`, `v.wDtype == ONNX_DT_INT8`
  and `v.wBlock > 0`.
- **To lift**: with a newer DirectML / ONNX Runtime, delete the entry and run
  `--transformer-block` on the DirectML device; the block's `int8_weight`
  session is the reproducer.

## Core ML backend

### macOS 27: the block's decode form on the CPU and GPU compute units

- **Fence**: `decodeFence` in `src/coreml/block.cpp`, gated on
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
- **To lift**: on a later release, drop the gate (or turn it into the range of
  releases known to crash) and run `clpeak --coreml --transformer-block` on
  the CPU and GPU compute units (`--devices coreml:1,coreml:2` on a Mac).

## LiteRT backend, GPU accelerator (LiteRT 2.2.0)

`src/litert/AGENTS.md` § "What the accelerator libraries do wrong" has the
full account; each fence is one branch.

### bf16, int16x8 and fp8_weight are never sent to the GPU

- **Fence**: `litertPlanFor()` in `src/litert/litert_model.cpp`, the
  `gpuAborts` branches.
- **Withheld**: on the GPU device, `litert_gemm` and `litert_numeric_error`
  rows `bf16`, `int16x8`, `fp8_weight`; `litert_block_*` rows `bf16` and
  `fp8_weight`.
- **Fault**: the graph reader `CHECK`s that a tensor is
  fp32/fp16/int8/uint8/int4/int2/bool/int32 (`object_reader.cc`, "Tensor
  type(INT16) is not supported") and glog aborts on anything else.
- **To lift**: with a newer LiteRT, delete the `gpuAborts` calls and run
  `clpeak --litert --gemm` on the GPU device: a declined format comes back as
  a refusal with the runtime's words, an accepted one is measured.

### Blockwise int4_weight is never sent to the GPU

- **Fence**: `litertPlanFor()`, the `Int4Weight` branch.
- **Withheld**: on the GPU device, `litert_gemm`, `litert_numeric_error` and
  `litert_block_*` rows `int4_weight`.
- **Fault**: the weight conversion transposes the packed nibbles with an
  8-bit 16x16 transpose microkernel
  (`xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon`) that runs past the
  buffer; guard malloc catches it on the first model, a normal run dies some
  rungs later (4096-cubed on an M1 Pro).  Per-row int4 is safe but is not
  the format language models ship in, so nothing is measured under the label.
- **To lift**: delete the branch and verify under guard malloc
  (`DYLD_INSERT_LIBRARIES=/usr/lib/libgmalloc.dylib`) before trusting a run
  that merely completes: the overrun completed several runs first.

### The composite block variant's decode rows are not sent to the GPU

- **Fence**: `decodeFence()` in `src/litert/block.cpp`.
- **Withheld**: on the GPU device, `litert_block_decode` `fp16_composite_kv2048`
  and `litert_block_latency` `fp16_composite_decode_kv*`.  Its prefill rows
  run.
- **Fault**: the graph reader `CHECK`-fails `CanReadValue(node_input_index)`
  (`object_reader.cc`) on a `STABLEHLO_COMPOSITE` whose operand is a constant
  tensor, which decode's K/V cache is.
- **To lift**: delete `decodeFence` and run `clpeak --litert
  --transformer-block` on the GPU device.

### Related guards (nothing withheld on a working machine)

- **No loadable OpenCL library**: the GPU device is not offered
  (`litertUsableDevices()`, `src/litert/litert_peak.cpp`), because
  `libLiteRtClGlAccelerator` dereferences a null inside `strlen` compiling
  its first model instead of declining.
- **Metal residency-set leak**: one per compiled model, and
  `IOGPUMetalCommandQueue` asserts on the 33rd, so the GPU environment is
  rebuilt every `kGpuModelsPerEnvironment` (24) models
  (`src/litert/litert_session.cpp`), ~300 ms each.  Set it to 0 when a
  release stops leaking.
