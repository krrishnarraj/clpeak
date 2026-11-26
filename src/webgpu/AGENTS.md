# src/webgpu — WebGPU Backend Implementation

`WebgpuPeak` class implementation: dynamic WebGPU C API loader (`--webgpu-lib`
or standard library search), adapter/device setup, WGSL compute shaders, and
per-benchmark runners. Built as `peak_webgpu` static library with only the
vendored standard `webgpu.h` header (`third_party/webgpu/webgpu.h`, refreshed
via `tools/update_webgpu_headers.sh`) needed at build time.

## Key Files

| File | Purpose |
|------|---------|
| `webgpu_runtime.h` / `webgpu_runtime.cpp` | `dlopen` loader for a `webgpu.h` implementation (or `--webgpu-lib`), X-macro symbol resolution, and `wgpuWaitOne()` future waiter |
| `webgpu_device.cpp` | `WebgpuDevice`: adapter info/limits/feature probing, device creation requesting full adapter limits and optional `ShaderF16` feature, error scopes, buffer/texture/pipeline helpers |
| `webgpu_peak.cpp` | `WebgpuPeak`: instance/adapter enumeration, `runKernel()` calibrated timing, `runAll()`, `enumerate()` |
| `wgsl_shaders.h` | WGSL compute shader generators matching the exact MAD-chain, DP4a, bandwidth, and latency op/byte budgets in `include/common/common.h` |
| `compute_kernel.cpp` | Shared compute-peak runner (output storage + uniform seed buffer, squaring vs alt chain race with `MAX_ALT_CHAIN_RATIO` guard) |
| `compute_float.cpp` | `runComputeSP`, `runComputeHP` (`f16`), `runComputeMP` (`f16 -> f32`) |
| `compute_int.cpp` | `runComputeInt32`, `runComputeInt8DP` (`dot4I8Packed`) |
| `global_bandwidth.cpp` | `runGlobalBandwidth` (`float` / `float2` / `float4`) |
| `local_bandwidth.cpp` | `runLocalBandwidth` (`var<workgroup>`) |
| `image_bandwidth.cpp` | `runImageBandwidth` (`rgba32float` `textureLoad`, 4 walk shapes via WGSL `override` constants) |
| `transfer_bandwidth.cpp` | `runTransferBandwidth` (`copyBufferToBuffer` H2D/D2H) |
| `kernel_latency.cpp` | `runKernelLatency` (empty 1x1x1 dispatch roundtrip) |

## wgpu-native

It departs from `webgpu.h`: some calls abort the process and every future it
returns is null.  `WebgpuRuntime::wgpuNative` marks it; the crash gates are in
the root `NOTES.md`, and a new call into the runtime has to be tried against it.
