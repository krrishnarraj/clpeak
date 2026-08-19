# third_party — Vendored Dependencies

## Git submodules

| Directory | Purpose |
|-----------|---------|
| `libopencl-stub/` | dlopen-based `libOpenCL` stub + CL headers — lets the Android build link OpenCL without a vendor SDK; the real driver is loaded at runtime (`uses-native-library libOpenCL.so` in the app manifest) |
| `Vulkan-Headers/` | Khronos Vulkan headers, newer than the NDK sysroot copy — placed ahead of it so the Vulkan backend can compile against current spec declarations while linking the NDK loader |

Both are consumed by `src/ffi/android/CMakeLists.txt`. Run
`git submodule update --init` after cloning.

## Checked-in headers

| Directory | Purpose |
|-----------|---------|
| `onnxruntime/` | ONNX Runtime C API headers (`onnxruntime_c_api.h` + its two includes), copied from one upstream release tag. Header-only on purpose: the ONNX backend dlopens the runtime, so no library or SDK is needed to build it. The pinned `ORT_API_VERSION` is the compatibility contract `src/onnx/onnx_runtime.cpp` negotiates down from |

Deliberately **not** a submodule: `microsoft/onnxruntime` is a whole runtime,
so a `--depth 1` clone costs ~814 MB (10,866 files) for three headers, and
no upstream headers-only repo exists. `tool/update_onnx_headers.sh <tag>`
refreshes them and rewrites the recorded pin; see
`third_party/onnxruntime/README.md`.
