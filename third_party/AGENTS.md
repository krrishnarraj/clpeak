# third_party — Vendored Dependencies (git submodules)

| Directory | Purpose |
|-----------|---------|
| `libopencl-stub/` | dlopen-based `libOpenCL` stub + CL headers — lets the Android build link OpenCL without a vendor SDK; the real driver is loaded at runtime (`uses-native-library libOpenCL.so` in the app manifest) |
| `Vulkan-Headers/` | Khronos Vulkan headers, newer than the NDK sysroot copy — placed ahead of it so the Vulkan backend can compile against current spec declarations while linking the NDK loader |

Both are consumed by `src/ffi/android/CMakeLists.txt`. Run
`git submodule update --init` after cloning.
