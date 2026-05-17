# android/app/src/main/cpp — Android JNI Layer

JNI entry points for the Android app: benchmark execution, device enumeration,
and Android-specific logging.  The shared library (`libclpeak.so`) is loaded
by `BenchmarkRepository.kt`.

## Quick Lookups

- Looking for the JNI entry points? → `entry_android.cpp`, `enumerate_backends.cpp`
- Looking for Android logging? → `logger_android.cpp`
- Looking for the JNI function declarations? → `jni_entry.h`
- Looking for the OpenCL stub loader? → `libopencl-stub/`
- Looking for the CMake build? → `CMakeLists.txt`

## Key Files

| File | Purpose |
|------|---------|
| `entry_android.cpp` | `Java_kr_clpeak_BenchmarkRepository_launchClpeak` — runs benchmarks |
| `enumerate_backends.cpp` | `nativeEnumerateBackends` — device inventory → JSON |
| `logger_android.cpp` | Android logcat + JNI callback logger |
| `jni_entry.h` | JNI function prototypes for Kotlin callbacks |
| `libopencl-stub/` | Stub OpenCL library — dlopens real driver at runtime |
| `CMakeLists.txt` | Builds `libclpeak.so`, compiles common + OpenCL (+ Vulkan) sources |

## When You Change This Directory

- If you add a new JNI entry point → update `jni_entry.h` + the corresponding `.cpp`.
- If you add a new source file → update `CMakeLists.txt`.
- If you change the JSON inventory schema → coordinate with `BackendCatalog.kt`.
- If you change the version setup → see `src/common/cmake/version.cmake` (GIT_ROOT parameter).
