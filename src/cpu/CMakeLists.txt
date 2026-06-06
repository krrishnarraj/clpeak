if(NOT CLPEAK_ENABLE_CPU)
    return()
endif()

add_library(peak_cpu STATIC
    cpu_peak.cpp
    cpu_device.cpp
    thread_pool.cpp
    compute_float.cpp
    compute_int.cpp
    cpu_matrix.cpp
    bandwidth.cpp
    latency.cpp
)

target_link_libraries(peak_cpu PUBLIC peak_common)
if(TARGET Threads::Threads)
    target_link_libraries(peak_cpu PUBLIC Threads::Threads)
endif()

target_include_directories(peak_cpu PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)
target_compile_features(peak_cpu PRIVATE cxx_std_17)
target_compile_definitions(peak_cpu    PRIVATE ENABLE_CPU)
# options.cpp / help text live in peak_common and gate CPU flags on ENABLE_CPU.
target_compile_definitions(peak_common PRIVATE ENABLE_CPU)

# ---- Aggressive optimization for the compute/bandwidth kernels -------------
# A "peak" CPU tool only reaches spec with full optimization + native SIMD +
# FMA contraction.  These are scoped to peak_cpu so they don't leak into the
# rest of the build.
include(CheckCXXCompilerFlag)
if(MSVC)
    target_compile_options(peak_cpu PRIVATE /O2 /fp:fast)
    check_cxx_compiler_flag("/arch:AVX512" CLPEAK_CPU_HAS_ARCH_AVX512)
    if(CLPEAK_CPU_HAS_ARCH_AVX512)
        target_compile_options(peak_cpu PRIVATE /arch:AVX512)
    else()
        target_compile_options(peak_cpu PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(peak_cpu PRIVATE -O3 -ffast-math)
    # -march/-mcpu=native target the *build* host, so they are only correct for
    # a native desktop build. For cross builds (Android NDK, iOS device) we keep
    # the toolchain's default target ISA — NEON baseline always works, and the
    # advanced-dtype kernels (fp16/bf16/i8mm) compile out to clean Unsupported
    # rows when the target baseline lacks the feature.
    if(NOT CMAKE_CROSSCOMPILING AND NOT ANDROID AND NOT IOS)
        # On Apple Silicon / ARM, -march=native is conservative and under-enables
        # host features (fp16/bf16/i8mm), so prefer -mcpu=native there; on x86
        # -march=native is the canonical "enable everything this host has" flag.
        check_cxx_compiler_flag("-march=native" CLPEAK_CPU_HAS_MARCH_NATIVE)
        check_cxx_compiler_flag("-mcpu=native"  CLPEAK_CPU_HAS_MCPU_NATIVE)
        if(APPLE OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
            if(CLPEAK_CPU_HAS_MCPU_NATIVE)
                target_compile_options(peak_cpu PRIVATE -mcpu=native)
            elseif(CLPEAK_CPU_HAS_MARCH_NATIVE)
                target_compile_options(peak_cpu PRIVATE -march=native)
            endif()
        else()
            if(CLPEAK_CPU_HAS_MARCH_NATIVE)
                target_compile_options(peak_cpu PRIVATE -march=native)
            elseif(CLPEAK_CPU_HAS_MCPU_NATIVE)
                target_compile_options(peak_cpu PRIVATE -mcpu=native)
            endif()
        endif()
    endif()
endif()
