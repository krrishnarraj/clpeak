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

# ---- Optimization flags ----------------------------------------------------
# -O3/-ffast-math (FMA contraction) are ISA-neutral and always applied.  The
# ISA *baseline* depends on CLPEAK_CPU_NATIVE_ARCH:
#   OFF (default) -> portable: SSE4.2 (x86) / armv8-a (ARM).  Safe to distribute;
#                    higher ISA is reached at runtime by the dispatched kernel TUs.
#   ON            -> -march/-mcpu=native: tuned for the build host, NOT portable.
include(CheckCXXCompilerFlag)
set(_clpeak_cpu_native 0)
if(MSVC)
    target_compile_options(peak_cpu PRIVATE /O2 /fp:fast)
    if(CLPEAK_CPU_NATIVE_ARCH)
        check_cxx_compiler_flag("/arch:AVX512" CLPEAK_CPU_HAS_ARCH_AVX512)
        if(CLPEAK_CPU_HAS_ARCH_AVX512)
            target_compile_options(peak_cpu PRIVATE /arch:AVX512)
        else()
            target_compile_options(peak_cpu PRIVATE /arch:AVX2)
        endif()
        set(_clpeak_cpu_native 1)
    endif()
    # else: MSVC x64 baseline is SSE2 and SSE4.2 intrinsics are available without
    # a flag, so no baseline flag is needed.
else()
    target_compile_options(peak_cpu PRIVATE -O3 -ffast-math)
    set(_clpeak_is_x86 OFF)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86|x86")
        set(_clpeak_is_x86 ON)
    endif()

    if(CLPEAK_CPU_NATIVE_ARCH AND NOT CMAKE_CROSSCOMPILING AND NOT ANDROID AND NOT IOS)
        # Tune for the build host. On Apple/ARM -march=native under-enables, so
        # prefer -mcpu=native there; on x86 -march=native is canonical.
        check_cxx_compiler_flag("-march=native" CLPEAK_CPU_HAS_MARCH_NATIVE)
        check_cxx_compiler_flag("-mcpu=native"  CLPEAK_CPU_HAS_MCPU_NATIVE)
        if(APPLE OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
            if(CLPEAK_CPU_HAS_MCPU_NATIVE)
                target_compile_options(peak_cpu PRIVATE -mcpu=native)
            elseif(CLPEAK_CPU_HAS_MARCH_NATIVE)
                target_compile_options(peak_cpu PRIVATE -march=native)
            endif()
        elseif(CLPEAK_CPU_HAS_MARCH_NATIVE)
            target_compile_options(peak_cpu PRIVATE -march=native)
        endif()
        set(_clpeak_cpu_native 1)
    elseif(_clpeak_is_x86)
        # Portable x86 baseline: SSE4.2 (x86-64-v2, universal since ~2009).
        check_cxx_compiler_flag("-msse4.2" CLPEAK_CPU_HAS_SSE42)
        if(CLPEAK_CPU_HAS_SSE42)
            target_compile_options(peak_cpu PRIVATE -msse4.2)
        endif()
    endif()
    # ARM portable baseline is armv8-a (the aarch64 default), which always has
    # NEON — no flag required.
endif()

# Tell the C++ side whether it was tuned natively (drives the startup note that
# advises a native rebuild for best results).
target_compile_definitions(peak_cpu PRIVATE CLPEAK_CPU_NATIVE_ARCH=${_clpeak_cpu_native})
