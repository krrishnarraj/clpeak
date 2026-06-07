if(NOT CLPEAK_ENABLE_CPU)
    return()
endif()

include(CheckCXXCompilerFlag)

# ---------------------------------------------------------------------------
# Core library: orchestration + runtime dispatcher.  Compiled at the safe
# baseline (no advanced ISA) so it never SIGILLs on an older CPU.  The actual
# SIMD kernels live in per-feature TU object libraries (below), selected at
# runtime by cpu_dispatch.cpp.
# ---------------------------------------------------------------------------
add_library(peak_cpu STATIC
    cpu_peak.cpp
    cpu_device.cpp
    thread_pool.cpp
    cpu_dispatch.cpp
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
target_include_directories(peak_cpu PUBLIC  ${CMAKE_CURRENT_LIST_DIR}/../../include)
target_include_directories(peak_cpu PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_compile_features(peak_cpu PRIVATE cxx_std_17)
target_compile_definitions(peak_cpu    PRIVATE ENABLE_CPU)
target_compile_definitions(peak_common PRIVATE ENABLE_CPU)

# ISA-neutral optimization (FMA contraction etc.) for everything in the backend.
if(MSVC)
    set(CLPEAK_CPU_OPT /O2 /fp:fast)
else()
    set(CLPEAK_CPU_OPT -O3 -ffast-math)
endif()
target_compile_options(peak_cpu PRIVATE ${CLPEAK_CPU_OPT})

# Target arch.
set(_clpeak_x86 OFF)
set(_clpeak_arm OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86|x86")
    set(_clpeak_x86 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    set(_clpeak_arm ON)
endif()

# Add one feature TU: compiles cpu_kernels_tu.cpp with `tag` + extra flags into
# an OBJECT lib, links it into peak_cpu, and registers CLPEAK_TU_<tag> so the
# dispatcher knows it exists.  ARGN = extra compile options (and defs prefixed
# with -D handled by caller via a separate list is overkill; we only need flags).
function(clpeak_add_isa_tu tag)
    set(_t peak_cpu_tu_${tag})
    add_library(${_t} OBJECT ${CMAKE_CURRENT_LIST_DIR}/cpu_kernels_tu.cpp)
    target_include_directories(${_t} PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../../include ${CMAKE_CURRENT_LIST_DIR})
    target_compile_features(${_t} PRIVATE cxx_std_17)
    target_compile_definitions(${_t} PRIVATE ENABLE_CPU CLPEAK_ISA_TAG=${tag} ${CLPEAK_CPU_TU_DEFS})
    target_compile_options(${_t} PRIVATE ${CLPEAK_CPU_OPT} ${ARGN})
    target_link_libraries(peak_cpu PRIVATE ${_t})
    target_compile_definitions(peak_cpu PRIVATE CLPEAK_TU_${tag}=1)
endfunction()

set(CLPEAK_CPU_TU_DEFS "")   # extra per-TU defs (e.g. CLPEAK_CORE_ONLY for MSVC)

if(CLPEAK_CPU_NATIVE_ARCH AND NOT CMAKE_CROSSCOMPILING AND NOT ANDROID AND NOT IOS)
    # -------- Native single-target build (fastest locally, NOT portable) --------
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX512" _clpeak_msvc_avx512)
        if(_clpeak_msvc_avx512)
            clpeak_add_isa_tu(native /arch:AVX512)
        else()
            clpeak_add_isa_tu(native /arch:AVX2)
        endif()
    else()
        check_cxx_compiler_flag("-march=native" _clpeak_has_march_native)
        check_cxx_compiler_flag("-mcpu=native"  _clpeak_has_mcpu_native)
        if((APPLE OR _clpeak_arm) AND _clpeak_has_mcpu_native)
            clpeak_add_isa_tu(native -mcpu=native)
        elseif(_clpeak_has_march_native)
            clpeak_add_isa_tu(native -march=native)
        elseif(_clpeak_has_mcpu_native)
            clpeak_add_isa_tu(native -mcpu=native)
        else()
            clpeak_add_isa_tu(native)
        endif()
    endif()
    target_compile_definitions(peak_cpu PRIVATE CLPEAK_NATIVE_BUILD=1)

else()
    # -------- Portable build: baseline + runtime-dispatched feature TUs ---------
    # Always-on floor, merged ungated by the dispatcher so kernels() is never
    # null.  SSE2 on x86-64; NEON on aarch64; scalar elsewhere.  On Apple,
    # AppleClang defaults to the *host* CPU, which would bake host-only features
    # (e.g. M4 bf16/i8mm) into this ungated floor and SIGILL on an older Mac, so
    # pin it to the oldest Apple Silicon (apple-m1: NEON+fp16+dotprod+fp16fml,
    # common to every Apple Silicon Mac).
    if(APPLE AND _clpeak_arm)
        check_cxx_compiler_flag("-mcpu=apple-m1" _clpeak_apple_m1)
        if(_clpeak_apple_m1)
            clpeak_add_isa_tu(generic -mcpu=apple-m1)
        else()
            clpeak_add_isa_tu(generic)
        endif()
    else()
        clpeak_add_isa_tu(generic)
    endif()

    if(_clpeak_x86)
        if(MSVC)
            # MSVC can't isolate AVX-512 sub-features, so MSVC TUs are core-only
            # (fp32/fp64/int32 + read); advanced dtypes need a GCC/Clang or native build.
            set(CLPEAK_CPU_TU_DEFS CLPEAK_CORE_ONLY)
            clpeak_add_isa_tu(sse42)                       # x64 baseline (SSE2/SSE4.2 intrinsics)
            check_cxx_compiler_flag("/arch:AVX2"   _clpeak_msvc_avx2)
            check_cxx_compiler_flag("/arch:AVX512" _clpeak_msvc_avx512)
            if(_clpeak_msvc_avx2)
                clpeak_add_isa_tu(avx2 /arch:AVX2)
            endif()
            if(_clpeak_msvc_avx512)
                clpeak_add_isa_tu(avx512 /arch:AVX512)
            endif()
            set(CLPEAK_CPU_TU_DEFS "")
        else()
            check_cxx_compiler_flag("-msse4.2" _clpeak_f_sse42)
            if(_clpeak_f_sse42)
                clpeak_add_isa_tu(sse42 -msse4.2)
            endif()
            check_cxx_compiler_flag("-mavx2 -mfma -mf16c" _clpeak_f_avx2)
            if(_clpeak_f_avx2)
                clpeak_add_isa_tu(avx2 -mavx2 -mfma -mf16c)
            endif()
            set(_avx512_core -mavx512f -mavx512bw -mavx512vl -mavx512dq)
            check_cxx_compiler_flag("${_avx512_core}" _clpeak_f_avx512)
            if(_clpeak_f_avx512)
                clpeak_add_isa_tu(avx512 ${_avx512_core})
            endif()
            check_cxx_compiler_flag("${_avx512_core} -mavx512vnni" _clpeak_f_vnni)
            if(_clpeak_f_vnni)
                clpeak_add_isa_tu(avx512vnni ${_avx512_core} -mavx512vnni)
            endif()
            check_cxx_compiler_flag("${_avx512_core} -mavx512bf16" _clpeak_f_bf16x)
            if(_clpeak_f_bf16x)
                clpeak_add_isa_tu(avx512bf16 ${_avx512_core} -mavx512bf16)
            endif()
            check_cxx_compiler_flag("${_avx512_core} -mavx512fp16" _clpeak_f_fp16x)
            if(_clpeak_f_fp16x)
                clpeak_add_isa_tu(avx512fp16 ${_avx512_core} -mavx512fp16)
            endif()
            if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
                set(_amx ${_avx512_core} -mamx-tile -mamx-int8 -mamx-bf16)
                check_cxx_compiler_flag("${_amx}" _clpeak_f_amx)
                if(_clpeak_f_amx)
                    clpeak_add_isa_tu(amx ${_amx})
                endif()
            endif()
        endif()

    elseif(_clpeak_arm AND NOT APPLE)
        # Linux / Android aarch64: NEON baseline from the generic TU; advanced
        # features via -march=armv8.x-a+<feat> TUs, runtime-gated by HWCAP.
        # (Apple's advanced dtypes come from the apple-m1 generic floor; bf16 /
        # i8mm on M2+ need a native build.)
        check_cxx_compiler_flag("-march=armv8.2-a+fp16"    _clpeak_f_a_fp16)
        if(_clpeak_f_a_fp16)
            clpeak_add_isa_tu(fp16 -march=armv8.2-a+fp16)
        endif()
        check_cxx_compiler_flag("-march=armv8.2-a+fp16fml" _clpeak_f_a_fhm)
        if(_clpeak_f_a_fhm)
            clpeak_add_isa_tu(fp16fml -march=armv8.2-a+fp16fml)
        endif()
        check_cxx_compiler_flag("-march=armv8.2-a+dotprod" _clpeak_f_a_dp)
        if(_clpeak_f_a_dp)
            clpeak_add_isa_tu(dotprod -march=armv8.2-a+dotprod)
        endif()
        check_cxx_compiler_flag("-march=armv8.6-a+bf16"    _clpeak_f_a_bf16)
        if(_clpeak_f_a_bf16)
            clpeak_add_isa_tu(bf16 -march=armv8.6-a+bf16)
        endif()
        check_cxx_compiler_flag("-march=armv8.6-a+i8mm"    _clpeak_f_a_i8mm)
        if(_clpeak_f_a_i8mm)
            clpeak_add_isa_tu(i8mm -march=armv8.6-a+i8mm)
        endif()
    endif()
    # (unknown arch: the generic TU above is the scalar baseline)
endif()
