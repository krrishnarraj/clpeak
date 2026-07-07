if(NOT CLPEAK_ENABLE_CPU)
    return()
endif()

include(CheckCXXCompilerFlag)

# Benchmark kernels are meaningless unoptimized, so force optimization for the
# CPU backend in EVERY config -- a user who forgets `--config Release` (or builds
# the VS-generator default, which is Debug) would otherwise see CPU numbers ~10x
# too low.  GCC/clang already get this for free (the -O3 we append below overrides
# Debug's lack of -O).  MSVC needs help: its Debug config injects /Od /Ob0 /RTC1,
# which respectively disable optimization, disable inlining, and add runtime
# checks (and /RTC1 is mutually exclusive with /O2).  Strip all three by pattern
# -- so we don't depend on the exact flag string of a given MSVC/CMake version --
# and then unconditionally force /O2 /Ob2.  Done with a non-CACHE set() so it is
# directory-scoped: it applies to the CPU targets defined here (incl. the ISA TU
# object libs below) but not to other backends.
if(MSVC)
    foreach(_cfgflag CMAKE_CXX_FLAGS_DEBUG CMAKE_C_FLAGS_DEBUG)
        set(_v " ${${_cfgflag}} ")
        string(REGEX REPLACE " /RTC[1csu]+ " " " _v "${_v}")  # runtime checks (conflict w/ /O2)
        string(REGEX REPLACE " /Ob[0-9] "    " " _v "${_v}")  # inlining control -> let /O2 set /Ob2
        string(REGEX REPLACE " /Od "         " " _v "${_v}")  # disable-optimization
        string(STRIP "${_v}" _v)
        set(${_cfgflag} "${_v} /O2 /Ob2")   # directory-scoped override
    endforeach()
    message(STATUS "clpeak CPU: forcing optimized Debug build -> CMAKE_CXX_FLAGS_DEBUG = [${CMAKE_CXX_FLAGS_DEBUG}]")
endif()

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
target_compile_definitions(peak_cpu    PRIVATE ENABLE_CPU)
target_compile_definitions(peak_common PRIVATE ENABLE_CPU)

# Compiler classification.  clang-cl reports MSVC=TRUE (it drives the MSVC ABI
# and flag style) but is a real clang: CMAKE_CXX_COMPILER_ID is "Clang" with
# SIMULATE_ID "MSVC".  Only genuine cl.exe lacks the GNU -m/-march ISA flags
# and the __ARM_FEATURE_* / AVX-512 sub-feature macros, so the restricted
# "MSVC" paths below gate on _clpeak_real_msvc, not the MSVC variable.  GNU-
# style ISA flags reach clang-cl through its /clang: passthrough prefix.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(_clpeak_real_msvc ON)
else()
    set(_clpeak_real_msvc OFF)
endif()
if(MSVC AND NOT _clpeak_real_msvc)
    set(_clpeak_gnuflag "/clang:")
else()
    set(_clpeak_gnuflag "")
endif()

# ISA-neutral optimization (FMA contraction etc.) for everything in the backend.
# For MSVC, /O2 is intentionally omitted here: CMake injects it automatically via
# CMAKE_CXX_FLAGS_RELEASE for Release builds, and adding it explicitly would
# conflict with /RTC1 (runtime checks) that MSVC's Debug config injects -- the
# two flags are mutually exclusive.  Only /fp:fast needs explicit addition since
# CMake never sets it on its own.  GCC/clang keep -O3 (a step above the default
# -O2) plus -ffast-math.
if(MSVC)
    set(CLPEAK_CPU_OPT /fp:fast)
    if(NOT _clpeak_real_msvc)
        # clang-cl: lift CMake's /O2 (-O2) to -O3 like the GNU-driver builds.
        list(APPEND CLPEAK_CPU_OPT /clang:-O3)
    endif()
else()
    set(CLPEAK_CPU_OPT -O3 -ffast-math)
endif()
target_compile_options(peak_cpu PRIVATE ${CLPEAK_CPU_OPT})

# Target arch.
set(_clpeak_x86 OFF)
set(_clpeak_arm OFF)
set(_clpeak_arm64 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i.86|x86")
    set(_clpeak_x86 ON)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64")
    set(_clpeak_arm ON)
    # The NEON kernels are AArch64-only (fused FMA + horizontal reduce); 32-bit
    # ARMv7 (armeabi-v7a) lacks those, so it gets the scalar generic TU only and
    # none of the armv8.x feature TUs (whose -march=armv8.x flags are invalid in
    # AArch32 mode anyway).
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(_clpeak_arm64 ON)
    endif()
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
    target_compile_definitions(${_t} PRIVATE ENABLE_CPU CLPEAK_ISA_TAG=${tag} ${CLPEAK_CPU_TU_DEFS})
    target_compile_options(${_t} PRIVATE ${CLPEAK_CPU_OPT} ${ARGN})
    target_link_libraries(peak_cpu PRIVATE ${_t})
    target_compile_definitions(peak_cpu PRIVATE CLPEAK_TU_${tag}=1)
endfunction()

set(CLPEAK_CPU_TU_DEFS "")   # extra per-TU defs (e.g. CLPEAK_CORE_ONLY for MSVC)

if(CLPEAK_CPU_NATIVE_ARCH AND NOT CMAKE_CROSSCOMPILING AND NOT ANDROID AND NOT IOS)
    # -------- Native single-target build (fastest locally, NOT portable) --------
    if(_clpeak_real_msvc)
        if(_clpeak_x86)
            check_cxx_compiler_flag("/arch:AVX512" _clpeak_msvc_avx512)
            if(_clpeak_msvc_avx512)
                clpeak_add_isa_tu(native /arch:AVX512)
            else()
                clpeak_add_isa_tu(native /arch:AVX2)
            endif()
        else()
            # MSVC ARM64: NEON is the baseline and the x86 /arch: flags are
            # invalid here; cpu_simd.h selects NEON from _M_ARM64.
            clpeak_add_isa_tu(native)
        endif()
    else()
        check_cxx_compiler_flag("${_clpeak_gnuflag}-march=native" _clpeak_has_march_native)
        check_cxx_compiler_flag("${_clpeak_gnuflag}-mcpu=native"  _clpeak_has_mcpu_native)
        if((APPLE OR _clpeak_arm) AND _clpeak_has_mcpu_native)
            clpeak_add_isa_tu(native ${_clpeak_gnuflag}-mcpu=native)
        elseif(_clpeak_has_march_native)
            clpeak_add_isa_tu(native ${_clpeak_gnuflag}-march=native)
        elseif(_clpeak_has_mcpu_native)
            clpeak_add_isa_tu(native ${_clpeak_gnuflag}-mcpu=native)
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
        if(_clpeak_real_msvc)
            # cl.exe can't isolate AVX-512 sub-features, so its TUs are core-only
            # (fp32/fp64/int32 + read); advanced dtypes need clang(-cl)/GCC or a
            # native build.
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
            # GCC / clang / clang-cl (the ${_clpeak_gnuflag} /clang: prefix
            # routes the GNU -m flags through clang-cl's passthrough).
            check_cxx_compiler_flag("${_clpeak_gnuflag}-msse4.2" _clpeak_f_sse42)
            if(_clpeak_f_sse42)
                clpeak_add_isa_tu(sse42 ${_clpeak_gnuflag}-msse4.2)
            endif()
            set(_avx2 ${_clpeak_gnuflag}-mavx2 ${_clpeak_gnuflag}-mfma ${_clpeak_gnuflag}-mf16c)
            check_cxx_compiler_flag("${_avx2}" _clpeak_f_avx2)
            if(_clpeak_f_avx2)
                clpeak_add_isa_tu(avx2 ${_avx2})
            endif()
            # 256-bit AVX-VNNI: int8 dot for the huge installed base of client
            # x86 without AVX-512 (Alder Lake -> Arrow Lake, Sierra Forest, Zen 5).
            # Implies AVX2; the TU's int8dp comes in via the __AVXVNNI__ path.
            set(_avxvnni ${_avx2} ${_clpeak_gnuflag}-mavxvnni)
            check_cxx_compiler_flag("${_avxvnni}" _clpeak_f_avxvnni)
            if(_clpeak_f_avxvnni)
                clpeak_add_isa_tu(avxvnni ${_avxvnni})
            endif()
            # AVX-VNNI-INT8 (256-bit signed×signed int8 dot): Zen 6, Lunar/Arrow Lake.
            set(_avxvnniint8 ${_avx2} ${_clpeak_gnuflag}-mavxvnniint8)
            check_cxx_compiler_flag("${_avxvnniint8}" _clpeak_f_avxvnniint8)
            if(_clpeak_f_avxvnniint8)
                clpeak_add_isa_tu(avxvnniint8 ${_avxvnniint8})
            endif()
            # AVX10.2 (512-bit): native full-rate bf16 vector FMA (Diamond Rapids).
            set(_avx10bf16 ${_clpeak_gnuflag}-mavx10.2)
            check_cxx_compiler_flag("${_avx10bf16}" _clpeak_f_avx10bf16)
            if(_clpeak_f_avx10bf16)
                clpeak_add_isa_tu(avx10bf16 ${_avx10bf16})
            endif()
            set(_avx512_core ${_clpeak_gnuflag}-mavx512f ${_clpeak_gnuflag}-mavx512bw
                             ${_clpeak_gnuflag}-mavx512vl ${_clpeak_gnuflag}-mavx512dq)
            check_cxx_compiler_flag("${_avx512_core}" _clpeak_f_avx512)
            if(_clpeak_f_avx512)
                clpeak_add_isa_tu(avx512 ${_avx512_core})
            endif()
            check_cxx_compiler_flag("${_avx512_core} ${_clpeak_gnuflag}-mavx512vnni" _clpeak_f_vnni)
            if(_clpeak_f_vnni)
                clpeak_add_isa_tu(avx512vnni ${_avx512_core} ${_clpeak_gnuflag}-mavx512vnni)
            endif()
            check_cxx_compiler_flag("${_avx512_core} ${_clpeak_gnuflag}-mavx512bf16" _clpeak_f_bf16x)
            if(_clpeak_f_bf16x)
                clpeak_add_isa_tu(avx512bf16 ${_avx512_core} ${_clpeak_gnuflag}-mavx512bf16)
            endif()
            check_cxx_compiler_flag("${_avx512_core} ${_clpeak_gnuflag}-mavx512fp16" _clpeak_f_fp16x)
            if(_clpeak_f_fp16x)
                clpeak_add_isa_tu(avx512fp16 ${_avx512_core} ${_clpeak_gnuflag}-mavx512fp16)
            endif()
            # AMX is a 64-bit-only ISA: GCC/clang accept the -mamx-* flags in
            # 32-bit mode (so check_cxx_compiler_flag passes), but <immintrin.h>
            # only declares the _tile_* intrinsics under #ifdef __x86_64__, so an
            # i686 build would compile the AMX TU and then fail with "_tile_loadd
            # was not declared".  Gate on a 64-bit target.  This is in the non-
            # real-MSVC branch, so it covers Linux clang/gcc AND Windows clang-cl
            # (the -mamx-* flags route through ${_clpeak_gnuflag}'s /clang: prefix);
            # the AMX XSTATE grant is arch_prctl on Linux / EnableProcessOptional-
            # XStateFeatures on Windows (cpu_dispatch.cpp).  Real cl.exe is core-
            # only and never reaches here.
            if(CMAKE_SIZEOF_VOID_P EQUAL 8)
                set(_amx ${_avx512_core} ${_clpeak_gnuflag}-mamx-tile
                         ${_clpeak_gnuflag}-mamx-int8 ${_clpeak_gnuflag}-mamx-bf16)
                check_cxx_compiler_flag("${_amx}" _clpeak_f_amx)
                if(_clpeak_f_amx)
                    clpeak_add_isa_tu(amx ${_amx})
                endif()
                # Newer AMX dtypes as their own TUs (each is a separate CPUID
                # feature + a newer compiler; runtime-gated so an older CPU never
                # issues them).  amx-tile is the common base; the -mamx-fp16 /
                # -mamx-tf32 / -mamx-fp8 flags need clang>=~18/20 (older toolchains
                # fail the check and simply skip the TU -> Unsupported row).
                set(_amxfp16 ${_clpeak_gnuflag}-mamx-tile ${_clpeak_gnuflag}-mamx-fp16)
                check_cxx_compiler_flag("${_amxfp16}" _clpeak_f_amxfp16)
                if(_clpeak_f_amxfp16)
                    clpeak_add_isa_tu(amxfp16 ${_amxfp16})
                endif()
                set(_amxtf32 ${_clpeak_gnuflag}-mamx-tile ${_clpeak_gnuflag}-mamx-tf32)
                check_cxx_compiler_flag("${_amxtf32}" _clpeak_f_amxtf32)
                if(_clpeak_f_amxtf32)
                    clpeak_add_isa_tu(amxtf32 ${_amxtf32})
                endif()
                set(_amxfp8 ${_clpeak_gnuflag}-mamx-tile ${_clpeak_gnuflag}-mamx-fp8)
                check_cxx_compiler_flag("${_amxfp8}" _clpeak_f_amxfp8)
                if(_clpeak_f_amxfp8)
                    clpeak_add_isa_tu(amxfp8 ${_amxfp8})
                endif()
            endif()
        endif()

    elseif(_clpeak_arm64 AND NOT APPLE)
        # Linux / Android / Windows-clang aarch64: NEON baseline from the
        # generic TU; advanced features via -march=armv8.x-a+<feat> TUs,
        # runtime-gated (HWCAP on Linux, registry ID registers on Windows).
        # Real MSVC has no -march flags and defines no __ARM_FEATURE_* macros,
        # so it gets the generic (NEON-floor) TU only.
        # (Apple's advanced dtypes come from the apple-m1 generic floor; bf16 /
        # i8mm on M2+ need a native build.)
        if(NOT _clpeak_real_msvc)
            check_cxx_compiler_flag("${_clpeak_gnuflag}-march=armv8.2-a+fp16"    _clpeak_f_a_fp16)
            if(_clpeak_f_a_fp16)
                clpeak_add_isa_tu(fp16 ${_clpeak_gnuflag}-march=armv8.2-a+fp16)
            endif()
            check_cxx_compiler_flag("${_clpeak_gnuflag}-march=armv8.2-a+fp16fml" _clpeak_f_a_fhm)
            if(_clpeak_f_a_fhm)
                clpeak_add_isa_tu(fp16fml ${_clpeak_gnuflag}-march=armv8.2-a+fp16fml)
            endif()
            check_cxx_compiler_flag("${_clpeak_gnuflag}-march=armv8.2-a+dotprod" _clpeak_f_a_dp)
            if(_clpeak_f_a_dp)
                clpeak_add_isa_tu(dotprod ${_clpeak_gnuflag}-march=armv8.2-a+dotprod)
            endif()
            check_cxx_compiler_flag("${_clpeak_gnuflag}-march=armv8.6-a+bf16"    _clpeak_f_a_bf16)
            if(_clpeak_f_a_bf16)
                clpeak_add_isa_tu(bf16 ${_clpeak_gnuflag}-march=armv8.6-a+bf16)
            endif()
            check_cxx_compiler_flag("${_clpeak_gnuflag}-march=armv8.6-a+i8mm"    _clpeak_f_a_i8mm)
            if(_clpeak_f_a_i8mm)
                clpeak_add_isa_tu(i8mm ${_clpeak_gnuflag}-march=armv8.6-a+i8mm)
            endif()
            # SVE (vector-length-agnostic): base compute + int8 SDOT from a plain
            # +sve TU (covers Graviton3/Neoverse V1 without SVE2, and every SVE2
            # core: Neoverse V2, Grace/Vera, Cortex-X925, Oryon Gen3).  BFDOT/
            # BFMMLA and SMMLA ride in separate feature TUs (present on V1 too),
            # runtime-gated by HWCAP2_SVEBF16 / HWCAP2_SVEI8MM.  Apple Silicon has
            # no non-streaming SVE, so this whole block is NOT APPLE.
            check_cxx_compiler_flag("${_clpeak_gnuflag}-march=armv8.2-a+sve"      _clpeak_f_sve)
            if(_clpeak_f_sve)
                clpeak_add_isa_tu(sve ${_clpeak_gnuflag}-march=armv8.2-a+sve)
            endif()
            check_cxx_compiler_flag("${_clpeak_gnuflag}-march=armv8.6-a+sve+bf16" _clpeak_f_svebf16)
            if(_clpeak_f_svebf16)
                clpeak_add_isa_tu(svebf16 ${_clpeak_gnuflag}-march=armv8.6-a+sve+bf16)
            endif()
            check_cxx_compiler_flag("${_clpeak_gnuflag}-march=armv8.6-a+sve+i8mm" _clpeak_f_svei8mm)
            if(_clpeak_f_svei8mm)
                clpeak_add_isa_tu(svei8mm ${_clpeak_gnuflag}-march=armv8.6-a+sve+i8mm)
            endif()
        endif()
    endif()
    # (unknown arch: the generic TU above is the scalar baseline)
endif()
