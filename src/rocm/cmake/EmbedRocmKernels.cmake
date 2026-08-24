# EmbedRocmKernels.cmake
#
# Ahead-of-time HIP kernel compilation.  Each .hip is compiled by hipcc
# (`--genco`) into a bundled code object covering the requested gfx arches and
# embedded into the binary as a byte array.  At run time the HIP runtime selects
# the slice matching the device's gfx arch -- so the shipped binary needs only
# the HIP runtime (amdhip64), no HIPRTC and no ROCm headers.
#
# Usage (one call per arch group, since kernels differ in their valid arch set
# -- mfma is CDNA-only, wmma is RDNA3+, fp8/mxfp4 are newest-gen only):
#
#   embed_rocm_kernels(
#     TARGET   peak_rocm
#     ARCHS    "gfx908;gfx90a;gfx942;gfx950"   # candidate gfx targets
#     [CXX17]                                  # pass -std=c++17 (rocWMMA)
#     [ROCWMMA]                                # add the rocWMMA include dir
#     KERNELS  mfma_fp16 mfma_bf16 ...         # bare stems under rocm_kernels/
#   )
#
# Candidate arches are intersected with what the installed hipcc can target
# (probed by trial compile); a kernel with no buildable arch gets an empty stub
# Blob so the symbol still links (it is capability-gated off at run time).

set(_CLPEAK_EMBED_ROCM_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Windows-only command prefix that runs hipcc with the MSVC environment's
# include paths scrubbed; empty everywhere else.
#
# Clang honours the INCLUDE environment variable when it targets the MSVC
# toolchain.  With INCLUDE set, HIP *device* compilation fails parsing MSVC's
# own headers: vcruntime.h skips its uintptr_t typedef -- while intptr_t, two
# lines above it in the same file, comes through fine -- and that cascades into
# __clang_hip_math.h not seeing uint64_t and __clang_hip_cmath.h not seeing the
# FP_* classification macros.
#
# What pins this on the environment rather than on the source or the arch list:
# _clpeak_rocm_supported_archs() below compiles a kernel of the same shape as
# the real ones (#include <hip/hip_runtime.h> plus an extern "C" __global__
# function) and succeeds for every candidate arch -- but it runs through
# execute_process() at configure time, where no MSVC environment is present.
# The kernel compiles run inside MSBuild's custom-build step, which injects the
# full MSVC environment.  Same compiler, same headers, same include chain; the
# difference is INCLUDE.
#
# So emptying INCLUDE reproduces the environment the probe already proves works,
# leaving clang to its own MSVC/Windows SDK detection.  EXTERNAL_INCLUDE goes
# too, since clang also consults it for system headers.  Emptied rather than
# unset because cmake -E env --unset needs CMake 3.24 and this project targets
# 3.20.  Device-only compilation never links against host objects, so nothing
# here can skew the ABI of what the build actually ships.
function(_clpeak_hipcc_env_wrap out)
  if(WIN32)
    set(${out} "${CMAKE_COMMAND}" -E env "INCLUDE=" "EXTERNAL_INCLUDE=" PARENT_SCOPE)
  else()
    set(${out} "" PARENT_SCOPE)
  endif()
endfunction()

# Probe once which gfx targets the installed hipcc accepts.  Result cached in a
# global property so repeated embed calls don't re-run the trial compiles.
function(_clpeak_rocm_supported_archs out)
  get_property(_cached GLOBAL PROPERTY CLPEAK_ROCM_SUPPORTED_ARCHS SET)
  if(_cached)
    get_property(_v GLOBAL PROPERTY CLPEAK_ROCM_SUPPORTED_ARCHS)
    set(${out} "${_v}" PARENT_SCOPE)
    return()
  endif()

  set(_candidates
      gfx906 gfx908 gfx90a gfx942 gfx950
      gfx1030 gfx1100 gfx1101 gfx1102 gfx1200 gfx1201)
  set(_probe "${CMAKE_CURRENT_BINARY_DIR}/_clpeak_archprobe.hip")
  file(WRITE "${_probe}" "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void p(){}\n")

  # Same env scrubbing as the real compiles: without it, configuring from a
  # Visual Studio developer prompt (where INCLUDE *is* set) would fail every
  # probe and silently fall through to stub kernels for the whole backend.
  _clpeak_hipcc_env_wrap(_wrap)

  set(_ok "")
  foreach(_a ${_candidates})
    execute_process(
      COMMAND ${_wrap} "${CLPEAK_HIPCC}" --genco --offload-arch=${_a}
              -o "${CMAKE_CURRENT_BINARY_DIR}/_clpeak_archprobe_${_a}.co" "${_probe}"
      RESULT_VARIABLE _r OUTPUT_QUIET ERROR_QUIET)
    if(_r EQUAL 0)
      list(APPEND _ok "${_a}")
    endif()
  endforeach()

  set_property(GLOBAL PROPERTY CLPEAK_ROCM_SUPPORTED_ARCHS "${_ok}")
  message(STATUS "clpeak ROCm: hipcc can target: ${_ok}")
  set(${out} "${_ok}" PARENT_SCOPE)
endfunction()

function(embed_rocm_kernels)
  cmake_parse_arguments(ER "CXX17;ROCWMMA" "TARGET" "ARCHS;KERNELS" ${ARGN})

  _clpeak_rocm_supported_archs(_supported)

  # Intersect requested arches with what hipcc supports.
  set(_final "")
  foreach(_a ${ER_ARCHS})
    list(FIND _supported "${_a}" _idx)
    if(_idx GREATER -1)
      list(APPEND _final "${_a}")
    endif()
  endforeach()

  # genco flags.
  set(_flags "")
  foreach(_a ${_final})
    list(APPEND _flags "--offload-arch=${_a}")
  endforeach()
  if(ER_CXX17)
    list(APPEND _flags "-std=c++17")
  endif()
  if(ER_ROCWMMA AND CLPEAK_ROCWMMA_INCLUDE_DIR)
    list(APPEND _flags "-I${CLPEAK_ROCWMMA_INCLUDE_DIR}")
  endif()

  set(_codir  "${CMAKE_CURRENT_BINARY_DIR}/rocm_codeobjs")
  set(_gendir "${CMAKE_CURRENT_BINARY_DIR}/rocm_kernels_gen")
  file(MAKE_DIRECTORY "${_codir}" "${_gendir}")
  set(_embed "${_CLPEAK_EMBED_ROCM_DIR}/EmbedBin.cmake")

  _clpeak_hipcc_env_wrap(_wrap)

  set(_gen_srcs "")
  foreach(_kn ${ER_KERNELS})
    set(_hip "${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/${_kn}.hip")
    if(NOT EXISTS "${_hip}")
      message(FATAL_ERROR "embed_rocm_kernels: source not found: ${_hip}")
    endif()
    set(_gen "${_gendir}/${_kn}.cpp")

    if(_final)
      set(_co "${_codir}/${_kn}.co")
      add_custom_command(
        OUTPUT  "${_co}"
        COMMAND ${_wrap} "${CLPEAK_HIPCC}" --genco ${_flags} -O3 -o "${_co}" "${_hip}"
        DEPENDS "${_hip}"
        COMMENT "hipcc --genco ${_kn}.hip"
        VERBATIM)
      add_custom_command(
        OUTPUT  "${_gen}"
        COMMAND "${CMAKE_COMMAND}" -DINPUT=${_co} -DOUTPUT=${_gen}
                -DSYMBOL=${_kn} -DSRCNAME=${_kn}.hip -DNAMESPACE=rocm_kernels
                -P "${_embed}"
        DEPENDS "${_co}" "${_embed}"
        COMMENT "embed ${_kn}.co"
        VERBATIM)
    else()
      message(STATUS "clpeak ROCm: ${_kn} has no buildable arch for this toolkit; emitting stub")
      add_custom_command(
        OUTPUT  "${_gen}"
        COMMAND "${CMAKE_COMMAND}" -DINPUT=NONE -DOUTPUT=${_gen}
                -DSYMBOL=${_kn} -DSRCNAME=${_kn}.hip -DNAMESPACE=rocm_kernels
                -P "${_embed}"
        DEPENDS "${_embed}"
        COMMENT "stub ${_kn} (no supported arch)"
        VERBATIM)
    endif()

    list(APPEND _gen_srcs "${_gen}")
    string(TOUPPER "${_kn}" _ku)
    target_compile_definitions(${ER_TARGET} PRIVATE CLPEAK_ROCM_HAS_${_ku})
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_hip}")
  endforeach()

  target_sources(${ER_TARGET} PRIVATE ${_gen_srcs})
endfunction()
