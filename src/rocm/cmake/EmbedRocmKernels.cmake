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
# (probed by trial compile) -- and, for ROCWMMA, with what the installed
# rocWMMA headers accept, since they static_assert on any arch their version
# does not know.  A kernel with no buildable arch gets an empty stub Blob so
# the symbol still links; at run time a Blob without the device's slice is
# reported from its bundle header (RocmDevice::getKernel).

set(_CLPEAK_EMBED_ROCM_DIR "${CMAKE_CURRENT_LIST_DIR}")

# -parallel-jobs runs one hipcc call's per-arch device compiles side by side;
# without it they run one after another, and with ~30 targets in a group that
# is most of the ROCm build -- all of it on a generator that runs custom
# commands serially.  Each of those compiles holds ~150 MB, and make -j or
# ninja runs several kernels' calls at once on top of this, so it is capped;
# under Ninja the calls also share a pool sized so calls x jobs ~= cores.
cmake_host_system_information(RESULT _clpeak_cores QUERY NUMBER_OF_LOGICAL_CORES)
if(NOT _clpeak_cores OR _clpeak_cores LESS 1)
  set(_clpeak_cores 1)
endif()
set(_clpeak_jobs ${_clpeak_cores})
if(_clpeak_jobs GREATER 8)
  set(_clpeak_jobs 8)
endif()
set(_CLPEAK_HIPCC_JOBS "-parallel-jobs=${_clpeak_jobs}")
math(EXPR _clpeak_pool "${_clpeak_cores} / ${_clpeak_jobs}")
get_property(_clpeak_pools GLOBAL PROPERTY JOB_POOLS)
if(NOT "${_clpeak_pools}" MATCHES "clpeak_hipcc=")
  set_property(GLOBAL APPEND PROPERTY JOB_POOLS clpeak_hipcc=${_clpeak_pool})
endif()

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

# Which of `archs` hipcc can build `source` for.  All of them go into one call,
# so a toolkit that takes every arch answers in a single compile; a call that
# fails is split in half and each half retried, so an arch the toolkit lacks
# costs a few quick failures instead of a compile per candidate.
function(_clpeak_rocm_probe_split source flags archs out)
  set(_args "")
  foreach(_a ${archs})
    list(APPEND _args "--offload-arch=${_a}")
  endforeach()

  # Same env scrubbing as the real compiles: without it, configuring from a
  # Visual Studio developer prompt (where INCLUDE *is* set) would fail every
  # probe and silently fall through to stub kernels for the whole backend.
  # -Wfatal-errors: an arch rocWMMA rejects fails in its config header, and
  # without it clang goes on to parse the whole library before giving up.
  _clpeak_hipcc_env_wrap(_wrap)
  execute_process(
    COMMAND ${_wrap} "${CLPEAK_HIPCC}" --genco ${_args} ${flags} ${_CLPEAK_HIPCC_JOBS}
            -Wfatal-errors -o "${CMAKE_CURRENT_BINARY_DIR}/_clpeak_archprobe.co" "${source}"
    RESULT_VARIABLE _r OUTPUT_QUIET ERROR_QUIET)

  list(LENGTH archs _n)
  if(_r EQUAL 0)
    set(${out} "${archs}" PARENT_SCOPE)
  elseif(_n LESS 2)
    set(${out} "" PARENT_SCOPE)
  else()
    math(EXPR _half "${_n} / 2")
    list(SUBLIST archs 0 ${_half} _lo)
    list(SUBLIST archs ${_half} -1 _hi)
    _clpeak_rocm_probe_split("${source}" "${flags}" "${_lo}" _lo_ok)
    _clpeak_rocm_probe_split("${source}" "${flags}" "${_hi}" _hi_ok)
    set(_ok ${_lo_ok} ${_hi_ok})
    set(${out} "${_ok}" PARENT_SCOPE)
  endif()
endfunction()

# Which of `archs` a kernel of `kind` can be built for: "hip" is a bare kernel
# (can the installed hipcc target the arch at all), "rocwmma" one that includes
# the rocWMMA headers.  Probing rocWMMA parses the whole library once per arch
# -- about 20 s -- so verdicts are kept in the cache, keyed by the toolkit they
# were taken against (hipcc's --version and, for rocWMMA, its header), and only
# an arch with no verdict yet is compiled; a different toolkit starts over.
function(_clpeak_rocm_supported_archs kind archs out)
  get_property(_version GLOBAL PROPERTY _CLPEAK_HIPCC_VERSION)
  if(NOT _version)
    execute_process(COMMAND "${CLPEAK_HIPCC}" --version
                    OUTPUT_VARIABLE _version ERROR_QUIET)
    set_property(GLOBAL PROPERTY _CLPEAK_HIPCC_VERSION "${_version}")
  endif()
  set(_key "${CLPEAK_HIPCC}|${_version}")
  if(kind STREQUAL "rocwmma")
    file(TIMESTAMP "${CLPEAK_ROCWMMA_INCLUDE_DIR}/rocwmma/rocwmma.hpp" _stamp)
    string(APPEND _key "|${CLPEAK_ROCWMMA_INCLUDE_DIR}|${_stamp}")
  endif()
  string(MD5 _key "${_key}")
  if(NOT "${_CLPEAK_ROCM_PROBE_${kind}_KEY}" STREQUAL "${_key}")
    set(_CLPEAK_ROCM_PROBE_${kind}_KEY "${_key}" CACHE INTERNAL "")
    set(_CLPEAK_ROCM_PROBE_${kind}_OK "" CACHE INTERNAL "")
    set(_CLPEAK_ROCM_PROBE_${kind}_NO "" CACHE INTERNAL "")
  endif()
  set(_ok "${_CLPEAK_ROCM_PROBE_${kind}_OK}")
  set(_no "${_CLPEAK_ROCM_PROBE_${kind}_NO}")

  set(_unknown "")
  foreach(_a ${archs})
    if(NOT _a IN_LIST _ok AND NOT _a IN_LIST _no)
      list(APPEND _unknown "${_a}")
    endif()
  endforeach()

  if(_unknown)
    set(_probe "${CMAKE_CURRENT_BINARY_DIR}/_clpeak_archprobe_${kind}.hip")
    set(_flags "")
    if(kind STREQUAL "rocwmma")
      file(WRITE "${_probe}" "#include <rocwmma/rocwmma.hpp>\nextern \"C\" __global__ void p(){}\n")
      set(_flags -std=c++17 "-I${CLPEAK_ROCWMMA_INCLUDE_DIR}")
    else()
      file(WRITE "${_probe}" "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void p(){}\n")
    endif()
    _clpeak_rocm_probe_split("${_probe}" "${_flags}" "${_unknown}" _built)
    foreach(_a ${_unknown})
      if(_a IN_LIST _built)
        list(APPEND _ok "${_a}")
      else()
        list(APPEND _no "${_a}")
      endif()
    endforeach()
    set(_CLPEAK_ROCM_PROBE_${kind}_OK "${_ok}" CACHE INTERNAL "")
    set(_CLPEAK_ROCM_PROBE_${kind}_NO "${_no}" CACHE INTERNAL "")
  endif()

  # Said once per configure: the first call of a kind is the widest group.
  get_property(_said GLOBAL PROPERTY _CLPEAK_ROCM_PROBE_${kind}_SAID)
  if(NOT _said)
    set_property(GLOBAL PROPERTY _CLPEAK_ROCM_PROBE_${kind}_SAID TRUE)
    if(kind STREQUAL "rocwmma")
      message(STATUS "clpeak ROCm: rocWMMA supports: ${_ok}")
    elseif(NOT _ok)
      message(WARNING "clpeak ROCm: hipcc (${CLPEAK_HIPCC}) could not build a "
                      "trivial kernel for any gfx target, so every ROCm kernel in "
                      "this build is an empty stub")
    else()
      message(STATUS "clpeak ROCm: hipcc can target: ${_ok}")
      if(_no)
        message(STATUS "clpeak ROCm: hipcc cannot target: ${_no} -- no kernels for them in this build")
      endif()
    endif()
  endif()

  set(_result "")
  foreach(_a ${archs})
    if(_a IN_LIST _ok)
      list(APPEND _result "${_a}")
    endif()
  endforeach()
  set(${out} "${_result}" PARENT_SCOPE)
endfunction()

function(embed_rocm_kernels)
  cmake_parse_arguments(ER "CXX17;ROCWMMA" "TARGET" "ARCHS;KERNELS" ${ARGN})

  # Keep the requested arches the installed toolkit can build this group for.
  _clpeak_rocm_supported_archs(hip "${ER_ARCHS}" _final)
  if(ER_ROCWMMA)
    _clpeak_rocm_supported_archs(rocwmma "${_final}" _final)
  endif()

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
        COMMAND ${_wrap} "${CLPEAK_HIPCC}" --genco ${_flags} ${_CLPEAK_HIPCC_JOBS}
                -O3 -o "${_co}" "${_hip}"
        DEPENDS "${_hip}"
        JOB_POOL clpeak_hipcc
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
  endforeach()

  target_sources(${ER_TARGET} PRIVATE ${_gen_srcs})
endfunction()
