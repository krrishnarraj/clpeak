# EmbedCudaKernels.cmake
#
# Ahead-of-time CUDA kernel compilation.  Each .cu is compiled by nvcc to a
# multi-arch *fatbin* (SASS for the requested arches + PTX at the top arch for
# forward compatibility) and embedded into the binary as a byte array.  At run
# time the CUDA *driver* selects the matching cubin or JITs the embedded PTX --
# so the shipped binary needs only the NVIDIA driver, no NVRTC and no toolkit
# headers.
#
# Usage (one call per arch group, since kernels differ in their valid arch
# range -- e.g. fp8 needs sm_89+, int4 is sm_75..sm_89, fp4 is sm_120a):
#
#   embed_cuda_kernels(
#     TARGET       peak_cuda
#     MASTER_ARCHS "70;75;80;86;89;90;100;120"   # ascending
#     MIN_ARCH     80                            # keep MASTER >= this
#     [MAX_ARCH    89]                           # and <= this (optional)
#     [EXTRA_ARCHS "120a"]                       # accelerated variants to add
#     KERNELS      compute_bf16 wmma_bf16 ...    # bare stems under cuda_kernels/
#   )
#
# The requested arches are intersected with what the installed nvcc can target
# (`nvcc --list-gpu-arch`); a kernel left with no buildable arch gets an empty
# stub Blob so the symbol still links (it would be capability-gated off anyway).

set(_CLPEAK_EMBED_CUDA_DIR "${CMAKE_CURRENT_LIST_DIR}")

# Cache the arch list the toolkit supports (numbers only, e.g. 52;...;90;120).
function(_clpeak_cuda_supported_archs out)
  execute_process(
    COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" --list-gpu-arch
    OUTPUT_VARIABLE _o RESULT_VARIABLE _r
    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(_list "")
  if(_r EQUAL 0)
    string(REGEX MATCHALL "compute_([0-9]+)" _m "${_o}")
    foreach(_x ${_m})
      string(REGEX REPLACE "compute_" "" _n "${_x}")
      list(APPEND _list "${_n}")
    endforeach()
  endif()
  set(${out} "${_list}" PARENT_SCOPE)
endfunction()

function(embed_cuda_kernels)
  cmake_parse_arguments(EC "" "TARGET;MIN_ARCH;MAX_ARCH"
                          "MASTER_ARCHS;EXTRA_ARCHS;KERNELS" ${ARGN})

  _clpeak_cuda_supported_archs(_supported)

  # Build this group's arch list: MASTER filtered by [MIN_ARCH, MAX_ARCH],
  # then EXTRA_ARCHS appended, all intersected with toolkit support.
  set(_archs "")
  foreach(_a ${EC_MASTER_ARCHS})
    if(DEFINED EC_MIN_ARCH AND _a LESS EC_MIN_ARCH)
      continue()
    endif()
    if(DEFINED EC_MAX_ARCH AND _a GREATER EC_MAX_ARCH)
      continue()
    endif()
    list(APPEND _archs "${_a}")
  endforeach()
  foreach(_a ${EC_EXTRA_ARCHS})
    list(APPEND _archs "${_a}")
  endforeach()

  # Intersect with toolkit-supported arches (strip any trailing 'a'/'f' suffix
  # before the membership test).
  set(_final "")
  foreach(_a ${_archs})
    string(REGEX REPLACE "[a-z]+$" "" _base "${_a}")
    if(_supported)
      list(FIND _supported "${_base}" _idx)
      if(_idx GREATER -1)
        list(APPEND _final "${_a}")
      endif()
    else()
      list(APPEND _final "${_a}")   # no toolkit info -> trust the request
    endif()
  endforeach()

  # gencode flags: SASS per arch + one PTX entry at the highest arch.
  set(_gencode "")
  foreach(_a ${_final})
    list(APPEND _gencode "-gencode" "arch=compute_${_a},code=sm_${_a}")
  endforeach()
  if(_final)
    list(LENGTH _final _nfinal)
    math(EXPR _ilast "${_nfinal} - 1")
    list(GET _final ${_ilast} _top)
    list(APPEND _gencode "-gencode" "arch=compute_${_top},code=compute_${_top}")
  endif()

  set(_fatdir "${CMAKE_CURRENT_BINARY_DIR}/cuda_fatbins")
  set(_gendir "${CMAKE_CURRENT_BINARY_DIR}/cuda_kernels_gen")
  file(MAKE_DIRECTORY "${_fatdir}" "${_gendir}")
  set(_embed "${_CLPEAK_EMBED_CUDA_DIR}/EmbedBin.cmake")

  set(_gen_srcs "")
  foreach(_kn ${EC_KERNELS})
    set(_cu  "${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels/${_kn}.cu")
    if(NOT EXISTS "${_cu}")
      message(FATAL_ERROR "embed_cuda_kernels: source not found: ${_cu}")
    endif()
    set(_gen "${_gendir}/${_kn}.cpp")

    if(_final)
      set(_fat "${_fatdir}/${_kn}.fatbin")
      add_custom_command(
        OUTPUT  "${_fat}"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}" -fatbin ${_gencode} -o "${_fat}" "${_cu}"
        DEPENDS "${_cu}"
        COMMENT "nvcc -fatbin ${_kn}.cu"
        VERBATIM)
      add_custom_command(
        OUTPUT  "${_gen}"
        COMMAND "${CMAKE_COMMAND}" -DINPUT=${_fat} -DOUTPUT=${_gen}
                -DSYMBOL=${_kn} -DSRCNAME=${_kn}.cu -DNAMESPACE=cuda_kernels
                -P "${_embed}"
        DEPENDS "${_fat}" "${_embed}"
        COMMENT "embed ${_kn}.fatbin"
        VERBATIM)
    else()
      message(STATUS "clpeak CUDA: ${_kn} has no buildable arch for this toolkit; emitting stub")
      add_custom_command(
        OUTPUT  "${_gen}"
        COMMAND "${CMAKE_COMMAND}" -DINPUT=NONE -DOUTPUT=${_gen}
                -DSYMBOL=${_kn} -DSRCNAME=${_kn}.cu -DNAMESPACE=cuda_kernels
                -P "${_embed}"
        DEPENDS "${_embed}"
        COMMENT "stub ${_kn} (no supported arch)"
        VERBATIM)
    endif()

    list(APPEND _gen_srcs "${_gen}")
    string(TOUPPER "${_kn}" _ku)
    target_compile_definitions(${EC_TARGET} PRIVATE CLPEAK_CUDA_HAS_${_ku})
    set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_cu}")
  endforeach()

  target_sources(${EC_TARGET} PRIVATE ${_gen_srcs})
endfunction()
