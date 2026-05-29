if(NOT CLPEAK_ENABLE_ROCM OR NOT hip_FOUND OR NOT CLPEAK_HIPRTC_FOUND)
    return()
endif()

add_library(peak_rocm STATIC
    rocm_peak.cpp
    rocm_device.cpp
    compute_kernel.cpp
    compute_float.cpp
    compute_int.cpp
    rocwmma.cpp
    mfma.cpp
    rocblas.cpp
    hipblaslt_gemm.cpp
    global_bandwidth.cpp
    local_bandwidth.cpp
    image_bandwidth.cpp
    transfer_bandwidth.cpp
    atomic_throughput.cpp
    kernel_latency.cpp
)

target_link_libraries(peak_rocm PUBLIC
    peak_common
    hip::host
    ${CLPEAK_HIPRTC_LINK}
)
if(rocblas_FOUND AND TARGET roc::rocblas)
    target_link_libraries(peak_rocm PUBLIC roc::rocblas)
    target_compile_definitions(peak_rocm PRIVATE CLPEAK_ROCM_HAS_ROCBLAS)
endif()
if(hipblaslt_FOUND AND TARGET roc::hipblaslt)
    target_link_libraries(peak_rocm PUBLIC roc::hipblaslt)
    target_compile_definitions(peak_rocm PRIVATE CLPEAK_ROCM_HAS_HIPBLASLT)
endif()
target_include_directories(peak_rocm PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)
target_compile_features(peak_rocm PRIVATE cxx_std_14)

set(_clpeak_rocm_inc "/opt/rocm/include")
if(TARGET hip::host)
    get_target_property(_clpeak_hip_incs hip::host INTERFACE_INCLUDE_DIRECTORIES)
    if(_clpeak_hip_incs)
        foreach(_clpeak_hip_inc IN LISTS _clpeak_hip_incs)
            if(NOT _clpeak_hip_inc MATCHES "^\\$<")
                set(_clpeak_rocm_inc "${_clpeak_hip_inc}")
                break()
            endif()
        endforeach()
    endif()
endif()

target_compile_definitions(peak_rocm PRIVATE
    ENABLE_ROCM
    CLPEAK_ROCM_INCLUDE_DIR="${_clpeak_rocm_inc}"
)
if(CLPEAK_ROCWMMA_FOUND)
    target_compile_definitions(peak_rocm PRIVATE
        CLPEAK_ROCM_HAS_ROCWMMA
        CLPEAK_ROCWMMA_INCLUDE_DIR="${CLPEAK_ROCWMMA_INCLUDE_DIR}"
    )
endif()
target_compile_definitions(peak_common PRIVATE ENABLE_ROCM)

set(CLPEAK_ROCM_KERNELS
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/compute_sp.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/compute_hp.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/compute_dp.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/compute_mp.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/compute_bf16.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/compute_int32.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/compute_int4_packed.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/compute_int8_dp.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/rocwmma_fp16.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/rocwmma_int8.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/mfma_fp16.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/mfma_bf16.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/mfma_int8.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/mfma_fp8.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/global_bandwidth.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/local_bandwidth.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/image_bandwidth.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/atomic_throughput.hip
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_kernels/kernel_latency.hip
)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedRocmKernels.cmake)
embed_rocm_kernels(
    TARGET peak_rocm
    KERNELS ${CLPEAK_ROCM_KERNELS}
)
