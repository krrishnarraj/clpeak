if(NOT APPLE OR NOT CLPEAK_ENABLE_METAL)
    return()
endif()

find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(IOKIT_FRAMEWORK IOKit)
find_library(MPS_FRAMEWORK MetalPerformanceShaders)
find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)

if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK OR NOT MPS_FRAMEWORK OR NOT MPSGRAPH_FRAMEWORK)
    return()
endif()

add_library(peak_metal STATIC
    mtl_peak.mm
    mtl_blas.mm
)
target_link_libraries(peak_metal PUBLIC
    peak_common
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${IOKIT_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
)
target_include_directories(peak_metal PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_compile_definitions(peak_metal PRIVATE ENABLE_METAL)

# Compile .mm files with ObjC ARC -- saves us from sprinkling release
# calls on autoreleased Metal objects.
set_source_files_properties(mtl_peak.mm mtl_blas.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# Embed .metal kernel sources as C++ raw-string literals
set(CLPEAK_MTL_KERNELS
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/compute_sp.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/compute_hp.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/compute_mp.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/compute_int8_dp.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/compute_int4_packed.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/global_bandwidth.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/kernel_latency.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/simdgroup_matrix_fp16.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/simdgroup_matrix_bf16.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/simdgroup_matrix_int8.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/local_bandwidth.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/image_bandwidth.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/atomic_throughput.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/atomic_throughput_float.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/atomic_throughput_ulong.metal
)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalKernels.cmake)
embed_metal_kernels(
    TARGET peak_metal
    KERNELS ${CLPEAK_MTL_KERNELS}
)
