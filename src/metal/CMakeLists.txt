if(NOT APPLE OR NOT CLPEAK_ENABLE_METAL)
  return()
endif()

find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)
find_library(MPS_FRAMEWORK MetalPerformanceShaders)
find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
  find_library(IOKIT_FRAMEWORK IOKit)
endif()

if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK OR NOT MPS_FRAMEWORK OR NOT MPSGRAPH_FRAMEWORK)
  return()
endif()
if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS" AND NOT IOKIT_FRAMEWORK)
  return()
endif()

add_library(peak_metal STATIC
    mtl_peak.mm
    mtl_device.mm
    mtl_utils.mm
    compute_kernel.mm
    mtl_blas.mm
    compute_float.mm
    simdgroup.mm
    global_bandwidth.mm
    local_bandwidth.mm
    image_bandwidth.mm
    kernel_latency.mm
)
target_link_libraries(peak_metal PUBLIC
    peak_common
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${MPS_FRAMEWORK}
    ${MPSGRAPH_FRAMEWORK}
)
if(IOKIT_FRAMEWORK)
  target_link_libraries(peak_metal PUBLIC ${IOKIT_FRAMEWORK})
endif()
target_include_directories(peak_metal PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)
target_compile_definitions(peak_metal PRIVATE ENABLE_METAL)
target_compile_definitions(peak_common PRIVATE ENABLE_METAL)

# Compile .mm files with ObjC ARC -- saves us from sprinkling release
# calls on autoreleased Metal objects.
set_source_files_properties(mtl_peak.mm mtl_device.mm mtl_utils.mm compute_kernel.mm mtl_blas.mm compute_float.mm simdgroup.mm global_bandwidth.mm local_bandwidth.mm image_bandwidth.mm kernel_latency.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# Embed .metal kernel sources as C++ raw-string literals
set(CLPEAK_MTL_KERNELS
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/compute_sp.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/compute_hp.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/compute_mp.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/global_bandwidth.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/kernel_latency.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/simdgroup_matrix_fp16.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/simdgroup_matrix_bf16.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/local_bandwidth.metal
    ${CMAKE_CURRENT_SOURCE_DIR}/mtl_kernels/image_bandwidth.metal
)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalKernels.cmake)
embed_metal_kernels(
    TARGET peak_metal
    KERNELS ${CLPEAK_MTL_KERNELS}
)
