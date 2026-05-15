if(NOT CLPEAK_ENABLE_VULKAN OR NOT Vulkan_FOUND)
    return()
endif()

add_library(peak_vulkan STATIC
    vk_peak.cpp
)
target_link_libraries(peak_vulkan PUBLIC peak_common ${Vulkan_LIBRARIES})
target_include_directories(peak_vulkan PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${Vulkan_INCLUDE_DIRS}
)
target_compile_definitions(peak_vulkan PRIVATE ENABLE_VULKAN)

# Compile GLSL shaders to SPIR-V and embed as C++ arrays
set(CLPEAK_VK_SHADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_sp_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_sp_v2.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_sp_v4.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_hp_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_hp_v2.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_hp_v4.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_dp_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_dp_v2.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_dp_v4.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_int32_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_int32_v2.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_int32_v4.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/global_bandwidth_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/global_bandwidth_v2.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/global_bandwidth_v4.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_int8_dp_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_int8_dp_v2.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_int8_dp_v4.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_mp_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_mp_v2.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_mp_v4.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_int4_packed_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_bf16_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_bf16_v2.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/compute_bf16_v4.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/coopmat_fp16.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/coopmat_bf16.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/coopmat_int8.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/coopmat_int8_k32.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/coopmat_fp8_e4m3.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/coopmat_fp8_e5m2.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/coopmat_fp32.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/local_bandwidth_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/local_bandwidth_v2.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/local_bandwidth_v4.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/image_bandwidth_v1.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/atomic_throughput_global.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/atomic_throughput_global_float.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/atomic_throughput_global_uint64.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/atomic_throughput_local.comp
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/kernel_latency.comp
)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileShaders.cmake)
compile_shaders(
    TARGET peak_vulkan
    SHADERS ${CLPEAK_VK_SHADERS}
)
