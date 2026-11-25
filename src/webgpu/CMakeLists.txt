if(NOT CLPEAK_ENABLE_WEBGPU)
    return()
endif()

# No find_package: the WebGPU C API (webgpu.h) library is dlopen'd at run time
# (see webgpu_runtime.cpp) and only the vendored standard C header in
# third_party/webgpu/ is needed to build.
add_library(peak_webgpu STATIC
    webgpu_runtime.cpp
    webgpu_device.cpp
    webgpu_peak.cpp
    compute_kernel.cpp
    compute_float.cpp
    compute_int.cpp
    global_bandwidth.cpp
    local_bandwidth.cpp
    image_bandwidth.cpp
    transfer_bandwidth.cpp
    kernel_latency.cpp
)

target_link_libraries(peak_webgpu PUBLIC peak_common)
target_include_directories(peak_webgpu PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)
target_include_directories(peak_webgpu PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../third_party/webgpu
)
target_compile_definitions(peak_webgpu PRIVATE ENABLE_WEBGPU)
target_compile_definitions(peak_common PRIVATE ENABLE_WEBGPU)
