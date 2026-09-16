if(NOT CLPEAK_ENABLE_LITERT)
    return()
endif()

# No find_package: LiteRT is dlopen'd at run time (see litert_runtime.cpp)
# and only its vendored C headers are needed to build.
add_library(peak_litert STATIC
    litert_peak.cpp
    litert_runtime.cpp
    litert_session.cpp
    litert_model.cpp
    tflite_model.cpp
    gemm.cpp
    conv.cpp
    numeric_error.cpp
    block.cpp
    activation.cpp
    tensor_bandwidth.cpp
    transfer.cpp
    dispatch_latency.cpp
)

target_link_libraries(peak_litert PUBLIC peak_common)
target_include_directories(peak_litert PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)
target_include_directories(peak_litert PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../third_party/litert
)
target_compile_definitions(peak_litert PRIVATE ENABLE_LITERT)
target_compile_definitions(peak_common PRIVATE ENABLE_LITERT)
