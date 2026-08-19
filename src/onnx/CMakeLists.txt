if(NOT CLPEAK_ENABLE_ONNX)
    return()
endif()

# No find_package: the ONNX Runtime library is dlopen'd at run time (see
# onnx_runtime.cpp) and only its vendored C-API header is needed to build.
add_library(peak_onnx STATIC
    onnx_peak.cpp
    onnx_runtime.cpp
    onnx_session.cpp
    onnx_model.cpp
    gemm.cpp
    numeric_error.cpp
    block.cpp
)

target_link_libraries(peak_onnx PUBLIC peak_common)
target_include_directories(peak_onnx PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)
target_include_directories(peak_onnx PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/../../third_party/onnxruntime
)
target_compile_definitions(peak_onnx PRIVATE ENABLE_ONNX)
target_compile_definitions(peak_common PRIVATE ENABLE_ONNX)
