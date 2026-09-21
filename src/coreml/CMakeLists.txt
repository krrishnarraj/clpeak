if(NOT APPLE OR NOT CLPEAK_ENABLE_COREML)
  return()
endif()

find_library(COREML_FRAMEWORK CoreML)
find_library(COREML_FOUNDATION_FRAMEWORK Foundation)
find_library(COREML_METAL_FRAMEWORK Metal)
find_library(COREML_ACCELERATE_FRAMEWORK Accelerate)
if(NOT COREML_FRAMEWORK OR NOT COREML_FOUNDATION_FRAMEWORK OR NOT COREML_METAL_FRAMEWORK
   OR NOT COREML_ACCELERATE_FRAMEWORK)
  return()
endif()

# The Objective-C++ files are the session (compile / load / compute plan /
# predict) and the device enumeration; everything else is plain C++ against
# the C++ surface in coreml_session.h.
set(CLPEAK_COREML_OBJC
    coreml_peak.mm
    coreml_session.mm
)
add_library(peak_coreml STATIC
    ${CLPEAK_COREML_OBJC}
    coreml_model.cpp
    gemm.cpp
    numeric_error.cpp
    conv.cpp
    block.cpp
    activation.cpp
    tensor_bandwidth.cpp
    transfer.cpp
    dispatch_latency.cpp
)
target_link_libraries(peak_coreml PUBLIC
    peak_common
    ${COREML_FRAMEWORK}
    ${COREML_FOUNDATION_FRAMEWORK}
    ${COREML_METAL_FRAMEWORK}
    ${COREML_ACCELERATE_FRAMEWORK}
)
target_include_directories(peak_coreml PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)
target_compile_definitions(peak_coreml PRIVATE ENABLE_COREML)
target_compile_definitions(peak_common PRIVATE ENABLE_COREML)

# ARC for the Objective-C++ files, as in src/metal.
set_source_files_properties(${CLPEAK_COREML_OBJC} PROPERTIES COMPILE_FLAGS "-fobjc-arc")
