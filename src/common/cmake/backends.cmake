# clpeak_link_backends(<target>)
#
# Wires a binary that runs benchmarks -- the clpeak executable, the
# clpeak_ffi library behind the GUI -- to every backend in this build: it
# adds the backend registry (src/registry/backend_registry.cpp) to the
# target's sources, links each backend library whose target exists, and
# defines the matching ENABLE_* macro so the registry and the binary's own
# sources see the same set.  One function, called from both, so the CLI and
# the GUI can never be built with different backends.
#
# oneAPI: the SYCL device images packed into libpeak_oneapi.a are extracted
# at the final link, and the registry includes <sycl/sycl.hpp> through
# oneapi_peak.h, so both that source's compile and the target's link need
# the SYCL toolchain flags.
set(CLPEAK_BACKEND_REGISTRY_SOURCE
    "${CMAKE_CURRENT_LIST_DIR}/../../registry/backend_registry.cpp")

function(clpeak_link_backends target)
    target_sources(${target} PRIVATE "${CLPEAK_BACKEND_REGISTRY_SOURCE}")

    foreach(_backend OPENCL VULKAN CUDA ROCM METAL ONEAPI CPU ONNX COREML LITERT)
        string(TOLOWER "${_backend}" _lower)
        if(TARGET peak_${_lower})
            target_link_libraries(${target} PRIVATE peak_${_lower})
            target_compile_definitions(${target} PRIVATE ENABLE_${_backend})
        endif()
    endforeach()

    if(TARGET peak_oneapi)
        add_sycl_to_target(TARGET ${target} SOURCES "${CLPEAK_BACKEND_REGISTRY_SOURCE}")
    endif()
endfunction()
