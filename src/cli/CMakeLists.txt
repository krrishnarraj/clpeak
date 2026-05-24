add_executable(clpeak
    main.cpp
    logger_cli.cpp
)
# peak_common comes transitively through every backend target
# Place binary at build root (not under src/cli/)
set_target_properties(clpeak PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Link enabled backends (targets exist only if their CMakeLists was processed)
if(TARGET peak_opencl)
    target_link_libraries(clpeak PRIVATE peak_opencl)
    target_compile_definitions(clpeak PRIVATE ENABLE_OPENCL)
endif()
if(TARGET peak_vulkan)
    target_link_libraries(clpeak PRIVATE peak_vulkan)
    target_compile_definitions(clpeak PRIVATE ENABLE_VULKAN)
endif()
if(TARGET peak_cuda)
    target_link_libraries(clpeak PRIVATE peak_cuda)
    target_compile_definitions(clpeak PRIVATE ENABLE_CUDA)
endif()
if(TARGET peak_rocm)
    target_link_libraries(clpeak PRIVATE peak_rocm)
    target_compile_definitions(clpeak PRIVATE ENABLE_ROCM)
endif()
if(TARGET peak_metal)
    target_link_libraries(clpeak PRIVATE peak_metal)
    target_compile_definitions(clpeak PRIVATE ENABLE_METAL)
endif()

target_include_directories(clpeak PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)
include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake/version.cmake)
clpeak_setup_version(clpeak)
