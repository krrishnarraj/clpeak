add_executable(clpeak
    main.cpp
    logger.cpp
)
target_link_libraries(clpeak PRIVATE peak_common)

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
if(TARGET peak_metal)
    target_link_libraries(clpeak PRIVATE peak_metal)
    target_compile_definitions(clpeak PRIVATE ENABLE_METAL)
endif()

target_include_directories(clpeak PRIVATE ${CMAKE_SOURCE_DIR}/include)
include(${CMAKE_SOURCE_DIR}/cmake/version.cmake)
clpeak_setup_version(clpeak)
