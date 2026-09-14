add_executable(clpeak
    main.cpp
)
# peak_common comes transitively through every backend target
# Place binary at build root (not under src/cli/)
set_target_properties(clpeak PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Every backend in this build, through the shared registry (also what the
# GUI bridge links), so both binaries run the same backends in the same order.
include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake/backends.cmake)
clpeak_link_backends(clpeak)

target_include_directories(clpeak PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)
include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake/version.cmake)
clpeak_setup_version(clpeak)
