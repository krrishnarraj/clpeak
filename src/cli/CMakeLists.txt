add_executable(xpupeak
    main.cpp
)
# peak_common comes transitively through every backend target
# Place binary at build root (not under src/cli/)
set_target_properties(xpupeak PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# Every backend in this build, through the shared registry (also what the
# GUI bridge links), so both binaries run the same backends in the same order.
include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake/backends.cmake)
xpupeak_link_backends(xpupeak)

target_include_directories(xpupeak PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../../include)
include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake/version.cmake)
xpupeak_setup_version(xpupeak)
# The configuration goes into every run document's `build` block: a debug
# build's host-side work (model generation, reference arithmetic) runs
# unoptimised, and a report from one should say so.
target_compile_definitions(xpupeak PRIVATE XPUPEAK_BUILD_CONFIG="$<CONFIG>")
