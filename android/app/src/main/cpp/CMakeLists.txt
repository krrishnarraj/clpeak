cmake_minimum_required(VERSION 3.4.1)

# Build stub
add_subdirectory(libopencl-stub)

set(ROOT_FOLDER "../../../../..")
include(${ROOT_FOLDER}/cmake/common.cmake)

set(SRC_FOLDER "${ROOT_FOLDER}/src")
add_library(clpeak SHARED
        ${SRC_FOLDER}/common.cpp
        ${SRC_FOLDER}/clpeak.cpp
        ${SRC_FOLDER}/global_bandwidth.cpp
        ${SRC_FOLDER}/compute_sp.cpp
        ${SRC_FOLDER}/compute_hp.cpp
        ${SRC_FOLDER}/compute_dp.cpp
        ${SRC_FOLDER}/compute_integer.cpp
        ${SRC_FOLDER}/transfer_bandwidth.cpp
        ${SRC_FOLDER}/kernel_latency.cpp
        ${SRC_FOLDER}/entry_android.cpp
        ${SRC_FOLDER}/options.cpp
        ${SRC_FOLDER}/logger_android.cpp
)

target_include_directories(clpeak PRIVATE
        libopencl-stub/include/
        ${ROOT_FOLDER}/include/
        ${SRC_FOLDER}/kernels/
)

target_compile_definitions(clpeak PRIVATE ANDROID_LOGGER USE_STUB_OPENCL)
target_compile_options(clpeak PRIVATE -fPIC -O2 -fexceptions)
target_link_libraries(clpeak log atomic dl OpenCL)
