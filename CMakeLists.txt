
cmake_minimum_required(VERSION 3.1)
project(clpeak)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# FindOpenCL.cmake doesn't use OPENCL_ROOT so force it to copy OPENCL_ROOT to
# AMDAPPSDKROOT which it does use
if(OPENCL_ROOT)
  set(ENV{AMDAPPSDKROOT} ${OPENCL_ROOT})
endif()

find_package(OpenCL)

if(NOT OpenCL_FOUND)
  message( FATAL_ERROR "Could not find OpenCL include/libs. Set OPENCL_ROOT to your OpenCL SDK. Download AMD APP SDK "
      "http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/ for x86/x64 "
      "or pocl http://pocl.sourceforge.net/ for ARM systems" )
else()
  message(STATUS "Selected OpenCL includes from ${OpenCL_INCLUDE_DIRS}")
  message(STATUS "Selected OpenCL lib ${OpenCL_LIBRARIES}")
endif()

set(CMAKE_CXX_STANDARD "11")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_FILES
    src/common.cpp
    src/clpeak.cpp
    src/options.cpp
    src/logger.cpp
    src/global_bandwidth.cpp
    src/compute_sp.cpp
    src/compute_hp.cpp
    src/compute_dp.cpp
    src/compute_integer.cpp
    src/transfer_bandwidth.cpp
    src/kernel_latency.cpp
    src/entry.cpp
)

add_executable(clpeak ${SOURCE_FILES})

target_link_libraries(clpeak ${OpenCL_LIBRARIES})

# override cl.hp from deps. Its buggy or not present in some systems
target_include_directories(clpeak PRIVATE "deps/OpenCL-CLHPP/include/")
target_include_directories(clpeak PRIVATE ${OpenCL_INCLUDE_DIRS} "include" "src/kernels")


if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU"   OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(clpeak PRIVATE -fPIC -Wall -Wextra -Wno-deprecated-declarations -Wno-unused-parameter -Wno-ignored-attributes)
endif()
