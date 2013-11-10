
cmake_minimum_required(VERSION 2.6)
project(clpeak)

if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to Release")
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Determine machine bitness
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BITNESS 64)
else()
    set(BITNESS 32)
endif()

# Find OpenCL include directories
find_path( OPENCL_INCLUDES
	NAMES CL/cl.h OpenCL/cl.h
    HINTS
        $ENV{AMDAPPSDKROOT}/include
        $ENV{INTELOCLSDKROOT}/include
        $ENV{CUDA_PATH}/include
        $ENV{OPENCL_ROOT}/include
	PATHS
        /usr/include
        /usr/local/include
    )

# Find OpenCL libraries
if(BITNESS EQUAL 64)
    find_library( OPENCL_LIBS
        NAMES OpenCL
        HINTS
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{INTELOCLSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            $ENV{OPENCL_ROOT}/lib
        PATH_SUFFIXES x86_64 x64
        PATHS
            /usr/lib64
            /usr/lib
    )
elseif(BITNESS EQUAL 32)
    find_library( OPENCL_LIBS
        NAMES OpenCL
        HINTS
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{INTELOCLSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            $ENV{OPENCL_ROOT}/lib
        PATH_SUFFIXES x86 Win32
        PATHS
            /usr/lib32
            /usr/lib
    )
endif()

if( (NOT OPENCL_INCLUDES) OR (NOT OPENCL_LIBS) )
    message( FATAL_ERROR "Could not find OpenCL include/libs. Download AMD APP SDK "
            "http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/ for x86/x64 "
            "or pocl http://pocl.sourceforge.net/ for ARM systems" )
endif()

include_directories(${OPENCL_INCLUDES} "src/include" "src/kernels")


if(CMAKE_COMPILER_IS_GNUCXX)
    execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if(GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7)
        add_definitions("-std=gnu++11")
    elseif(GCC_VERSION VERSION_GREATER 4.3 OR GCC_VERSION VERSION_EQUAL 4.3)
        add_definitions("-std=gnu++0x")
    else()
        message(FATAL_ERROR "C++11 needed. Therefore a gcc compiler with a version higher than 4.3 is needed.")   
    endif()
endif()


set(SOURCE_FILES
    src/common.cpp
    src/clpeak.cpp
    src/bandwidth.cpp
    src/compute_sp.cpp
    src/compute_dp.cpp
    src/main.cpp
    )

add_executable(clpeak ${SOURCE_FILES})

target_link_libraries(clpeak ${OPENCL_LIBS})

