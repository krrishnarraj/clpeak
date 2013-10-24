
cmake_minimum_required(VERSION 2.6)
project(clPeak)

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
            $ENV{INTELOCLSDKROOT}/lib
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            $ENV{OPENCL_ROOT}/lib
        PATH_SUFFIXES x86 Win32
        PATHS
            /usr/lib32
            /usr/lib
    )
endif()

if( (NOT OPENCL_INCLUDES) OR (NOT OPENCL_LIBS) )
    message( FATAL_ERROR "Could not find OpenCL include/libs. Sepcify env variable OPENCL_ROOT" )
endif()

include_directories(${OPENCL_INCLUDES} "src/include" "src/kernels")

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(" -std=gnu++11 ")
endif()

set(SOURCE_FILES
    src/common.cpp
    src/clPeak.cpp
    src/main.cpp
    )

add_executable(clPeak ${SOURCE_FILES})

target_link_libraries(clPeak ${OPENCL_LIBS})
