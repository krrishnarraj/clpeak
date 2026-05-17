if(NOT CLPEAK_ENABLE_OPENCL)
    return()
endif()

# ---- Find OpenCL --------------------------------------------------------
# Allow callers (Android) to pre-define OpenCL_LIBRARIES / OpenCL_INCLUDE_DIRS.
if(NOT DEFINED CLPEAK_OPENCL_PREDEFINED)
find_package(OpenCL)
FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS ${OpenCL_INCLUDE_DIRS})
if(NOT OpenCL_FOUND OR NOT HPP_FOUND)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    include(BuildSdk)
    set(OpenCL_ROOT "${SDK_ROOT}")
    find_package(OpenCL REQUIRED)
    FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS "${SDK_ROOT}/include" ${OpenCL_INCLUDE_DIRS} REQUIRED)
endif()
# Ensure the directory containing opencl.hpp is on the include path (needed on
# systems like macOS where the framework lacks the C++ bindings header).
list(APPEND OpenCL_INCLUDE_DIRS "${HPP_FOUND}")
list(REMOVE_DUPLICATES OpenCL_INCLUDE_DIRS)
message(STATUS "Selected OpenCL ${OpenCL_VERSION_STRING} includes from ${OpenCL_INCLUDE_DIRS}")
message(STATUS "Selected OpenCL lib ${OpenCL_LIBRARIES}")

# Does the selected OpenCL provide the 3.0 symbol set? (CL_VERSION_3_0 is only
# defined by the Khronos headers when CL_TARGET_OPENCL_VERSION >= 300, so set
# that for the probe.)
include(CheckSymbolExists)
set(_clpeak_saved_required_defs "${CMAKE_REQUIRED_DEFINITIONS}")
set(_clpeak_saved_required_incs "${CMAKE_REQUIRED_INCLUDES}")
list(APPEND CMAKE_REQUIRED_DEFINITIONS "-DCL_TARGET_OPENCL_VERSION=300")
set(CMAKE_REQUIRED_INCLUDES ${OpenCL_INCLUDE_DIRS})
check_symbol_exists(CL_VERSION_3_0 "CL/cl.h" CLPEAK_HAS_OPENCL_30)
set(CMAKE_REQUIRED_DEFINITIONS "${_clpeak_saved_required_defs}")
set(CMAKE_REQUIRED_INCLUDES    "${_clpeak_saved_required_incs}")

if(CLPEAK_HAS_OPENCL_30)
    message(STATUS "OpenCL 3.0 headers detected -- targeting 3.0 host API")
    set(CLPEAK_OPENCL_TARGET_VERSION 300)
else()
    message(STATUS "OpenCL 1.2 baseline -- targeting 1.2 host API")
    set(CLPEAK_OPENCL_TARGET_VERSION 120)
endif()

# extra libs for static library
if(UNIX AND ${OpenCL_LIBRARIES} MATCHES "libOpenCL.a")
    set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES} pthread dl)
elseif(WIN32 AND ${OpenCL_LIBRARIES} MATCHES "OpenCL.lib")
    set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES} cfgmgr32.lib runtimeobject.lib)
endif()
endif()  # CLPEAK_OPENCL_PREDEFINED

# Fallback when caller provides OpenCL vars (e.g. Android with stub)
if(NOT DEFINED CLPEAK_OPENCL_TARGET_VERSION)
    set(CLPEAK_OPENCL_TARGET_VERSION 300)
endif()

# ---- peak_opencl static library -----------------------------------------
add_library(peak_opencl STATIC
    cl_peak.cpp
    cl_common.cpp
    cl_utils.cpp
    global_bandwidth.cpp
    local_bandwidth.cpp
    image_bandwidth.cpp
    atomic_throughput.cpp
    transfer_bandwidth.cpp
    kernel_latency.cpp
)
target_link_libraries(peak_opencl PUBLIC peak_common ${OpenCL_LIBRARIES})
target_include_directories(peak_opencl PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
    ${CMAKE_CURRENT_LIST_DIR}
    ${OpenCL_INCLUDE_DIRS}
)
target_compile_definitions(peak_opencl PRIVATE
    CL_TARGET_OPENCL_VERSION=${CLPEAK_OPENCL_TARGET_VERSION}
)
target_compile_definitions(peak_common PRIVATE ENABLE_OPENCL)
