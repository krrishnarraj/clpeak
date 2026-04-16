cmake_minimum_required(VERSION 3.5)
project(clpeak)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake)
include(GNUInstallDirs)

option(ENABLE_VULKAN "Build with Vulkan compute backend (--vulkan flag)" OFF)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_package(OpenCL)
FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS ${OpenCL_INCLUDE_DIRS})
if(NOT OpenCL_FOUND OR NOT HPP_FOUND)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(BuildSdk)

  find_package(OpenCL REQUIRED)

  set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS} "${SDK_ROOT}/include")
  FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS ${OpenCL_INCLUDE_DIRS} REQUIRED)
endif()

if(NOT OpenCL_FOUND)
  message(FATAL_ERROR "Could not find OpenCL include/libs. Set OCL_ROOT to your OpenCL SDK")
else()
  message(STATUS "Selected OpenCL includes from ${OpenCL_INCLUDE_DIRS}")
  message(STATUS "Selected OpenCL lib ${OpenCL_LIBRARIES}")
endif()

# extra libs for static library
if(UNIX AND ${OpenCL_LIBRARIES} MATCHES "libOpenCL.a")
  set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES} pthread dl)
elseif(WIN32 AND ${OpenCL_LIBRARIES} MATCHES "OpenCL.lib")
  set(OpenCL_LIBRARIES ${OpenCL_LIBRARIES} cfgmgr32.lib runtimeobject.lib)
endif()

set(CMAKE_CXX_STANDARD "11")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CLPEAK_SOURCE_FILES
    src/common.cpp
    src/result_store.cpp
    src/clpeak.cpp
    src/options.cpp
    src/logger.cpp
    src/global_bandwidth.cpp
    src/local_bandwidth.cpp
    src/image_bandwidth.cpp
    src/atomic_throughput.cpp
    src/transfer_bandwidth.cpp
    src/kernel_latency.cpp
    src/entry.cpp
)

# ---- Optional Vulkan backend ----------------------------------------------
if(ENABLE_VULKAN)
  find_package(Vulkan REQUIRED)
  message(STATUS "Vulkan found: ${Vulkan_LIBRARIES}")

  list(APPEND CLPEAK_SOURCE_FILES
    src/vk_peak.cpp
    src/entry_vk.cpp
  )
endif()

add_executable(clpeak ${CLPEAK_SOURCE_FILES})
clpeak_setup_version(clpeak)
target_link_libraries(clpeak ${OpenCL_LIBRARIES})
target_include_directories(clpeak PRIVATE ${OpenCL_INCLUDE_DIRS} "include" "src/kernels")
target_compile_definitions(clpeak PRIVATE CL_TARGET_OPENCL_VERSION=120)

if(ENABLE_VULKAN)
  target_link_libraries(clpeak ${Vulkan_LIBRARIES})
  target_include_directories(clpeak PRIVATE ${Vulkan_INCLUDE_DIRS})
  target_compile_definitions(clpeak PRIVATE ENABLE_VULKAN)

  # Compile GLSL shaders to SPIR-V and embed as C++ arrays
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileShaders.cmake)
  compile_shaders(
    TARGET clpeak
    SHADERS
      ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/compute_sp_v1.comp
      ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/global_bandwidth_v1.comp
  )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU"   OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(clpeak PRIVATE -fPIC -Wall -Wextra -Wno-unused-parameter)
endif()

install(TARGETS clpeak RUNTIME DESTINATION bin)
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/clpeak")

enable_testing()
add_test(clpeak_test_run clpeak)

set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${CLPEAK_VERSION_STR}")
set(CPACK_GENERATOR "ZIP")
include (CPack)
