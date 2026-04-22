cmake_minimum_required(VERSION 3.20)
project(clpeak)

set(CLPEAK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
include(${CLPEAK_ROOT}/cmake/version.cmake)
include(${CLPEAK_ROOT}/cmake/sources.cmake)
include(GNUInstallDirs)

if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# FindOpenCL.cmake checks for the CL_VERSION_X_Y symbol to determine the header
# version. In the modern Khronos headers CL_VERSION_3_0 is only defined when
# CL_TARGET_OPENCL_VERSION >= 300, so we pre-set the required definition for
# the version probe before calling find_package.
set(_clpeak_saved_required_defs "${CMAKE_REQUIRED_DEFINITIONS}")
list(APPEND CMAKE_REQUIRED_DEFINITIONS "-DCL_TARGET_OPENCL_VERSION=300")

find_package(OpenCL 3.0)
FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS ${OpenCL_INCLUDE_DIRS})
if(NOT OpenCL_FOUND OR NOT HPP_FOUND)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(BuildSdk)

  # Re-search after BuildSdk installed a 3.0-capable SDK to ${SDK_ROOT}.
  set(OpenCL_ROOT "${SDK_ROOT}")
  if(APPLE)
    # macOS system framework is OpenCL 1.2; the first find_package cached those
    # results. Clear them and deprioritize frameworks so the 3.0 SDK wins.
    unset(OpenCL_FOUND CACHE)
    unset(OpenCL_INCLUDE_DIR CACHE)
    unset(OpenCL_LIBRARY CACHE)
    foreach(_v 3_0 2_2 2_1 2_0 1_2 1_1 1_0)
      unset(OPENCL_VERSION_${_v} CACHE)
    endforeach()
    set(_clpeak_saved_find_framework ${CMAKE_FIND_FRAMEWORK})
    set(CMAKE_FIND_FRAMEWORK LAST)
  endif()
  find_package(OpenCL 3.0 REQUIRED)
  if(APPLE)
    set(CMAKE_FIND_FRAMEWORK ${_clpeak_saved_find_framework})
  endif()

  set(OpenCL_INCLUDE_DIRS ${OpenCL_INCLUDE_DIRS} "${SDK_ROOT}/include")
  FIND_PATH(HPP_FOUND CL/opencl.hpp PATHS ${OpenCL_INCLUDE_DIRS} REQUIRED)
endif()

set(CMAKE_REQUIRED_DEFINITIONS "${_clpeak_saved_required_defs}")

if(NOT OpenCL_FOUND)
  message(FATAL_ERROR "Could not find OpenCL 3.0+ include/libs. Set OCL_ROOT to an OpenCL 3.0 SDK")
else()
  message(STATUS "Selected OpenCL ${OpenCL_VERSION_STRING} includes from ${OpenCL_INCLUDE_DIRS}")
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
    ${CLPEAK_CORE_SOURCES}
    src/logger.cpp
    src/entry.cpp
)

# ---- Vulkan backend (enabled by default when Vulkan is available) ---------
find_package(Vulkan)
if(Vulkan_FOUND)
  message(STATUS "Vulkan found: ${Vulkan_LIBRARIES} -- Vulkan backend enabled")
  list(APPEND CLPEAK_SOURCE_FILES
    src/vk_peak.cpp
    src/entry_vk.cpp
  )
else()
  message(WARNING "Vulkan SDK/loader not found -- Vulkan tests will be disabled. Install the Vulkan SDK to enable the Vulkan backend.")
endif()

add_executable(clpeak ${CLPEAK_SOURCE_FILES})
clpeak_setup_version(clpeak)
target_link_libraries(clpeak ${OpenCL_LIBRARIES})
target_include_directories(clpeak PRIVATE ${OpenCL_INCLUDE_DIRS} ${CLPEAK_CORE_INCLUDE_DIRS})
target_compile_definitions(clpeak PRIVATE CL_TARGET_OPENCL_VERSION=300)

if(Vulkan_FOUND)
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
      ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/compute_int8_dp_v1.comp
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
