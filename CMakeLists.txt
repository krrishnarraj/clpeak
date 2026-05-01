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

# ---- OpenCL: 1.2 is the baseline; 3.0-only tests are enabled when the
#      headers/runtime expose CL_VERSION_3_0 (detected below).
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
  message(STATUS "OpenCL 3.0 headers detected -- 3.0-only tests enabled")
  set(CLPEAK_OPENCL_TARGET_VERSION 300)
else()
  message(STATUS "OpenCL 1.2 baseline -- 3.0-only tests disabled")
  set(CLPEAK_OPENCL_TARGET_VERSION 120)
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

# ---- Optional backends: Vulkan (default-on if Vulkan loader present)
#      and CUDA (default-on if CUDA Toolkit driver API + NVRTC are present).
find_package(Vulkan)
find_package(CUDAToolkit)
if(Vulkan_FOUND)
  message(STATUS "Vulkan found: ${Vulkan_LIBRARIES} -- Vulkan backend enabled")
  list(APPEND CLPEAK_SOURCE_FILES ${CLPEAK_VK_SOURCES})
else()
  message(WARNING "Vulkan SDK/loader not found -- Vulkan tests will be disabled. Install the Vulkan SDK to enable the Vulkan backend.")
endif()
# CUDA sources are appended to the target after add_executable below, since
# embed_cuda_kernels() needs the target to exist before it can target_sources()
# the generated file onto it.

add_executable(clpeak ${CLPEAK_SOURCE_FILES})
clpeak_setup_version(clpeak)
target_link_libraries(clpeak ${OpenCL_LIBRARIES})
target_include_directories(clpeak PRIVATE ${OpenCL_INCLUDE_DIRS} ${CLPEAK_CORE_INCLUDE_DIRS})
target_compile_definitions(clpeak PRIVATE CL_TARGET_OPENCL_VERSION=${CLPEAK_OPENCL_TARGET_VERSION})
if(CLPEAK_HAS_OPENCL_30)
  target_compile_definitions(clpeak PRIVATE CLPEAK_HAS_OPENCL_30)
endif()

if(Vulkan_FOUND)
  target_link_libraries(clpeak ${Vulkan_LIBRARIES})
  target_include_directories(clpeak PRIVATE ${Vulkan_INCLUDE_DIRS})
  target_compile_definitions(clpeak PRIVATE ENABLE_VULKAN)

  # Compile GLSL shaders to SPIR-V and embed as C++ arrays
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CompileShaders.cmake)
  compile_shaders(
    TARGET clpeak
    SHADERS ${CLPEAK_VK_SHADERS}
  )
endif()

# ---- CUDA backend (NVRTC + driver API; enabled by default when CUDA
#      Toolkit is present).  No nvcc / cubin / fatbin: kernels are .cu
#      source text embedded into the binary and compiled at runtime by
#      NVRTC, mirroring the OpenCL string-kernel pattern.
if(CUDAToolkit_FOUND)
  message(STATUS "CUDA Toolkit found (${CUDAToolkit_VERSION}) -- CUDA backend enabled")
  target_sources(clpeak PRIVATE ${CLPEAK_CUDA_SOURCES})
  target_link_libraries(clpeak CUDA::cuda_driver CUDA::nvrtc)
  target_include_directories(clpeak PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_definitions(clpeak PRIVATE ENABLE_CUDA)
  # NVRTC needs the toolkit include dir to resolve <mma.h>, <cuda_fp16.h>
  # etc. at runtime.  Embed it as a string literal the runtime hands to
  # nvrtcCompileProgram via "-I<dir>".
  list(GET CUDAToolkit_INCLUDE_DIRS 0 _clpeak_cuda_inc)
  target_compile_definitions(clpeak PRIVATE
    CLPEAK_CUDA_INCLUDE_DIR="${_clpeak_cuda_inc}")

  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedCudaKernels.cmake)
  embed_cuda_kernels(
    TARGET clpeak
    KERNELS ${CLPEAK_CUDA_KERNELS}
  )
else()
  message(STATUS "CUDA Toolkit not found -- CUDA backend disabled")
endif()

# ---- Metal backend (Apple silicon only).  Source text is embedded as
#      string literals; the runtime hands it to MTLDevice newLibraryWithSource:
#      so there's no offline metal/metallib step.
if(APPLE)
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  find_library(IOKIT_FRAMEWORK IOKit)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders)
  find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)
  if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK AND MPS_FRAMEWORK AND MPSGRAPH_FRAMEWORK)
    message(STATUS "Metal + MPSGraph frameworks found -- Metal backend enabled")
    target_sources(clpeak PRIVATE ${CLPEAK_MTL_SOURCES})
    target_link_libraries(clpeak ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
                                  ${IOKIT_FRAMEWORK}
                                  ${MPS_FRAMEWORK} ${MPSGRAPH_FRAMEWORK})
    target_compile_definitions(clpeak PRIVATE ENABLE_METAL)
    # Compile .mm files with ObjC ARC -- saves us from sprinkling release
    # calls on autoreleased Metal objects.
    set_source_files_properties(${CLPEAK_MTL_SOURCES}
      PROPERTIES COMPILE_FLAGS "-fobjc-arc")

    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalKernels.cmake)
    embed_metal_kernels(
      TARGET clpeak
      KERNELS ${CLPEAK_MTL_KERNELS}
    )
  endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU"   OR
   CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  target_compile_options(clpeak PRIVATE -fPIC -Wall -Wextra -Wno-unused-parameter)
endif()

# ---- Backend status summary ------------------------------------------------
# Print a single block at the end of configure so the user can see at a
# glance which backends will be built into the resulting clpeak binary.
set(_clpeak_summary "")
string(APPEND _clpeak_summary "  OpenCL : ENABLED  (target ${CLPEAK_OPENCL_TARGET_VERSION}, runtime ${OpenCL_VERSION_STRING})\n")
if(Vulkan_FOUND)
  string(APPEND _clpeak_summary "  Vulkan : ENABLED  (${Vulkan_VERSION})\n")
else()
  string(APPEND _clpeak_summary "  Vulkan : disabled (Vulkan SDK / loader not found)\n")
endif()
if(CUDAToolkit_FOUND)
  string(APPEND _clpeak_summary "  CUDA   : ENABLED  (Toolkit ${CUDAToolkit_VERSION})\n")
else()
  string(APPEND _clpeak_summary "  CUDA   : disabled (CUDA Toolkit not found)\n")
endif()
if(APPLE AND METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK AND MPS_FRAMEWORK AND MPSGRAPH_FRAMEWORK)
  string(APPEND _clpeak_summary "  Metal  : ENABLED  (Apple silicon target, MPSGraph linked)\n")
elseif(APPLE)
  string(APPEND _clpeak_summary "  Metal  : disabled (Metal/Foundation/MPSGraph framework not found)\n")
else()
  string(APPEND _clpeak_summary "  Metal  : disabled (non-Apple host)\n")
endif()
message(STATUS "")
message(STATUS "===============================================================")
message(STATUS "clpeak backend summary")
message(STATUS "")
# Print the multi-line block.  CMake `message(STATUS "...")` re-prefixes
# every newline so we split manually.
string(REPLACE "\n" ";" _summary_lines "${_clpeak_summary}")
foreach(_line IN LISTS _summary_lines)
  if(NOT "${_line}" STREQUAL "")
    message(STATUS "${_line}")
  endif()
endforeach()
message(STATUS "===============================================================")
message(STATUS "")

install(TARGETS clpeak RUNTIME DESTINATION bin)
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/clpeak")

enable_testing()
add_test(clpeak_test_run clpeak)

set(CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-${CLPEAK_VERSION_STR}")
set(CPACK_GENERATOR "ZIP")
include (CPack)
