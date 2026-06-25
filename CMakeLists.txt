cmake_minimum_required(VERSION 3.20)

# Prefer clang over the platform-default compiler when the user has not pinned
# one.  Two reasons, both from the CPU backend:
#  - codegen: the kernels keep 24 independent FMA accumulator chains in flight;
#    clang schedules those cleanly, but GCC<=14 serialises them into a single
#    register (and skips the k-loop unroll), roughly halving fp32/fp64.
#  - dtype coverage: MSVC has no GNU -m/-march ISA flags and defines no
#    __ARM_FEATURE_* / AVX-512 sub-feature macros, so the advanced-dtype TUs
#    (fp16 / bf16 / int8-dot / matrix) cannot be built with cl.exe at all;
#    clang-cl gives Windows dtype parity with the Linux build.
# Must run before project().  Respects any explicit choice: CC/CXX env vars,
# -DCMAKE_C[XX]_COMPILER, -T <toolset>, or a toolchain file.  (macOS already
# defaults to AppleClang and is left alone.)
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED CMAKE_C_COMPILER
   AND NOT DEFINED ENV{CXX} AND NOT DEFINED ENV{CC}
   AND NOT CMAKE_TOOLCHAIN_FILE)
    if(CMAKE_HOST_UNIX AND NOT CMAKE_HOST_APPLE)
        find_program(CLPEAK_CLANG_C   NAMES clang)
        find_program(CLPEAK_CLANG_CXX NAMES clang++)
        if(CLPEAK_CLANG_C AND CLPEAK_CLANG_CXX)
            set(CMAKE_C_COMPILER   "${CLPEAK_CLANG_C}"   CACHE FILEPATH "C compiler")
            set(CMAKE_CXX_COMPILER "${CLPEAK_CLANG_CXX}" CACHE FILEPATH "C++ compiler")
            message(STATUS "clpeak: preferring clang (${CLPEAK_CLANG_CXX}) for CPU-backend codegen; pass -DCMAKE_CXX_COMPILER=g++ to override")
        endif()
    elseif(CMAKE_HOST_WIN32)
        if(CMAKE_GENERATOR MATCHES "Visual Studio")
            # The VS generator selects the compiler via the toolset (-T), not
            # CMAKE_CXX_COMPILER.  Switch to ClangCL only when the VS "C++
            # Clang compiler" component is actually installed (probed via
            # vswhere), so a stock MSVC install still configures fine.
            # CMAKE_GENERATOR_TOOLSET is cached after the first configure, so
            # this runs once per build dir and stays consistent on reconfigure.
            if(NOT CMAKE_GENERATOR_TOOLSET)
                set(_clpeak_pf86 "ProgramFiles(x86)")  # indirection: () in $ENV{} confuses the parser
                set(_clpeak_vswhere "$ENV{${_clpeak_pf86}}/Microsoft Visual Studio/Installer/vswhere.exe")
                if(NOT EXISTS "${_clpeak_vswhere}")
                    set(_clpeak_vswhere "$ENV{ProgramFiles}/Microsoft Visual Studio/Installer/vswhere.exe")
                endif()
                if(EXISTS "${_clpeak_vswhere}")
                    execute_process(
                        COMMAND "${_clpeak_vswhere}" -latest -products * -find "VC/Tools/Llvm/**/bin/clang-cl.exe"
                        OUTPUT_VARIABLE _clpeak_vs_clang
                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
                    if(_clpeak_vs_clang)
                        set(CMAKE_GENERATOR_TOOLSET "ClangCL")
                        message(STATUS "clpeak: preferring clang-cl (-T ClangCL) for CPU-backend dtype coverage; pass -T v143 to keep MSVC")
                    endif()
                endif()
            endif()
        else()
            # Ninja / Makefiles on Windows: pick clang-cl from PATH (LLVM install).
            find_program(CLPEAK_CLANG_CL NAMES clang-cl)
            if(CLPEAK_CLANG_CL)
                set(CMAKE_C_COMPILER   "${CLPEAK_CLANG_CL}" CACHE FILEPATH "C compiler")
                set(CMAKE_CXX_COMPILER "${CLPEAK_CLANG_CL}" CACHE FILEPATH "C++ compiler")
                message(STATUS "clpeak: preferring clang-cl (${CLPEAK_CLANG_CL}) for CPU-backend dtype coverage; pass -DCMAKE_CXX_COMPILER=cl to override")
            endif()
        endif()
    endif()
endif()

project(clpeak)

set(CLPEAK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
include(${CLPEAK_ROOT}/src/common/cmake/version.cmake)
include(GNUInstallDirs)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Backend options
option(CLPEAK_ENABLE_OPENCL "Build OpenCL backend" ON)
option(CLPEAK_ENABLE_VULKAN "Build Vulkan backend" ON)
option(CLPEAK_ENABLE_CUDA   "Build CUDA backend"   ON)
option(CLPEAK_ENABLE_ROCM   "Build ROCm/HIP backend" ON)
option(CLPEAK_ENABLE_METAL  "Build Metal backend"  ON)
option(CLPEAK_ENABLE_ONEAPI "Build oneAPI/SYCL backend (requires icpx/icx or clang++ -fsycl)" ON)
option(CLPEAK_ENABLE_CPU    "Build native CPU backend (plain C++ / std::thread)" ON)
# OFF (default): build a portable binary (SSE4.2 / armv8-a baseline) that runtime-
# dispatches to higher ISA — safe to distribute across machines.  ON: tune the
# whole CPU backend for the BUILD host (-march/-mcpu=native); fastest for a local
# build, but the resulting binary may use illegal instructions on an older CPU.
option(CLPEAK_CPU_NATIVE_ARCH "Tune the CPU backend for the build host (non-portable binary)" OFF)

# The native CPU backend needs a threading library (always available).
if(CLPEAK_ENABLE_CPU)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads QUIET)
endif()

# Find Vulkan (needed by src/vulkan/CMakeLists.txt)
if(CLPEAK_ENABLE_VULKAN)
    find_package(Vulkan QUIET)
endif()
# Find CUDA (needed by src/cuda/CMakeLists.txt)
if(CLPEAK_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
endif()
# Find ROCm/HIP (needed by src/rocm/CMakeLists.txt)
if(CLPEAK_ENABLE_ROCM)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    find_package(hip QUIET CONFIG)
    find_package(rocblas QUIET CONFIG)
    find_package(hipblaslt QUIET CONFIG)

    # Kernels are compiled ahead-of-time with hipcc (--genco), so the shipped
    # binary needs only the HIP runtime -- no HIPRTC, no ROCm headers.
    find_program(CLPEAK_HIPCC NAMES hipcc HINTS /opt/rocm/bin)
    if(CLPEAK_HIPCC)
        set(CLPEAK_HIPCC_FOUND TRUE)
    else()
        set(CLPEAK_HIPCC_FOUND FALSE)
    endif()

    find_path(CLPEAK_ROCWMMA_INCLUDE_DIR
        NAMES rocwmma/rocwmma.hpp
        HINTS /opt/rocm/include)
    if(CLPEAK_ROCWMMA_INCLUDE_DIR)
        set(CLPEAK_ROCWMMA_FOUND TRUE)
    else()
        set(CLPEAK_ROCWMMA_FOUND FALSE)
    endif()
endif()
# Find Metal frameworks (needed by src/metal/CMakeLists.txt and summary)
if(APPLE AND CLPEAK_ENABLE_METAL)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(IOKIT_FRAMEWORK IOKit)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(MPSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)
endif()
# Find Intel oneAPI / SYCL (needed by src/oneapi/CMakeLists.txt)
if(CLPEAK_ENABLE_ONEAPI)
    # IntelSYCL ships with the Intel oneAPI Base Toolkit.  When compiling with
    # icpx (or clang++ -fsycl), it exposes add_sycl_to_target() and the
    # IntelSYCL::SYCL_CXX imported target.
    find_package(IntelSYCL CONFIG QUIET)
    # Optional vendor libraries.  MKL_SYCL is the SYCL-enabled BLAS subset of
    # oneMKL; only present when the user installed the MKL component.
    find_package(MKL CONFIG QUIET)

    # SYCL device code must be compiled AND linked by a SYCL-aware compiler
    # (CMAKE_CXX_COMPILER_ID == "IntelLLVM").  The kernels live in a static
    # lib that is link-resolved into the clpeak executable, so the *whole*
    # binary must be built with icpx/icx -- MSVC and GCC cannot embed device
    # images.
    #   Linux:   cmake -B build -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx
    #   Windows: cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=icx -DCMAKE_C_COMPILER=icx
    if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        set(CLPEAK_ONEAPI_COMPILER_OK TRUE)
    else()
        set(CLPEAK_ONEAPI_COMPILER_OK FALSE)
    endif()
endif()

# Always build common
add_subdirectory(src/common)

# Conditionally build backends
add_subdirectory(src/opencl)
add_subdirectory(src/vulkan)
add_subdirectory(src/cuda)
add_subdirectory(src/rocm)
add_subdirectory(src/metal)
add_subdirectory(src/oneapi)
add_subdirectory(src/cpu)

# CLI executable
add_subdirectory(src/cli)

# ---- Backend status summary ------------------------------------------------
# Print a single block at the end of configure so the user can see at a
# glance which backends will be built into the resulting clpeak binary.
set(_clpeak_summary "")
if(CLPEAK_ENABLE_OPENCL)
    string(APPEND _clpeak_summary "  OpenCL : ENABLED\n")
else()
    string(APPEND _clpeak_summary "  OpenCL : disabled (CLPEAK_ENABLE_OPENCL=OFF)\n")
endif()
if(CLPEAK_ENABLE_VULKAN AND Vulkan_FOUND)
    string(APPEND _clpeak_summary "  Vulkan : ENABLED  (${Vulkan_VERSION})\n")
elseif(CLPEAK_ENABLE_VULKAN)
    string(APPEND _clpeak_summary "  Vulkan : disabled (Vulkan SDK / loader not found)\n")
else()
    string(APPEND _clpeak_summary "  Vulkan : disabled (CLPEAK_ENABLE_VULKAN=OFF)\n")
endif()
if(CLPEAK_ENABLE_CUDA AND CUDAToolkit_FOUND)
    string(APPEND _clpeak_summary "  CUDA   : ENABLED  (Toolkit ${CUDAToolkit_VERSION})\n")
elseif(CLPEAK_ENABLE_CUDA)
    string(APPEND _clpeak_summary "  CUDA   : disabled (CUDA Toolkit not found)\n")
else()
    string(APPEND _clpeak_summary "  CUDA   : disabled (CLPEAK_ENABLE_CUDA=OFF)\n")
endif()
if(CLPEAK_ENABLE_ROCM AND hip_FOUND AND CLPEAK_HIPCC_FOUND)
    set(_clpeak_rocm_features "HIP")
    if(rocblas_FOUND AND TARGET roc::rocblas)
        string(APPEND _clpeak_rocm_features ", rocBLAS")
    endif()
    if(hipblaslt_FOUND AND TARGET roc::hipblaslt)
        string(APPEND _clpeak_rocm_features ", hipBLASLt")
    endif()
    if(CLPEAK_ROCWMMA_FOUND)
        string(APPEND _clpeak_rocm_features ", rocWMMA")
    endif()
    string(APPEND _clpeak_summary "  ROCm   : ENABLED  (${_clpeak_rocm_features})\n")
elseif(CLPEAK_ENABLE_ROCM AND NOT hip_FOUND)
    string(APPEND _clpeak_summary "  ROCm   : disabled (ROCm/HIP package not found)\n")
elseif(CLPEAK_ENABLE_ROCM)
    string(APPEND _clpeak_summary "  ROCm   : disabled (hipcc not found)\n")
else()
    string(APPEND _clpeak_summary "  ROCm   : disabled (CLPEAK_ENABLE_ROCM=OFF)\n")
endif()
if(CLPEAK_ENABLE_METAL AND APPLE AND METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK AND MPS_FRAMEWORK AND MPSGRAPH_FRAMEWORK)
    string(APPEND _clpeak_summary "  Metal  : ENABLED  (Apple silicon target, MPSGraph linked)\n")
elseif(CLPEAK_ENABLE_METAL AND APPLE)
    string(APPEND _clpeak_summary "  Metal  : disabled (Metal/Foundation/MPSGraph framework not found)\n")
elseif(CLPEAK_ENABLE_METAL)
    string(APPEND _clpeak_summary "  Metal  : disabled (non-Apple host)\n")
else()
    string(APPEND _clpeak_summary "  Metal  : disabled (CLPEAK_ENABLE_METAL=OFF)\n")
endif()
if(CLPEAK_ENABLE_ONEAPI AND IntelSYCL_FOUND AND CLPEAK_ONEAPI_COMPILER_OK)
    set(_clpeak_oneapi_features "SYCL")
    if(MKL_FOUND)
        string(APPEND _clpeak_oneapi_features ", oneMKL")
    endif()
    if(CLPEAK_ONEAPI_HAS_JOINT_MATRIX)
        string(APPEND _clpeak_oneapi_features ", joint_matrix")
    endif()
    string(APPEND _clpeak_summary "  oneAPI : ENABLED  (${_clpeak_oneapi_features})\n")
elseif(CLPEAK_ENABLE_ONEAPI AND IntelSYCL_FOUND AND NOT CLPEAK_ONEAPI_COMPILER_OK)
    if(WIN32)
        string(APPEND _clpeak_summary "  oneAPI : disabled (needs icx - reconfigure with -G Ninja -DCMAKE_CXX_COMPILER=icx, current is ${CMAKE_CXX_COMPILER_ID})\n")
    else()
        string(APPEND _clpeak_summary "  oneAPI : disabled (needs icpx - reconfigure with -DCMAKE_CXX_COMPILER=icpx, current is ${CMAKE_CXX_COMPILER_ID})\n")
    endif()
elseif(CLPEAK_ENABLE_ONEAPI)
    if(WIN32)
        string(APPEND _clpeak_summary "  oneAPI : disabled (IntelSYCL not found - run setvars.bat then reconfigure with -G Ninja -DCMAKE_CXX_COMPILER=icx)\n")
    else()
        string(APPEND _clpeak_summary "  oneAPI : disabled (IntelSYCL not found - source setvars.sh then reconfigure with -DCMAKE_CXX_COMPILER=icpx)\n")
    endif()
else()
    string(APPEND _clpeak_summary "  oneAPI : disabled (CLPEAK_ENABLE_ONEAPI=OFF)\n")
endif()
if(CLPEAK_ENABLE_CPU AND CLPEAK_CPU_NATIVE_ARCH)
    string(APPEND _clpeak_summary "  CPU    : ENABLED  (NATIVE_ARCH=ON -- tuned for build host, non-portable binary)\n")
elseif(CLPEAK_ENABLE_CPU)
    string(APPEND _clpeak_summary "  CPU    : ENABLED  (portable: baseline + runtime ISA dispatch, -DCLPEAK_CPU_NATIVE_ARCH=ON to tune for this host)\n")
else()
    string(APPEND _clpeak_summary "  CPU    : disabled (CLPEAK_ENABLE_CPU=OFF)\n")
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

install(TARGETS clpeak RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(FILES LICENSE DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/clpeak")
enable_testing()
add_test(clpeak_test_run clpeak)

# --- CPack: self-contained, clearly-named zip per build --------------------
# Name: clpeak-<version>-<os>-<arch>[-<backend>], e.g.
# clpeak-1.1.8-linux-x86_64-cuda.  Archive generators wrap the contents in a
# top-level dir of the same name, so each zip extracts to one clean folder.
string(TOLOWER "${CMAKE_SYSTEM_NAME}" _clpeak_os)
if(_clpeak_os STREQUAL "darwin")
  set(_clpeak_os "macos")
endif()
set(CLPEAK_PACKAGE_SUFFIX "" CACHE STRING
    "Backend tag appended to the package name (e.g. cuda, rocm, oneapi)")
set(_clpeak_pkg "clpeak-${CLPEAK_VERSION_STR}-${_clpeak_os}-${CMAKE_SYSTEM_PROCESSOR}")
if(NOT CLPEAK_PACKAGE_SUFFIX STREQUAL "")
  set(_clpeak_pkg "${_clpeak_pkg}-${CLPEAK_PACKAGE_SUFFIX}")
endif()
set(CPACK_PACKAGE_FILE_NAME "${_clpeak_pkg}")
set(CPACK_GENERATOR "ZIP")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY 1)
include(CPack)
