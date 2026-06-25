if(NOT CLPEAK_ENABLE_ONEAPI OR NOT IntelSYCL_FOUND OR NOT CLPEAK_ONEAPI_COMPILER_OK)
  return()
endif()

add_library(peak_oneapi STATIC
    oneapi_peak.cpp
    oneapi_device.cpp
    compute_kernel.cpp
    compute_float.cpp
    compute_int.cpp
    joint_matrix.cpp
    onemkl.cpp
    global_bandwidth.cpp
    local_bandwidth.cpp
    image_bandwidth.cpp
    transfer_bandwidth.cpp
    kernel_latency.cpp
)

target_link_libraries(peak_oneapi PUBLIC peak_common)
target_include_directories(peak_oneapi PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/../../include
)
target_compile_definitions(peak_oneapi PRIVATE ENABLE_ONEAPI)
target_compile_definitions(peak_common  PRIVATE ENABLE_ONEAPI)

# IntelSYCL package exposes add_sycl_to_target() which attaches the -fsycl
# compile/link flags, the SYCL headers, and the device-side compilation step
# to the given target / sources.
get_target_property(_clpeak_oneapi_srcs peak_oneapi SOURCES)
add_sycl_to_target(TARGET peak_oneapi SOURCES ${_clpeak_oneapi_srcs})

# Optional oneMKL (SYCL-enabled BLAS) for the vendor-library GEMM benchmark.
if(MKL_FOUND AND TARGET MKL::MKL_SYCL)
  target_link_libraries(peak_oneapi PUBLIC MKL::MKL_SYCL)
  target_compile_definitions(peak_oneapi PRIVATE CLPEAK_ONEAPI_HAS_ONEMKL)
endif()

# joint_matrix is a header-only SYCL extension.  Probe for the matrix header
# under -fsycl so we only enable the XMX benchmark when the toolchain ships
# it; older oneAPI releases lack the public header.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsycl")
check_cxx_source_compiles(
    "#include <sycl/sycl.hpp>
     #include <sycl/ext/oneapi/matrix/matrix.hpp>
     int main() { return 0; }"
    CLPEAK_ONEAPI_HAS_JOINT_MATRIX)
unset(CMAKE_REQUIRED_FLAGS)
if(CLPEAK_ONEAPI_HAS_JOINT_MATRIX)
  target_compile_definitions(peak_oneapi PRIVATE CLPEAK_ONEAPI_HAS_JOINT_MATRIX)
endif()
# Re-export so the top-level summary can show the joint_matrix feature.
set(CLPEAK_ONEAPI_HAS_JOINT_MATRIX ${CLPEAK_ONEAPI_HAS_JOINT_MATRIX} PARENT_SCOPE)
