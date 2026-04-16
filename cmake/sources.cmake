# cmake/sources.cmake
# Single source of truth for clpeak core source files.
# Set CLPEAK_ROOT before including this file.

set(CLPEAK_CORE_SOURCES
    ${CLPEAK_ROOT}/src/common.cpp
    ${CLPEAK_ROOT}/src/result_store.cpp
    ${CLPEAK_ROOT}/src/clpeak.cpp
    ${CLPEAK_ROOT}/src/options.cpp
    ${CLPEAK_ROOT}/src/global_bandwidth.cpp
    ${CLPEAK_ROOT}/src/local_bandwidth.cpp
    ${CLPEAK_ROOT}/src/image_bandwidth.cpp
    ${CLPEAK_ROOT}/src/atomic_throughput.cpp
    ${CLPEAK_ROOT}/src/transfer_bandwidth.cpp
    ${CLPEAK_ROOT}/src/kernel_latency.cpp
)

set(CLPEAK_CORE_INCLUDE_DIRS
    ${CLPEAK_ROOT}/include
    ${CLPEAK_ROOT}/src/kernels
)
