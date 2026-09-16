// Stand-in for the build_config.h LiteRT's own CMake generates at configure
// time (litert/build_common/build_config.h.in).  clpeak never builds LiteRT:
// it compiles against these headers and dlopens the runtime, and the two
// toggles below only trim declarations from the headers, so both stay off.
#ifndef LITERT_BUILD_COMMON_BUILD_CONFIG_H_
#define LITERT_BUILD_COMMON_BUILD_CONFIG_H_

#define LITERT_BUILD_CONFIG_DISABLE_GPU 0
#define LITERT_BUILD_CONFIG_DISABLE_NPU 0

#endif  // LITERT_BUILD_COMMON_BUILD_CONFIG_H_
