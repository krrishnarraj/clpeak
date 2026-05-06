#ifndef MTL_PEAK_H
#define MTL_PEAK_H

#ifdef ENABLE_METAL

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <bitset>
#include <common.h>
#include <benchmark_constants.h>
#include <logger.h>
#include <clpeak.h>      // Benchmark enum
#include <backend_gating.h>  // centralized benchmark gating

struct CliOptions; // forward decl

// Forward-declare the implementation as opaque pointers so this header can
// be included from pure C++ TUs (entry.cpp).  All Objective-C / Metal
// types live behind these void* pimpls in mtl_peak.mm.
struct MetalDeviceImpl;
struct MetalPeakImpl;

struct mtl_device_info_t {
  std::string deviceName;
  std::string osVersion;          // "macOS 26.4.1"

  uint64_t recommendedMaxWorkingSetSize;  // ~ usable VRAM-equivalent
  uint64_t maxBufferLength;
  uint32_t maxThreadsPerThreadgroup;

  bool isAppleSilicon;            // gate: refuse non-Apple7+ devices
  bool fp16Supported;             // always true on Apple silicon
  bool simdgroupMatrixFP16Supported; // M1+ (Apple7)
  bool simdgroupMatrixBF16Supported; // M3+ (Apple9)
  bool simdgroupMatrixInt8Supported; // M3+ (Apple9) -- int8 simdgroup_matrix
  bool mpsGraphBF16Supported;     // MPSGraph bf16 matmul: macOS 14 + Apple9 (M3)
  bool atomic64Supported;         // 64-bit int atomics: Apple8+ (M2)
  uint32_t appleFamily;           // largest MTLGPUFamilyApple<N> the device supports
  uint32_t gpuCoreCount;          // GPU core count (e.g. 8 on M1 base, 32 on M1 Max). 0 = unknown.
};

class MetalDevice
{
public:
  MetalDevice();
  ~MetalDevice();

  bool init(int devIndex);
  void cleanup();

  mtl_device_info_t info;

  // Opaque -- defined in mtl_peak.mm.
  MetalDeviceImpl *impl;
};

// Variant of one compute-peak kernel (matches the Vulkan / CUDA shape).
struct mtl_compute_variant_t
{
  const char *label;
  const char *kernelName;          // function name inside the .metal source
  const char *src;                 // .metal source text (may be shared by sibling variants)
  const char *srcName;
};

struct mtl_compute_desc_t
{
  const char *title;
  const char *resultTag;
  const char *unit;                // "gflops" / "gops" / "tflops" / "tops"
  double      unitDivider;         // 1e9 = G* (default when 0), 1e12 = T*

  // Single-variant fallback.
  const char *metricLabel;
  const char *kernelName;
  const char *src;
  const char *srcName;

  // Multi-variant.
  const mtl_compute_variant_t *variants;
  uint32_t numVariants;

  uint32_t workPerWI;
  uint32_t elemSize;
  uint32_t threadsPerGroup;        // 0 => default 256.  Simdgroup tests use 32.
  uint32_t outElemsPerGroup;       // 0 => threadsPerGroup.

  // Scalar A passed at buffer index 1 (output buffer is index 0).
  const void *scalarArg;
  uint32_t    scalarSize;

  bool        skip;
  const char *skipMsg;
  const char *extraAttribKey;
  const char *extraAttribVal;
};

class MetalPeak
{
public:
  std::unique_ptr<logger> log;
  unsigned int warmupCount;
  unsigned int specifiedIters;
  bool forceIters;
  int  deviceIndex; // -1 = run all

  BackendGating gating;

  MetalPeak();
  ~MetalPeak();

  void applyOptions(const CliOptions &opts);
  int runAll();

  int runComputeSP(MetalDevice &dev, benchmark_config_t &cfg);
  int runComputeHP(MetalDevice &dev, benchmark_config_t &cfg);
  int runComputeMP(MetalDevice &dev, benchmark_config_t &cfg);
  int runComputeInt8DP(MetalDevice &dev, benchmark_config_t &cfg);
  int runComputeInt4Packed(MetalDevice &dev, benchmark_config_t &cfg);
  int runGlobalBandwidth(MetalDevice &dev, benchmark_config_t &cfg);
  int runKernelLatency(MetalDevice &dev, benchmark_config_t &cfg);
  int runSimdgroupMatrix(MetalDevice &dev, benchmark_config_t &cfg);
  int runSimdgroupMatrixInt(MetalDevice &dev, benchmark_config_t &cfg);
  int runMpsGemm(MetalDevice &dev, benchmark_config_t &cfg);
  int runMpsGemmInt(MetalDevice &dev, benchmark_config_t &cfg);
  int runLocalBandwidth(MetalDevice &dev, benchmark_config_t &cfg);
  int runImageBandwidth(MetalDevice &dev, benchmark_config_t &cfg);
  int runAtomicThroughput(MetalDevice &dev, benchmark_config_t &cfg);
  int runAtomicThroughputFp(MetalDevice &dev, benchmark_config_t &cfg);

  // Internal -- exposed only so they can be reached from mtl_peak.mm without
  // an extra friend declaration.
  MetalPeakImpl *impl;

private:
  int runComputeKernel(MetalDevice &dev, benchmark_config_t &cfg,
                       const mtl_compute_desc_t &d);
};

// Embedded Metal kernel source text (generated at build time by
// embed_metal_kernels()).  One pair of externs per .metal file.
namespace mtl_kernels {
  extern const char *compute_sp_src;
  extern const char *compute_sp_name;
  extern const char *compute_hp_src;
  extern const char *compute_hp_name;
  extern const char *compute_mp_src;
  extern const char *compute_mp_name;
  extern const char *compute_int8_dp_src;
  extern const char *compute_int8_dp_name;
  extern const char *compute_int4_packed_src;
  extern const char *compute_int4_packed_name;
  extern const char *global_bandwidth_src;
  extern const char *global_bandwidth_name;
  extern const char *kernel_latency_src;
  extern const char *kernel_latency_name;
  extern const char *simdgroup_matrix_fp16_src;
  extern const char *simdgroup_matrix_fp16_name;
  extern const char *simdgroup_matrix_bf16_src;
  extern const char *simdgroup_matrix_bf16_name;
  extern const char *simdgroup_matrix_int8_src;
  extern const char *simdgroup_matrix_int8_name;
  extern const char *local_bandwidth_src;
  extern const char *local_bandwidth_name;
  extern const char *image_bandwidth_src;
  extern const char *image_bandwidth_name;
  extern const char *atomic_throughput_src;
  extern const char *atomic_throughput_name;
  extern const char *atomic_throughput_float_src;
  extern const char *atomic_throughput_float_name;
  extern const char *atomic_throughput_ulong_src;
  extern const char *atomic_throughput_ulong_name;
}

#endif // ENABLE_METAL
#endif // MTL_PEAK_H
