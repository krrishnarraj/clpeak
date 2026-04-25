#ifndef CUDA_PEAK_H
#define CUDA_PEAK_H

#ifdef ENABLE_CUDA

#include <cuda.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <common.h>
#include <benchmark_constants.h>
#include <logger.h>

// CUDA device info (mirrors vk_device_info_t for display + per-test gating).
struct cuda_device_info_t {
  std::string deviceName;
  std::string driverVersion;     // e.g. "12.6"
  std::string runtimeVersion;    // NVRTC version
  std::string archName;          // "sm_120" etc.

  int major;                     // compute-capability major
  int minor;                     // compute-capability minor
  int numSMs;
  int maxThreadsPerBlock;
  uint64_t totalGlobalMem;
  int clockRateKHz;

  // Per-test capability flags
  bool fp16Supported;            // cc >= 5.3
  bool bf16Supported;            // cc >= 8.0 (Ampere)
  bool dp4aSupported;            // cc >= 6.1 (Pascal)
  bool wmmaSupported;            // cc >= 7.0 (Volta) -- fp16 wmma
  bool wmmaInt8Supported;        // cc >= 7.2 (Turing) -- int8 wmma fragments
  bool fp8MmaSupported;          // cc >= 8.9 (Ada) -- inline mma.sync.e4m3/e5m2
};

// One CUDA device + the bookkeeping needed to launch kernels through the
// driver API.  Modules are loaded lazily and cached per-source-text.
class CudaDevice
{
public:
  CUdevice  device;
  CUcontext context;
  CUstream  stream;
  cuda_device_info_t info;

  CudaDevice();
  ~CudaDevice();

  bool init(int devIndex);
  void cleanup();

  // NVRTC-compile a .cu source string and load the resulting PTX into a
  // module, then resolve named kernels.  Caches by source pointer; the same
  // source compiled for two device archs lives in two cache entries.
  // Returns true on success; on compile failure logs the NVRTC log to stderr
  // and returns false.
  bool getKernel(const char *src, const char *srcName,
                 const char *kernelName, CUfunction &fn,
                 const std::vector<const char *> &extraOpts = {});

private:
  // moduleCache key = source pointer (string identity is the simplest
  // viable cache key because every .cu blob is a unique extern in the
  // generated cpp).  Value = loaded module handle.
  std::unordered_map<const char *, CUmodule> moduleCache;
};

// Variant of a single compute-peak kernel.  All variants of one benchmark
// share the same output buffer + dispatch geometry; only the kernel symbol
// (and possibly source file) differs.
struct cuda_compute_variant_t
{
  const char *label;             // column / xmlRecord key, e.g. "mp", "mp2"
  const char *kernelName;        // CUDA kernel symbol (extern "C")
  const char *src;               // .cu source text (may be shared by sibling
                                 // variants emitting from one file)
  const char *srcName;           // file stem for NVRTC diagnostics
};

struct cuda_compute_desc_t
{
  const char *title;             // header line
  const char *xmlTag;            // outer XML tag
  const char *unit;              // "gflops" / "giops" / "tflops"

  // Single-variant fallback (used when variants==nullptr).
  const char *metricLabel;
  const char *kernelName;
  const char *src;
  const char *srcName;

  // Multi-variant (preferred when set).
  const cuda_compute_variant_t *variants;
  uint32_t numVariants;

  // Scaling.
  uint32_t workPerWI;            // matches the kernel's per-thread op budget
  uint32_t elemSize;             // output element size
  uint32_t blockSize;            // threads per block; 0 => default 256.
                                 // WMMA kernels use 32 (one warp) per block.
  uint32_t outElemsPerBlock;     // output elements written per block;
                                 // 0 => defaults to blockSize.

  // Scalar A passed as the second kernel argument (after the output buffer).
  // Stored by the caller; pointer kept here so cuLaunchKernel can index it.
  // Using a 4-byte slot is enough for float / int32 / etc.
  const void *scalarArg;
  uint32_t    scalarSize;        // 0 => no scalar arg

  // Optional gates / attributes
  bool        skip;
  const char *skipMsg;
  const char *extraAttribKey;
  const char *extraAttribVal;

  // Optional NVRTC compile flags shared by all variants (e.g. wmma needs
  // --gpu-architecture matching the device, but per-test extras like
  // -default-device or -DSOMETHING go here).
  const char *const *extraNvrtcOpts;
  uint32_t           numExtraNvrtcOpts;
};

class CudaPeak
{
public:
  std::unique_ptr<logger> log;
  unsigned int warmupCount;
  unsigned int specifiedIters;
  bool forceIters;
  bool listDevices;

  CudaPeak();
  ~CudaPeak();

  int parseArgs(int argc, char **argv);
  int runAll();

  int runComputeSP(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeHP(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeDP(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeMP(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeBF16(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeInt8DP(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeInt4Packed(CudaDevice &dev, benchmark_config_t &cfg);
  int runGlobalBandwidth(CudaDevice &dev, benchmark_config_t &cfg);
  int runTransferBandwidth(CudaDevice &dev, benchmark_config_t &cfg);
  int runKernelLatency(CudaDevice &dev, benchmark_config_t &cfg);
  int runWmma(CudaDevice &dev, benchmark_config_t &cfg);

private:
  bool initialised;
  std::vector<int> devIndices;

  bool initDriver();

  // Event-timed launch helper.  Runs the kernel iters times after warmup,
  // returns mean per-launch time in microseconds.
  float runKernel(CudaDevice &dev, CUfunction fn,
                  uint32_t gridX, uint32_t blockX,
                  void **args, unsigned int iters);

  int runComputeKernel(CudaDevice &dev, benchmark_config_t &cfg,
                       const cuda_compute_desc_t &d);
};

// Embedded CUDA kernel source text (generated at build time).
namespace cuda_kernels {
  extern const char *compute_sp_src;
  extern const char *compute_sp_name;
  extern const char *compute_hp_src;
  extern const char *compute_hp_name;
  extern const char *compute_dp_src;
  extern const char *compute_dp_name;
  extern const char *compute_mp_src;
  extern const char *compute_mp_name;
  extern const char *compute_bf16_src;
  extern const char *compute_bf16_name;
  extern const char *compute_int8_dp_src;
  extern const char *compute_int8_dp_name;
  extern const char *compute_int4_packed_src;
  extern const char *compute_int4_packed_name;
  extern const char *global_bandwidth_src;
  extern const char *global_bandwidth_name;
  extern const char *kernel_latency_src;
  extern const char *kernel_latency_name;
  extern const char *wmma_fp16_src;
  extern const char *wmma_fp16_name;
  extern const char *wmma_bf16_src;
  extern const char *wmma_bf16_name;
  extern const char *wmma_int8_src;
  extern const char *wmma_int8_name;
  extern const char *wmma_int8_k32_src;
  extern const char *wmma_int8_k32_name;
  extern const char *wmma_fp8_e4m3_src;
  extern const char *wmma_fp8_e4m3_name;
  extern const char *wmma_fp8_e5m2_src;
  extern const char *wmma_fp8_e5m2_name;
}

#endif // ENABLE_CUDA
#endif // CUDA_PEAK_H
