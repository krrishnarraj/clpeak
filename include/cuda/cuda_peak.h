#ifndef CUDA_PEAK_H
#define CUDA_PEAK_H

#ifdef ENABLE_CUDA

#include <cuda.h>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <common/common.h>
#include <common/logger.h>
#include <common/peak.h>
#include <common/inventory.h>

struct CliOptions; // forward decl

// One embedded kernel: a precompiled multi-arch fatbin (nvcc -fatbin at build
// time) plus its file stem for diagnostics.  Defined fully at the bottom of
// this header; forward-declared here so the descriptor structs can point at it.
namespace cuda_kernels { struct Blob; }

// CUDA device info (mirrors vk_device_info_t for display + per-test gating).
struct cuda_device_info_t {
  std::string deviceName;
  std::string driverVersion;     // e.g. "12.6"
  std::string runtimeVersion;    // CUDA toolkit the fatbins were built with
  std::string archName;          // "sm_120" etc.

  DeviceType deviceType = DeviceType::Unknown;

  int major;                     // compute-capability major
  int minor;                     // compute-capability minor
  int numSMs;
  int maxThreadsPerBlock;
  uint64_t totalGlobalMem;
  int clockRateKHz;
  // CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE -- the last level of cache in front of
  // device memory.  0 when the driver does not report one.
  uint64_t l2CacheSize = 0;

  // Per-test capability flags
  bool fp16Supported;            // cc >= 5.3
  bool bf16Supported;            // cc >= 8.0 (Ampere)
  bool dp4aSupported;            // cc >= 6.1 (Pascal)
  bool wmmaSupported;            // cc >= 7.0 (Volta) -- fp16 wmma
  bool wmmaInt8Supported;        // cc >= 7.2 (Turing) -- int8 wmma fragments
  bool fp8MmaSupported;          // cc >= 8.9 (Ada) -- inline mma.sync.e4m3/e5m2
                                 // (also gates the fp16-accumulate fp8 variant)
  bool fp8MmaSparseSupported;    // cc >= 8.9 (Ada) -- mma.sp.e4m3 2:4 sparsity
                                 // at m16n8k64 (double the dense K), fp32 accum
  bool fp8MmaSparseF16Supported; // cc 12.x -- the same sparse shape with an fp16
                                 // accumulator; ptxas rejects it on sm_89 even
                                 // though the DENSE fp16-accum fp8 mma is fine
                                 // there.  Provisional floor, see CMakeLists.
  bool fp4MmaSupported;          // cc 12.x (consumer Blackwell) -- sm_120a/121a
                                 // raw mma.sync FP4/MXFP4 microbench kernels
  bool fp4MmaSparseSupported;    // cc 12.x (consumer Blackwell) -- sm_120a/121a
                                 // raw mma.sp FP4 2:4 microbench kernels
  bool fp4GemmSupported;         // cc >= 10.0 (all Blackwell) -- block-scaled
                                 // FP4 GEMM via cuBLASLt; covers datacenter
                                 // (sm_100/103, tcgen05) too, where the raw
                                 // mma.sync kernels above do not apply. cuBLASLt
                                 // self-skips (0 heuristics) if truly absent.
  bool tf32GemmSupported;        // cc >= 8.0 (Ampere) -- TF32 tensor cores
  bool int8GemmSupported;        // cc >= 7.5 (Turing) -- imma int8 GEMM
  bool int4GemmSupported;        // cc >= 9.0 (Hopper)  -- imma int4 GEMM
  bool dpTensorSupported;        // cc >= 8.0 (Ampere) -- fp64 wmma m8n8k4
  bool int4MmaSupported;         // cc 7.5..8.9 (Turing/Ampere/Ada) + 12.1 (GB10)
                                 // -- s4 mma.sync; dropped on Hopper/datacenter
                                 // Blackwell, re-added on consumer Blackwell GB10.
  bool int8MmaSparseSupported;   // cc >= 8.0 (Ampere+) -- mma.sp.s8 2:4 sparsity
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

  // Load a precompiled fatbin blob into a module (the driver selects the
  // matching cubin for this device or JITs the embedded PTX), then resolve a
  // named kernel.  Caches by blob-data pointer; needs only the CUDA driver at
  // runtime -- no NVRTC, no toolkit headers.  Returns true on success; on
  // failure logs via CLPEAK_VLOG and returns false.
  bool getKernel(const cuda_kernels::Blob &blob,
                 const char *kernelName, CUfunction &fn);

private:
  // moduleCache key = blob-data pointer (each embedded fatbin is a unique
  // array in the generated cpp, so pointer identity is sufficient).
  // Value = loaded module handle.
  std::unordered_map<const void *, CUmodule> moduleCache;
};

// Shared note for one reading of a vector-width sweep.  NOT for the int8-dot
// or WMMA rows: those are independent chains and distinct instructions,
// documented where they are declared.
static inline const char *cudaWidthNote(uint32_t width)
{
  switch (width)
  {
  case 1: return "One value per thread at a time -- the plain, unvectorised case.";
  case 2: return "Two values per thread at a time, packed into one instruction.";
  case 4: return "Four values per thread at a time, as one 4-wide vector.";
  default: return "";
  }
}

// Variant of a single compute-peak kernel.  All variants of one benchmark
// share the same output buffer + dispatch geometry; only the kernel symbol
// (and possibly source file) differs.
struct cuda_compute_variant_t
{
  const char *label;             // column / result metric, e.g. "mp", "mp2"
  const char *kernelName;        // CUDA kernel symbol (extern "C")
  const cuda_kernels::Blob *blob;// embedded fatbin (may be shared by sibling
                                 // variants emitting from one file)
  const char *description;       // what this one reading means (nullptr = undocumented)
};

struct cuda_compute_desc_t
{
  const char *title;             // header line
  const char *resultTag;            // persisted test name
  const char *unit;              // "flops" / "ops"

  // One or two plain-language sentences on what the test measures; travels to
  // logger::TestSpec::description (nullptr = undocumented).
  const char *description;

  // Whether the variants below are interchangeable forms of one measurement
  // (Homogeneous -- a vector-width sweep) or separate measurements sharing a
  // kernel shape (Heterogeneous -- a family of data types).  See
  // include/common/AGENTS.md.
  TestShape   shape;

  // What varies across the readings: "vector width", "data type".  Optional.
  const char *axis;

  // Single-variant fallback (used when variants==nullptr).
  const char *metricLabel;

  // Single-variant path only: what this one reading means, and the unit it is
  // measured in when that differs from the test's.  The unit override is what
  // lets the integer members of a data-type family share the test with the
  // floating-point ones instead of living in a `-int` twin.
  const char *metricDescription;
  const char *metricUnit;
  const char *kernelName;
  const cuda_kernels::Blob *blob;

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


  // Test to write into.  nullptr means "open one from the fields above" --
  // the ordinary case.  A family measured in several descs (every data type
  // its own #ifdef block) passes one scope instead, so the test opens and
  // closes once rather than once per reading.
  logger::TestScope *scope;

  // Optional gates / attributes
  bool        skip;
  const char *skipMsg;
};

class CudaPeak : public Peak
{
public:
  std::vector<int> deviceIndices;  // empty = run all

  CudaPeak();
  ~CudaPeak();

  void applyOptions(const CliOptions &opts) override;
  int runAll() override;

  // Inventory.
  static BackendInventory enumerate();
  static void printInventory(const BackendInventory &inv, std::ostream &os);

  int runComputeSP(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeHP(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeDP(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeMP(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeBF16(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeInt8DP(CudaDevice &dev, benchmark_config_t &cfg);
  int runComputeInt32(CudaDevice &dev, benchmark_config_t &cfg);
  int runGlobalBandwidth(CudaDevice &dev, benchmark_config_t &cfg);
  int runTransferBandwidth(CudaDevice &dev, benchmark_config_t &cfg);
  int runKernelLatency(CudaDevice &dev, benchmark_config_t &cfg);
  int runWmma(CudaDevice &dev, benchmark_config_t &cfg);
  int runCublas(CudaDevice &dev, benchmark_config_t &cfg);
  int runLocalBandwidth(CudaDevice &dev, benchmark_config_t &cfg);
  int runImageBandwidth(CudaDevice &dev, benchmark_config_t &cfg);

private:
  bool initialised;
  std::vector<int> devIndices;
  CUresult m_initResult = CUDA_SUCCESS;
  logger::DeviceScope *currentDeviceScope = nullptr;  // set during runAll

  bool initDriver();

  // Time a kernel batched as `iters` dispatches, where `iters` is calibrated
  // from a one-shot warmup so the timed phase lands at ~targetTimeUs.  Returns
  // mean per-launch time in microseconds.  forcedIters != 0 short-circuits
  // calibration (matches --iters).
  float runKernel(CudaDevice &dev, CUfunction fn,
                  uint32_t gridX, uint32_t blockX,
                  void **args,
                  unsigned int targetTimeUs, unsigned int forcedIters);

  int runComputeKernel(CudaDevice &dev, benchmark_config_t &cfg,
                       const cuda_compute_desc_t &d);
};

// Embedded CUDA kernel fatbins (nvcc-compiled at build time, generated into
// cuda_kernels_generated.cpp by EmbedCudaKernels.cmake).
namespace cuda_kernels {
  struct Blob {
    const unsigned char *data;   // multi-arch fatbin bytes
    unsigned int         len;    // byte count
    const char          *name;   // file stem, e.g. "compute_sp.cu"
  };

  extern const Blob compute_sp;
  extern const Blob compute_hp;
  extern const Blob compute_dp;
  extern const Blob compute_mp;
  extern const Blob compute_bf16;
  extern const Blob compute_int8_dp;
  extern const Blob compute_int32;
  extern const Blob global_bandwidth;
  extern const Blob kernel_latency;
  extern const Blob wmma_fp16;
  extern const Blob wmma_bf16;
  extern const Blob wmma_int8;
  extern const Blob wmma_int8_k32;
  extern const Blob wmma_int8_sparse;
  extern const Blob wmma_fp8_e4m3;
  extern const Blob wmma_fp8_e5m2;
  extern const Blob wmma_fp8_f16;
  extern const Blob wmma_fp8_sparse;
  extern const Blob wmma_fp8_sparse_f16;
  extern const Blob wmma_fp4_e2m1;
  extern const Blob wmma_mxf4_e2m1;
  extern const Blob wmma_nvf4_e2m1;
  extern const Blob wmma_mxf4_sparse;
  extern const Blob wmma_nvf4_sparse;
  extern const Blob wmma_tf32;
  extern const Blob wmma_fp64;
  extern const Blob wmma_int4;
  extern const Blob wmma_fp16_f16;
  extern const Blob local_bandwidth;
  extern const Blob image_bandwidth;
}

#endif // ENABLE_CUDA
#endif // CUDA_PEAK_H
