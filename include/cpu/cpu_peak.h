#ifndef CPU_PEAK_H
#define CPU_PEAK_H

#ifdef ENABLE_CPU

#include <common/common.h>
#include <common/inventory.h>
#include <common/logger.h>
#include <common/peak.h>

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct CliOptions;

// ---------------------------------------------------------------------------
// Backend-neutral description of the host CPU, filled by detectCpuInfo().
// Cache sizes drive the cache-bandwidth / memory-latency working-set sizing,
// and the ISA flags gate the advanced compute tests (bf16 dot, int8 VNNI/
// dotprod, AMX).  Unknown fields are left at 0/false and the consumer falls
// back to sane defaults.
// ---------------------------------------------------------------------------
struct cpu_device_info_t {
  std::string name   = "Unknown CPU";
  std::string vendor;
  std::string isaName = "scalar";   // widest SIMD the binary was built for

  int logicalCores  = 0;
  int physicalCores = 0;
  int perfCores     = 0;            // P-cores (0 when homogeneous / unknown)
  int effCores      = 0;            // E-cores
  int clockMHz      = 0;

  uint64_t l1dCacheBytes  = 0;       // per-core L1 data cache
  uint64_t l1dTotalBytes  = 0;       // aggregate L1d across all cores
  uint64_t l2CacheBytes   = 0;       // per-core (or per-cluster) L2
  uint64_t l2TotalBytes   = 0;       // aggregate L2 across all instances
  uint64_t l3CacheBytes   = 0;       // one L3 instance (per-CCX/CCD on AMD)
  uint64_t l3TotalBytes   = 0;       // aggregate L3 across all instances (= l3CacheBytes on a single-LLC chip)
  uint64_t totalMemBytes = 0;

  // ISA capability flags (best-effort runtime detection).
  bool hasFMA    = false;           // x86 FMA3
  bool hasAVX2   = false;
  bool hasAVX512 = false;
  bool hasNEON   = false;
  bool hasFP16   = false;           // native fp16 arithmetic (AVX512-FP16 / ARM FEAT_FP16)
  bool hasFP16FML = false;          // widening fp16xfp16 -> fp32 FMLA (ARM FEAT_FP16FML)
  bool hasBF16   = false;           // bf16 dot (AVX512-BF16 / ARM bfdot / SVE bfdot)
  bool hasInt8DP = false;           // int8 dot (AVX512-VNNI / AVX-VNNI / ARM dotprod / SVE sdot)
  bool hasAVXVNNI = false;          // 256-bit AVX-VNNI int8 dot (no AVX-512 needed)
  bool hasAMX    = false;           // x86 AMX tile matmul (int8 + bf16)
  bool hasSVE    = false;           // ARM SVE (vector-length-agnostic)
  bool hasSVE2   = false;           // ARM SVE2
  int  sveVLBytes = 0;              // active SVE vector length in bytes (0 if no SVE)
  bool hasSME    = false;           // ARM SME (streaming matrix engine; Apple M4+, Oryon Gen 3)
  bool hasSME2   = false;           // ARM SME2
  int  smeSVLBytes = 0;             // active SME streaming vector length in bytes (0 if no SME)
};

// Populate `info` from the host (cpu_device.cpp).
void detectCpuInfo(cpu_device_info_t &info);

// ---------------------------------------------------------------------------
// Persistent pinned thread pool (thread_pool.cpp).  Workers park on a
// condition variable between jobs so the timed region excludes thread-creation
// cost.  Each worker pins itself to its own core index (best-effort; advisory
// on macOS / Apple Silicon).
// ---------------------------------------------------------------------------
class CpuThreadPool {
public:
  explicit CpuThreadPool(int maxThreads);
  ~CpuThreadPool();

  int maxThreads() const { return nMax; }

  // Run body(tid) on worker threads [0, n) and block until all finish.
  void run(int n, const std::function<void(int)> &body);

private:
  void workerLoop(int tid);

  int                       nMax = 0;
  std::vector<std::thread>  workers;
  std::mutex                mtx;
  std::condition_variable   cvStart;
  std::condition_variable   cvDone;
  const std::function<void(int)> *job = nullptr;
  int                       activeCount = 0;     // workers that should run this job
  int                       remaining   = 0;     // workers still executing
  uint64_t                  generation  = 0;     // bumped each dispatch
  bool                      stop = false;
};

class CpuPeak : public Peak {
public:
  CpuPeak();
  ~CpuPeak();

  void applyOptions(const CliOptions &opts) override;
  int  runAll() override;

  static BackendInventory enumerate();
  static void printInventory(const BackendInventory &inv, std::ostream &os);

  // Timed launcher: runs body(tid, iters) across nThreads (warmups + one probe
  // + pickIters() timed batch) and returns the mean wall-clock microseconds per
  // outer iteration, or a negative value on failure.  `body` must loop `iters`
  // times internally so a single dispatch covers the whole timed batch (one
  // barrier per batch, not per iteration).
  using Workload = std::function<void(int tid, uint64_t iters)>;
  double runWorkload(int nThreads, const Workload &body,
                     unsigned int targetTimeUsLocal, unsigned int forcedIters);

  // ---- benchmarks ----
  int runComputeSP(benchmark_config_t &cfg);
  int runComputeDP(benchmark_config_t &cfg);
  int runComputeHP(benchmark_config_t &cfg);
  int runComputeBF16(benchmark_config_t &cfg);
  int runComputeMP(benchmark_config_t &cfg);
  int runComputeFP8DP(benchmark_config_t &cfg);
  int runComputeDivSqrt(benchmark_config_t &cfg);
  int runComputeInt32(benchmark_config_t &cfg);
  int runComputeInt8DP(benchmark_config_t &cfg);
  int runComputeInt16DP(benchmark_config_t &cfg);
  int runComputeIntDiv(benchmark_config_t &cfg);
  int runCpuMatrix(benchmark_config_t &cfg, Category category);
  int runCryptoAes(benchmark_config_t &cfg);
  int runCryptoSha256(benchmark_config_t &cfg);
  int runCryptoSha512(benchmark_config_t &cfg);
  int runCryptoCrc32c(benchmark_config_t &cfg);
  int runStringScan(benchmark_config_t &cfg);
  int runUtf8Validate(benchmark_config_t &cfg);
  int runDramBandwidth(benchmark_config_t &cfg);
  int runCacheBandwidth(benchmark_config_t &cfg);
  int runMemoryLatency(benchmark_config_t &cfg);
  int runAtomics(benchmark_config_t &cfg);
  int runBranchPenalty(benchmark_config_t &cfg);

  logger::DeviceScope *currentDeviceScope = nullptr;
  cpu_device_info_t    info;
  CpuThreadPool       *pool = nullptr;

private:
  bool initialised = false;
};

#endif // ENABLE_CPU
#endif // CPU_PEAK_H
