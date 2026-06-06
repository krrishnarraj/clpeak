#ifndef CPU_PEAK_H
#define CPU_PEAK_H

#ifdef ENABLE_CPU

#include <common/common.h>
#include <common/inventory.h>
#include <common/logger.h>
#include <common/peak.h>

#include <atomic>
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

  uint64_t l1dCacheBytes = 0;       // per-core L1 data cache
  uint64_t l2CacheBytes  = 0;       // per-core (or per-cluster) L2
  uint64_t l3CacheBytes  = 0;       // shared LLC
  uint64_t totalMemBytes = 0;

  // ISA capability flags (best-effort runtime detection).
  bool hasFMA    = false;           // x86 FMA3
  bool hasAVX2   = false;
  bool hasAVX512 = false;
  bool hasNEON   = false;
  bool hasFP16   = false;           // native fp16 arithmetic (AVX512-FP16 / ARM FEAT_FP16)
  bool hasFP16FML = false;          // widening fp16xfp16 -> fp32 FMLA (ARM FEAT_FP16FML)
  bool hasBF16   = false;           // bf16 dot (AVX512-BF16 / ARM bfdot)
  bool hasInt8DP = false;           // int8 dot (AVX512-VNNI / AVX-VNNI / ARM dotprod)
  bool hasAMX    = false;           // x86 AMX tile matmul (int8 + bf16)
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
  int runComputeInt32(benchmark_config_t &cfg);
  int runComputeInt8DP(benchmark_config_t &cfg);
  int runCpuMatrix(benchmark_config_t &cfg, Category category);
  int runDramBandwidth(benchmark_config_t &cfg);
  int runCacheBandwidth(benchmark_config_t &cfg);
  int runMemcpyBandwidth(benchmark_config_t &cfg);
  int runAtomicThroughput(benchmark_config_t &cfg);
  int runMemoryLatency(benchmark_config_t &cfg);
  int runThreadLatency(benchmark_config_t &cfg);

  logger::DeviceScope *currentDeviceScope = nullptr;
  cpu_device_info_t    info;
  CpuThreadPool       *pool = nullptr;

private:
  bool initialised = false;
};

#endif // ENABLE_CPU
#endif // CPU_PEAK_H
