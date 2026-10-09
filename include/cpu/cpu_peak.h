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
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct CliOptions;

// ---------------------------------------------------------------------------
// One logical CPU the pool runs a worker on, as the OS describes it (Linux /
// Android sysfs and device tree, Windows CPU sets).  The cache fields are the
// instance this CPU reads at each level: its size, and how many of the pool's
// workers read the same instance -- an SMT sibling counts, so two threads on
// one core each own half of its L1.  0 bytes = no such level, or no size for it.
// ---------------------------------------------------------------------------
struct cpu_core_t {
  int      id     = -1;     // OS logical CPU number -- what the worker pins to
  uint64_t rank   = 0;      // higher = faster; equal across a homogeneous chip
  int      maxMHz = 0;      // 0 when the OS does not say
  uint64_t l1dBytes = 0, l2Bytes = 0, l3Bytes = 0;
  int      l1dSharers = 1, l2Sharers = 1, l3Sharers = 1;
};

// ---------------------------------------------------------------------------
// Backend-neutral description of the host CPU, filled by detectCpuInfo().
// Cache sizes drive the cache-bandwidth / memory-latency working-set sizing,
// and the ISA flags gate the advanced compute tests (bf16 dot, int8 VNNI/
// dotprod, AMX).  Unknown fields are left at 0/false.  A cache size is never
// assumed: a level the OS gives no size for stays 0, and the rows that would
// need it skip (cacheSizeGap).
//
// The per-instance cache sizes and the clock describe the ST core -- the
// fastest one, which every single-thread row runs on -- not cpu0, which on a
// phone is a little core.
// ---------------------------------------------------------------------------
struct cpu_device_info_t {
  std::string name   = "Unknown CPU";
  std::string vendor;
  std::string isaName = "scalar";   // widest SIMD the binary was built for

  int logicalCores  = 0;
  int physicalCores = 0;
  int perfCores     = 0;            // P-cores (0 when homogeneous / unknown)
  int effCores      = 0;            // E-cores
  int clockMHz      = 0;            // the ST core's maximum clock

  // The CPUs this process may run on, fastest first: pool worker i pins to
  // cores[i].id, so worker 0 -- and with it every single-thread row -- runs on
  // the fastest core, and the all-core rows get one worker per usable CPU.
  // Empty where threads cannot be pinned (macOS): the scheduler places them.
  std::vector<cpu_core_t> cores;
  std::string stCore;               // names the ST core when the cores differ ("Cortex-X4, cpu7")

  uint64_t l1dCacheBytes  = 0;       // the ST core's L1 data cache
  uint64_t l1dTotalBytes  = 0;       // aggregate L1d across all cores
  uint64_t l2CacheBytes   = 0;       // the L2 instance the ST core reads (per-core or per-cluster)
  uint64_t l2TotalBytes   = 0;       // aggregate L2 across all instances
  uint64_t l3CacheBytes   = 0;       // the L3 instance the ST core reads (per-CCX/CCD on AMD)
  uint64_t l3TotalBytes   = 0;       // aggregate L3 across all instances (= l3CacheBytes on a single-LLC chip)
  // Every distinct instance of each level on the chip, so the header can say
  // "12 MB x 2 + 4 MB" where the instances differ; the totals are their sums.
  std::vector<uint64_t> l1dSizes, l2Sizes, l3Sizes;
  bool l1dUnsized = false;           // the OS lists the level but gives no size for it
  bool l2Unsized  = false;           // (an Android device tree without sizes); its
  bool l3Unsized  = false;           // size field stays 0
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
  bool emulatedX86OnArm = false;    // x86 binary translated on ARM64 (Prism/Rosetta/qemu-user):
                                    // CPUID describes the translator's virtual CPU, not the host
};

// Populate `info` from the host (cpu_device.cpp).
void detectCpuInfo(cpu_device_info_t &info);

// Why a working set cannot be sized for cache level `level` (1 = L1d, 2 = L2,
// 3 = L3), or nullptr when it can: that level's size and every smaller level's
// must be known, since a slice must also stay out of the cache beneath it.  An
// L2 the OS does not list at all is taken as absent below an L3, not unknown.
const char *cacheSizeGap(const cpu_device_info_t &info, int level);

// ---------------------------------------------------------------------------
// Persistent pinned thread pool (thread_pool.cpp).  Workers park on a
// condition variable between jobs so the timed region excludes thread-creation
// cost.  Each worker pins itself to its own core, again at every job: Android
// can drop an affinity when it pauses a core (best-effort; advisory on macOS /
// Apple Silicon).
// ---------------------------------------------------------------------------
class CpuThreadPool {
public:
  // `cpuIds` pins worker i to logical CPU cpuIds[i] instead of the default
  // core i -- the main pool passes the fastest core first, and the SMT-scaling
  // test one worker per physical core.  Must have >= maxThreads entries when
  // non-empty.
  explicit CpuThreadPool(int maxThreads, std::vector<int> cpuIds = {});
  ~CpuThreadPool();

  int maxThreads() const { return nMax; }

  // Run body(tid) on worker threads [0, n) and block until all finish.
  void run(int n, const std::function<void(int)> &body);

  // Workers whose last pin attempt failed, as "cpuN (error)" -- for a
  // --verbose line.  Empty where pinning is advisory.
  std::vector<std::string> pinFailures() const;

private:
  void workerLoop(int tid);
  int  pinTarget(int tid) const;

  int                       nMax = 0;
  std::vector<int>          pinIds;      // empty -> worker i pins to core i
  std::unique_ptr<std::atomic<int>[]> pinError;   // per worker: 0 = pinned
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

  // Which backend this is -- the one place that says so; the registry,
  // the inventory and the device selector all read it from here.
  static constexpr Backend kBackend = Backend::Cpu;
  Backend backend() const override { return kBackend; }
  void applyOptions(const CliOptions &opts) override;
  int  runAll() override;

  static BackendInventory enumerate();

  // Timed launcher: runs body(tid, iters) across nThreads and returns the rate
  // -- outer iterations completed per second, summed over the threads -- or a
  // negative value on failure.  With `weight`, an iteration of thread t counts
  // weight[t] (bytes, when each thread streams its own working set), so the
  // rate comes back in those units.  `body` must loop `iters` times internally
  // so one call covers many iterations.
  //
  // One thread: warmups, a probe, then one pickIters() timed batch.  Several:
  // warmups, a settle that keeps every core busy until a deadline (and
  // measures each thread's rate), then a batch the threads claim from a shared
  // counter about a millisecond of their own work at a time, so a fast core
  // does more of it than a slow one.  The clock stops when the counter runs
  // dry; what counts is the work finished by then, and the share of each slice
  // still running that fell inside that window.
  using Workload = std::function<void(int tid, uint64_t iters)>;
  double runWorkload(int nThreads, const Workload &body,
                     unsigned int targetTimeUsLocal, unsigned int forcedIters,
                     const std::vector<double> *weight = nullptr);

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
  int runCpuMatrix(benchmark_config_t &cfg);
#ifdef __APPLE__
  int runAppleBlas(benchmark_config_t &cfg);   // Accelerate GEMM + BNNS matmul
#endif
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
  int runStoreForward(benchmark_config_t &cfg);
  int runSmtScaling(benchmark_config_t &cfg);

  logger::DeviceScope *currentDeviceScope = nullptr;
  cpu_device_info_t    info;
  CpuThreadPool       *pool = nullptr;

private:
  bool initialised = false;
};

#endif // ENABLE_CPU
#endif // CPU_PEAK_H
