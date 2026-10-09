#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/options.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <locale>
#include <ostream>
#include <sstream>
#include <string>

CpuPeak::CpuPeak() {}
CpuPeak::~CpuPeak()
{
  delete pool;
  pool = nullptr;
}

void CpuPeak::applyOptions(const CliOptions &opts)
{
  Peak::applyOptions(opts);
  // The CPU backend ignores --max-time-gpu (a GPU-watchdog budget) and uses its
  // own, longer --max-time-cpu budget so the timed phases don't finish in a
  // few ms and fluctuate with turbo / scheduler jitter.
  targetTimeUs = opts.targetTimeUsCpu;
}

// Human-readable byte size for the device property block.  Streams pinned to
// the classic locale, not printf: this string is persisted in the dump files
// and the GUI runs with the host toolkit's locale set, which would otherwise
// write "8,0 GB" there but "8.0 GB" from the CLI on the same machine.
static std::string fmtBytes(uint64_t b)
{
  std::ostringstream ss;
  ss.imbue(std::locale::classic());
  ss << std::fixed;
  if (b >= (1ull << 30))      ss << std::setprecision(1) << b / (double)(1ull << 30) << " GB";
  else if (b >= (1ull << 20)) ss << std::setprecision(b % (1ull << 20) ? 1 : 0)
                                 << b / (double)(1ull << 20) << " MB";
  else                        ss << std::setprecision(0) << b / (double)(1ull << 10) << " KB";
  return ss.str();
}

// A cache level for the header: the total, then each instance size with its
// count, largest first -- "1.1 MB (128 KB x 8 + 64 KB x 2)".  "x N" of one
// size would misdescribe every chip whose cores differ, as the M1 Pro's L1d
// line once did with "128 KB x 10".  `total` stands alone where the OS listed
// no instances.
static std::string fmtCacheLevel(const std::vector<uint64_t> &sizes, uint64_t total)
{
  if (sizes.empty())
    return fmtBytes(total);
  std::vector<std::pair<uint64_t, int>> bySize;   // largest first
  uint64_t sum = 0;
  for (uint64_t sz : sizes)
  {
    sum += sz;
    auto it = std::find_if(bySize.begin(), bySize.end(),
                           [sz](const std::pair<uint64_t, int> &p) { return p.first == sz; });
    if (it != bySize.end()) it->second++;
    else bySize.push_back({sz, 1});
  }
  std::sort(bySize.begin(), bySize.end(),
            [](const std::pair<uint64_t, int> &a, const std::pair<uint64_t, int> &b) {
              return a.first > b.first;
            });
  std::string out = fmtBytes(sum);
  if (sizes.size() < 2)
    return out;
  out += " (";
  for (size_t i = 0; i < bySize.size(); i++)
  {
    if (i) out += " + ";
    out += fmtBytes(bySize[i].first);
    if (bySize[i].second > 1) out += " x " + std::to_string(bySize[i].second);
  }
  return out + ")";
}

double CpuPeak::runWorkload(int nThreads, const Workload &body,
                            unsigned int targetTimeUsLocal, unsigned int forcedIters,
                            const std::vector<double> *weight)
{
  if (nThreads < 1) nThreads = 1;
  if (pool && nThreads > pool->maxThreads()) nThreads = pool->maxThreads();

  using clock = std::chrono::high_resolution_clock;
  auto usSince = [](clock::time_point a, clock::time_point b) {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count() / 1000.0;
  };
  auto weightOf = [weight](int tid) {
    return weight && (size_t)tid < weight->size() ? (*weight)[(size_t)tid] : 1.0;
  };

  for (unsigned int w = 0; w < warmupCount; w++)
    pool->run(nThreads, [&](int tid) { body(tid, 1); });

  if (nThreads == 1)
  {
    // Adaptive probe: a single outer iteration of a cheap kernel is dominated
    // by the fixed pool-dispatch overhead (~tens of µs), which would inflate
    // the per-iter estimate and under-size the timed batch.  Grow the probe
    // batch until it runs long enough (>=2 ms) that the dispatch overhead is
    // amortized, then derive an accurate per-iteration time from it.
    double perIterUs = 1.0;  // unused when forced; pickIters short-circuits
    if (!forcedIters)
    {
      uint64_t probeIters = 1;
      double probeUs;
      for (;;)
      {
        auto p0 = clock::now();
        pool->run(1, [&](int tid) { body(tid, probeIters); });
        probeUs = usSince(p0, clock::now());
        if (probeUs >= 2000.0 || probeIters >= (1ull << 24))
          break;
        probeIters *= 4;
      }
      perIterUs = probeUs / (double)probeIters;
      if (perIterUs <= 0.0) perIterUs = 0.01;
    }

    // No per-dispatch command-buffer limit on the CPU, so allow far more than
    // the GPU default of 10000 — otherwise a cheap kernel (small per-iter
    // time) hits that cap and stops well short of the time budget, finishing
    // in ~100 ms.
    unsigned int iters = pickIters(perIterUs, targetTimeUsLocal, forcedIters,
                                   /*max_iters=*/100000000u);

    auto t0 = clock::now();
    pool->run(1, [&](int tid) { body(tid, iters); });
    double totalUs = usSince(t0, clock::now());
    return totalUs > 0.0 ? (double)iters * weightOf(0) / (totalUs * 1e-6) : -1.0;
  }

  // ---- Several threads ----------------------------------------------------
  // An equal share each, timed to the slowest thread, measures the slowest
  // core times the thread count.  On a heterogeneous chip that is the whole
  // row: a Galaxy S24's fp32 MT read 144 GFLOPS -- eight threads at the pace
  // of its two A520s, which share one vector unit -- against ~390 for the
  // cores together.  And one thread the OS holds up drags every other one
  // down with it.  So each thread claims a slice of the work at a time, sized
  // to about `chunkUs` of its own core's time, until none is left: a fast
  // core takes more slices than a slow one, and a stalled thread just takes
  // fewer.
  const double chunkUs =
      std::min(std::max((double)targetTimeUsLocal / 2000.0, 100.0), 1000.0);

  // One cache line per thread, so no thread's bookkeeping shares a line with
  // another's: the shared claim counter below is the only line they contend.
  struct alignas(64) Slot
  {
    uint64_t chunk = 1;                 // iterations per claim
    uint64_t measIters = 0, allIters = 0;
    double   measUs = 0.0, allUs = 0.0;
    std::atomic<uint64_t> done{0};      // iterations finished in the timed batch
    uint64_t lastK = 0;                 // the thread's last slice, and when it ran
    double   lastStartUs = 0.0, lastEndUs = 0.0;
  };
  std::vector<Slot> slot((size_t)nThreads);

  // Settle the package into the clock/power state that belongs to THIS thread
  // count before timing.  On parts whose boost tracks core residency this is
  // not optional: an MT measurement taken straight after the (2 s, full-boost)
  // single-thread phase spends part of its window still limited by the
  // single-core boost state.  Observed on a Threadripper PRO 3955WX, where the
  // 32-thread fp32 row read 1874 GFLOPS while the *same kernel* at 32 threads
  // measured 2089 in the SMT test -- which runs after an all-core phase -- and
  // 16 threads alone measured 1967.  A 32-thread result below the 16-thread
  // one cannot be caused by SMT, so the short-warmup row was the wrong number.
  // The `body(tid,1)` warmup above is microseconds and cannot do this job.
  //
  // Scale with the measurement budget: AMD's boost limits move on moving
  // averages measured in hundreds of ms, so a fixed 100 ms was far too short
  // (it recovered only ~1.4% of an 11% error on a 3955WX).  A quarter of the
  // budget, capped at 500 ms, keeps the cost proportional (~10% of total run
  // time) and scales down when the user lowers --max-time-cpu.
  //
  // Every thread runs until the deadline rather than through a fixed share, so
  // no core idles at a barrier while the package settles, and each one sizes
  // its chunk and reports its rate over the second half: that is the probe.
  const double settleUs =
      std::min(std::max((double)targetTimeUsLocal / 4.0, 100000.0), 500000.0);
  {
    const auto s0 = clock::now();
    const auto deadline = s0 + std::chrono::microseconds((int64_t)settleUs);
    const auto half     = s0 + std::chrono::microseconds((int64_t)(settleUs / 2.0));
    pool->run(nThreads, [&](int tid) {
      Slot &s = slot[(size_t)tid];
      for (;;)
      {
        const auto c0 = clock::now();
        if (c0 >= deadline)
          break;
        body(tid, s.chunk);
        const double us = usSince(c0, clock::now());
        s.allIters += s.chunk;
        s.allUs    += us;
        if (c0 >= half)
        {
          s.measIters += s.chunk;
          s.measUs    += us;
        }
        if (us < chunkUs / 2.0 && s.chunk < (1ull << 32))
          s.chunk *= 2;
      }
    });
  }

  // Total work for the timed batch: the threads' summed rate over the budget.
  uint64_t total;
  if (forcedIters)
  {
    total = (uint64_t)forcedIters * (uint64_t)nThreads;
  }
  else
  {
    double perUs = 0.0;   // iterations per µs, all threads together
    for (const Slot &s : slot)
    {
      if (s.measUs > 0.0)     perUs += (double)s.measIters / s.measUs;
      else if (s.allUs > 0.0) perUs += (double)s.allIters / s.allUs;
    }
    const double want = perUs * (double)(targetTimeUsLocal ? targetTimeUsLocal : 5000000u);
    const double cap  = 1e8 * (double)nThreads;
    total = (uint64_t)std::min(std::max(want, (double)nThreads), cap);
  }

  // The timed batch.  Unforced, the clock stops the moment the counter runs
  // dry: after that point some cores have nothing left to do, and a thread the
  // OS parked mid-slice can no longer stretch the window everyone else already
  // finished in.  The work counted is what finished by then, plus the share of
  // each slice still running that fell inside the window -- prorated by time,
  // since leaving those slices out reads low by half a slice per thread, and a
  // DRAM pass can be tens of milliseconds.  A forced batch (--iters) is a
  // fixed amount of work, so it is timed to the end, one iteration a claim.
  std::atomic<uint64_t> next{0};
  std::atomic<bool> dry{false};
  double dryUs = -1.0;
  double dryWork = 0.0;
  auto finished = [&]() {
    double w = 0.0;
    for (int t = 0; t < nThreads; t++)
      w += (double)slot[(size_t)t].done.load(std::memory_order_acquire) * weightOf(t);
    return w;
  };
  const auto t0 = clock::now();
  pool->run(nThreads, [&](int tid) {
    Slot &s = slot[(size_t)tid];
    const uint64_t claim = forcedIters ? 1 : s.chunk;
    for (;;)
    {
      const uint64_t start = next.fetch_add(claim, std::memory_order_relaxed);
      if (start >= total)
      {
        // The first thread to find nothing left reads what is finished, then
        // the clock -- in that order, so a slice that ends in between is left
        // out rather than counted against a window it fell outside.
        if (!forcedIters && !dry.exchange(true))
        {
          dryWork = finished();
          dryUs   = usSince(t0, clock::now());
        }
        break;
      }
      const uint64_t k = std::min(claim, total - start);
      const double b = usSince(t0, clock::now());
      body(tid, k);
      const double e = usSince(t0, clock::now());
      s.done.fetch_add(k, std::memory_order_release);
      s.lastK       = k;
      s.lastStartUs = b;
      s.lastEndUs   = e;
    }
  });
  const double wallUs = usSince(t0, clock::now());

  if (!forcedIters && dryUs > 0.0 && dryWork > 0.0)
  {
    double work = dryWork;
    for (int t = 0; t < nThreads; t++)
    {
      const Slot &s = slot[(size_t)t];
      if (s.lastK && s.lastStartUs < dryUs && s.lastEndUs > dryUs)
        work += (double)s.lastK * weightOf(t) * (dryUs - s.lastStartUs) /
                (s.lastEndUs - s.lastStartUs);
    }
    return work / (dryUs * 1e-6);
  }
  // Forced, or a batch too small for anything to finish before the counter
  // ran dry: everything, timed to the end.
  return wallUs > 0.0 ? finished() / (wallUs * 1e-6) : -1.0;
}

int CpuPeak::runAll()
{
  // The one CPU is device 0; a --devices list naming another index skips it.
  if (!isDeviceSelected(0))
    return 0;

  detectCpuInfo(info);
  if (!pool)
  {
    // Fastest core first: worker 0 runs every single-thread row.
    std::vector<int> ids;
    for (const cpu_core_t &c : info.cores)
      ids.push_back(c.id);
    pool = new CpuThreadPool(ids.empty() ? info.logicalCores : (int)ids.size(), ids);
  }
  if (clpeak::verboseEnabled())
  {
    std::string order;
    for (const cpu_core_t &c : info.cores)
      order += " cpu" + std::to_string(c.id);
    if (!order.empty())
      CLPEAK_VLOG("[cpu] workers pinned fastest first:%s\n", order.c_str());
    // Every worker pins on its first job; say which ones the OS refused.
    pool->run(pool->maxThreads(), [](int) {});
    for (const std::string &f : pool->pinFailures())
      CLPEAK_VLOG("[cpu] could not pin a worker to %s\n", f.c_str());
    // Each size the OS withheld; a level it lists nowhere (no L3) is not one.
    const bool withheld[3] = {!info.l1dCacheBytes, info.l2Unsized, info.l3Unsized};
    for (int lvl = 1; lvl <= 3; lvl++)
      if (withheld[lvl - 1])
        CLPEAK_VLOG("[cpu] %s; the cache rows that need it skip\n", cacheSizeGap(info, lvl));
  }

  auto backendScope = log->beginBackend("CPU");

  std::vector<logger::Prop> props;
  props.push_back({"Vendor", info.vendor.empty() ? "Unknown" : info.vendor});
  props.push_back({"ISA",    info.isaName});
  // An x86 binary translated on ARM64 (Prism/Rosetta/qemu-user) reports the
  // translator's virtual CPU, so name the translation: every row below is
  // translated code on an ARM core, not the x86 part the header describes.
  if (info.emulatedX86OnArm)
    props.push_back({"Emulation", "x86 translated on ARM64"});
  {
    std::string cores = std::to_string(info.logicalCores) + " threads / " +
                        std::to_string(info.physicalCores) + " cores";
    if (info.perfCores > 0 && info.effCores > 0)
      cores += " (" + std::to_string(info.perfCores) + "P+" +
               std::to_string(info.effCores) + "E)";
    // An Android app's cpuset, taskset or a container: the all-core rows run
    // on what this process may use, not on the whole chip.
    if (!info.cores.empty() && (int)info.cores.size() < info.logicalCores)
      cores += ", " + std::to_string(info.cores.size()) + " usable";
    props.push_back({"Cores", cores});
  }
  // Which core the single-thread rows ran on, where the cores differ.  The
  // clock and the per-core cache sizes below are that core's.
  if (!info.stCore.empty())
    props.push_back({"ST core", info.stCore});
  if (info.clockMHz > 0)
    props.push_back({"Clock", std::to_string(info.clockMHz) + " MHz"});
  // A level the OS gave no size for is left out, as are the rows that need it.
  if (info.l1dTotalBytes)
    props.push_back({"L1d", fmtCacheLevel(info.l1dSizes, info.l1dTotalBytes)});
  if (info.l2TotalBytes)
    props.push_back({"L2", fmtCacheLevel(info.l2Sizes, info.l2TotalBytes)});
  // Omitted entirely on the many CPUs that have no L3 at all (Apple Silicon,
  // Snapdragon X, most phone SoCs).
  if (info.l3TotalBytes)
    props.push_back({"L3", fmtCacheLevel(info.l3Sizes, info.l3TotalBytes)});
  if (info.totalMemBytes)
    props.push_back({"RAM", fmtBytes(info.totalMemBytes)});

  auto deviceScope = backendScope.beginDevice({
    info.name, "", "", props, -1, 0, DeviceType::Cpu});
  currentDeviceScope = &deviceScope;

  benchmark_config_t cfg = benchmark_config_t::forDevice(DeviceType::Cpu);
  cfg.targetTimeUs = targetTimeUs;
  if (forceIters)
    cfg.kernelLatencyIters = specifiedIters;

  // ---- Compute (GFLOPS/TFLOPS + GOPS/TOPS) ----
  if (isAllowed(Benchmark::ComputeSP))   runComputeSP(cfg);
  if (isAllowed(Benchmark::ComputeHP))   runComputeHP(cfg);
  if (isAllowed(Benchmark::ComputeDP))   runComputeDP(cfg);
  if (isAllowed(Benchmark::ComputeMP))   runComputeMP(cfg);
  if (isAllowed(Benchmark::ComputeBF16)) runComputeBF16(cfg);
  if (isAllowed(Benchmark::ComputeFP8DP)) runComputeFP8DP(cfg);
  if (isAllowed(Benchmark::ComputeDivSqrt)) runComputeDivSqrt(cfg);
  if (isAllowed(Benchmark::ComputeInt))     runComputeInt32(cfg);
  if (isAllowed(Benchmark::ComputeInt8DP))  runComputeInt8DP(cfg);
  if (isAllowed(Benchmark::ComputeInt16DP)) runComputeInt16DP(cfg);
  if (isAllowed(Benchmark::ComputeIntDiv))  runComputeIntDiv(cfg);
  if (isAllowed(Benchmark::MatrixCompute)) runCpuMatrix(cfg);
#ifdef __APPLE__
  if (isAllowed(Benchmark::Gemm))          runAppleBlas(cfg);
#endif
  if (isAllowed(Benchmark::SmtScaling)) runSmtScaling(cfg);

  // ---- Crypto (dedicated AES/SHA/CRC silicon; GB/s) ----
  if (isAllowed(Benchmark::CryptoAes))    runCryptoAes(cfg);
  if (isAllowed(Benchmark::CryptoSha256)) runCryptoSha256(cfg);
  if (isAllowed(Benchmark::CryptoSha512)) runCryptoSha512(cfg);
  if (isAllowed(Benchmark::CryptoCrc32c)) runCryptoCrc32c(cfg);

  // ---- String (SIMD text processing; GB/s over L1-resident buffers) ----
  if (isAllowed(Benchmark::StringScan))   runStringScan(cfg);
  if (isAllowed(Benchmark::Utf8Validate)) runUtf8Validate(cfg);

  // ---- Bandwidth ----
  // No TransferBW: on a CPU there is no host<->device bus, so a libc memcpy
  // measures the same DRAM path as the STREAM copy above (redundant).
  if (isAllowed(Benchmark::GlobalBW))       runDramBandwidth(cfg);
  if (isAllowed(Benchmark::CacheBandwidth)) runCacheBandwidth(cfg);

  // ---- Latency ----
  if (isAllowed(Benchmark::MemoryLatency)) runMemoryLatency(cfg);
  if (isAllowed(Benchmark::Atomics))       runAtomics(cfg);
  if (isAllowed(Benchmark::BranchPenalty)) runBranchPenalty(cfg);
  if (isAllowed(Benchmark::StoreForward))  runStoreForward(cfg);

  currentDeviceScope = nullptr;
  return 0;
}

BackendInventory CpuPeak::enumerate()
{
  BackendInventory inv;
  inv.id = kBackend;

  cpu_device_info_t info;
  detectCpuInfo(info);

  inv.available = true;
  InventoryPlatform plat;
  plat.index = 0;
  plat.name  = "Native CPU";

  InventoryDevice d;
  d.index           = 0;
  d.name            = info.name;
  d.typeStr         = "CPU";
  d.numComputeUnits = (unsigned)info.logicalCores;
  d.maxClockMHz     = (unsigned)info.clockMHz;
  d.globalMemBytes  = info.totalMemBytes;
  plat.devices.push_back(std::move(d));

  inv.platforms.push_back(std::move(plat));
  return inv;
}

#endif // ENABLE_CPU
