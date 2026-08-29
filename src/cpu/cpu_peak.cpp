#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/options.h>

#include <algorithm>
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
  // The CPU backend ignores --max-time (a GPU-watchdog budget) and uses its
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
  else if (b >= (1ull << 20)) ss << std::setprecision(0) << b / (double)(1ull << 20) << " MB";
  else                        ss << std::setprecision(0) << b / (double)(1ull << 10) << " KB";
  return ss.str();
}

double CpuPeak::runWorkload(int nThreads, const Workload &body,
                            unsigned int targetTimeUsLocal, unsigned int forcedIters)
{
  if (nThreads < 1) nThreads = 1;
  if (pool && nThreads > pool->maxThreads()) nThreads = pool->maxThreads();

  using clock = std::chrono::high_resolution_clock;
  auto usSince = [](clock::time_point a, clock::time_point b) {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count() / 1000.0;
  };

  for (unsigned int w = 0; w < warmupCount; w++)
    pool->run(nThreads, [&](int tid) { body(tid, 1); });

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
  // Single-thread runs need no settling (they are already the low-residency
  // case), so this costs nothing on the ST rows.
  if (nThreads > 1)
  {
    // Scale with the measurement budget: AMD's boost limits move on moving
    // averages measured in hundreds of ms, so a fixed 100 ms was far too
    // short (it recovered only ~1.4% of an 11% error on a 3955WX).  A quarter
    // of the budget, capped at 500 ms, keeps the cost proportional (~10% of
    // total run time) and scales down when the user lowers --max-time-cpu.
    const double settleUs =
        std::min(std::max((double)targetTimeUsLocal / 4.0, 100000.0), 500000.0);
    auto w0 = clock::now();
    uint64_t wIters = 1;
    while (usSince(w0, clock::now()) < settleUs)
    {
      pool->run(nThreads, [&](int tid) { body(tid, wIters); });
      if (wIters < (1ull << 32)) wIters *= 2;   // amortize dispatch overhead
    }
  }

  // Adaptive probe: a single outer iteration of a cheap kernel is dominated by
  // the fixed pool-dispatch overhead (~tens of µs), which would inflate the
  // per-iter estimate and under-size the timed batch.  Grow the probe batch
  // until it runs long enough (>=2 ms) that the dispatch overhead is amortized,
  // then derive an accurate per-iteration time from it.
  double perIterUs;
  if (forcedIters)
  {
    perIterUs = 1.0;  // unused; pickIters short-circuits on forced
  }
  else
  {
    uint64_t probeIters = 1;
    double probeUs;
    for (;;)
    {
      auto p0 = clock::now();
      pool->run(nThreads, [&](int tid) { body(tid, probeIters); });
      probeUs = usSince(p0, clock::now());
      if (probeUs >= 2000.0 || probeIters >= (1ull << 24))
        break;
      probeIters *= 4;
    }
    perIterUs = probeUs / (double)probeIters;
    if (perIterUs <= 0.0) perIterUs = 0.01;
  }

  // No per-dispatch command-buffer limit on the CPU, so allow far more than the
  // GPU default of 10000 — otherwise a cheap kernel (small per-iter time) hits
  // that cap and stops well short of the time budget, finishing in ~100 ms.
  unsigned int iters = pickIters(perIterUs, targetTimeUsLocal, forcedIters,
                                 /*max_iters=*/100000000u);

  auto t0 = clock::now();
  pool->run(nThreads, [&](int tid) { body(tid, iters); });
  double totalUs = usSince(t0, clock::now());

  return totalUs / (double)iters;
}

int CpuPeak::runAll()
{
  detectCpuInfo(info);
  if (!pool)
    pool = new CpuThreadPool(info.logicalCores);

  auto backendScope = log->beginBackend("CPU");

  std::vector<logger::Prop> props;
  props.push_back({"Vendor", info.vendor.empty() ? "Unknown" : info.vendor});
  props.push_back({"ISA",    info.isaName});
  {
    std::string cores = std::to_string(info.logicalCores) + " threads / " +
                        std::to_string(info.physicalCores) + " cores";
    if (info.perfCores > 0 && info.effCores > 0)
      cores += " (" + std::to_string(info.perfCores) + "P+" +
               std::to_string(info.effCores) + "E)";
    props.push_back({"Cores", cores});
  }
  if (info.clockMHz > 0)
    props.push_back({"Clock", std::to_string(info.clockMHz) + " MHz"});
  {
    std::string l1d = fmtBytes(info.l1dTotalBytes);
    if (info.l1dTotalBytes > info.l1dCacheBytes)
      l1d += " (" + fmtBytes(info.l1dCacheBytes) + " x " +
             std::to_string(info.l1dTotalBytes / info.l1dCacheBytes) + ")";
    props.push_back({"L1d", l1d});
  }
  {
    // The "x N" breakdown only holds when every instance is the same size.
    // Apple's clusters are not: an M1 Pro is 12 MB x 2 (P) + 4 MB (E) = 28 MB,
    // and "28 MB (12 MB x 2)" would be a wrong reading of a right total.
    std::string l2 = fmtBytes(info.l2TotalBytes);
    if (info.l2TotalBytes > info.l2CacheBytes &&
        info.l2TotalBytes % info.l2CacheBytes == 0)
      l2 += " (" + fmtBytes(info.l2CacheBytes) + " x " +
            std::to_string(info.l2TotalBytes / info.l2CacheBytes) + ")";
    props.push_back({"L2", l2});
  }
  // Show aggregate L3; note the per-instance size on multi-LLC chips (AMD CCX).
  // Omitted entirely on the many CPUs that have no L3 at all (Apple Silicon,
  // Snapdragon X, most phone SoCs) — see the fallbacks in cpu_device.cpp.
  if (info.l3TotalBytes)
  {
    std::string l3 = fmtBytes(info.l3TotalBytes);
    if (info.l3TotalBytes > info.l3CacheBytes &&
        info.l3TotalBytes % info.l3CacheBytes == 0)
      l3 += " (" + fmtBytes(info.l3CacheBytes) + " x " +
            std::to_string(info.l3TotalBytes / info.l3CacheBytes) + ")";
    props.push_back({"L3", l3});
  }
  if (info.totalMemBytes)
    props.push_back({"RAM", fmtBytes(info.totalMemBytes)});

  auto deviceScope = backendScope.beginDevice({
    info.name, "", "", props, -1, 0});
  currentDeviceScope = &deviceScope;

  benchmark_config_t cfg = benchmark_config_t::forDevice(DeviceType::Cpu);
  cfg.targetTimeUs = targetTimeUs;
  if (forceIters)
    cfg.kernelLatencyIters = specifiedIters;

  // ---- FP compute ----
  if (isAllowed(Benchmark::ComputeSP))   runComputeSP(cfg);
  if (isAllowed(Benchmark::ComputeHP))   runComputeHP(cfg);
  if (isAllowed(Benchmark::ComputeDP))   runComputeDP(cfg);
  if (isAllowed(Benchmark::ComputeMP))   runComputeMP(cfg);
  if (isAllowed(Benchmark::ComputeBF16)) runComputeBF16(cfg);
  if (isAllowed(Benchmark::ComputeFP8DP)) runComputeFP8DP(cfg);
  if (isAllowed(Benchmark::ComputeDivSqrt)) runComputeDivSqrt(cfg);
  if (isAllowedAs(Benchmark::Amx, Category::FpCompute))
    runCpuMatrix(cfg, Category::FpCompute);
#ifdef __APPLE__
  if (isAllowed(Benchmark::AppleBlas)) runAppleBlas(cfg);
#endif
  if (isAllowed(Benchmark::SmtScaling)) runSmtScaling(cfg);

  // ---- INT compute ----
  if (isAllowed(Benchmark::ComputeInt))     runComputeInt32(cfg);
  if (isAllowed(Benchmark::ComputeInt8DP))  runComputeInt8DP(cfg);
  if (isAllowed(Benchmark::ComputeInt16DP)) runComputeInt16DP(cfg);
  if (isAllowed(Benchmark::ComputeIntDiv))  runComputeIntDiv(cfg);
  if (isAllowedAs(Benchmark::Amx, Category::IntCompute))
    runCpuMatrix(cfg, Category::IntCompute);

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
  inv.backend = "CPU";

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

void CpuPeak::printInventory(const BackendInventory &b, std::ostream &os)
{
  os << "\n=== CPU backend ===\n";
  if (!b.available)
  {
    os << "CPU: no host CPU detected\n";
    return;
  }
  for (const auto &plat : b.platforms)
    for (const auto &d : plat.devices)
    {
      os << "  CPU Device " << d.index << ": " << d.name;
      if (d.numComputeUnits) os << " [" << d.numComputeUnits << " threads]";
      os << "\n";
    }
}

#endif // ENABLE_CPU
