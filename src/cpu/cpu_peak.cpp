#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/options.h>

#include <chrono>
#include <cstdio>
#include <ostream>
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

// Human-readable byte size for the device property block.
static std::string fmtBytes(uint64_t b)
{
  char buf[64];
  if (b >= (1ull << 30))      std::snprintf(buf, sizeof(buf), "%.1f GB", b / (double)(1ull << 30));
  else if (b >= (1ull << 20)) std::snprintf(buf, sizeof(buf), "%.0f MB", b / (double)(1ull << 20));
  else                        std::snprintf(buf, sizeof(buf), "%.0f KB", b / (double)(1ull << 10));
  return buf;
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
    std::string l2 = fmtBytes(info.l2TotalBytes);
    if (info.l2TotalBytes > info.l2CacheBytes)
      l2 += " (" + fmtBytes(info.l2CacheBytes) + " x " +
            std::to_string(info.l2TotalBytes / info.l2CacheBytes) + ")";
    props.push_back({"L2", l2});
  }
  {
    // Show aggregate L3; note the per-instance size on multi-LLC chips (AMD CCX).
    std::string l3 = fmtBytes(info.l3TotalBytes);
    if (info.l3TotalBytes > info.l3CacheBytes)
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
