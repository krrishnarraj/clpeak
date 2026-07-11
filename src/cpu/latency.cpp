#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/result_store.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <unistd.h>
#endif

// Full unroll for the K parallel chase cursors (kept in registers); local
// definition since this file doesn't pull in cpu_simd.h.
#if defined(__clang__)
#define CPU_UNROLL_FULL_LAT _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define CPU_UNROLL_FULL_LAT _Pragma("GCC unroll 32")
#else
#define CPU_UNROLL_FULL_LAT
#endif

// Pointer-chase latency over a random cycle of cache-line-spaced nodes.  Each
// node occupies one 64-byte line (16 uint32) and stores the index of the next
// node, so each hop is a dependent load that defeats stride prefetchers.
// Working set = nLines * 64 bytes, sized per cache level.
static double chaseLatencyNs(uint64_t levelBytes)
{
  size_t nLines = (size_t)std::max<uint64_t>(levelBytes / 64, 64);
  std::vector<uint32_t> buf(nLines * 16, 0);

  std::vector<uint32_t> order(nLines);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937 rng(0xC0FFEE);
  for (size_t i = nLines - 1; i > 0; i--)              // Sattolo: single cycle
  {
    size_t j = rng() % i;
    std::swap(order[i], order[j]);
  }
  for (size_t i = 0; i < nLines; i++)
    buf[(size_t)order[i] * 16] = order[(i + 1) % nLines] * 16;

  using clock = std::chrono::steady_clock;
  auto nsBetween = [](clock::time_point a, clock::time_point b) {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
  };

  // Probe to size the timed walk to ~300 ms.
  uint32_t cur = 0;
  uint64_t probe = 1ull << 20;
  auto p0 = clock::now();
  for (uint64_t k = 0; k < probe; k++) cur = buf[cur];
  double nsPer = nsBetween(p0, clock::now()) / (double)probe;
  if (nsPer < 0.1) nsPer = 0.1;

  uint64_t steps = (uint64_t)(300e6 / nsPer);
  steps = std::min<uint64_t>(std::max<uint64_t>(steps, 1ull << 20), 2000000000ull);

  double best = 1e30;
  for (int rep = 0; rep < 3; rep++)
  {
    auto t0 = clock::now();
    for (uint64_t k = 0; k < steps; k++) cur = buf[cur];
    double ns = nsBetween(t0, clock::now()) / (double)steps;
    best = std::min(best, ns);
  }
  volatile uint32_t keep = cur; (void)keep;
  return best;
}

// Memory-level parallelism: K independent dependent-load chains walking the
// same random cycle from staggered start points.  One chain measures pure
// latency (the row above); K chains measure how many misses the core can keep
// in flight -- the effective ns/access at depth K is latency / min(K, MLP).
// The DRAM-latency-to-x32 ratio is the MLP factor (Apple big cores ~16+
// outstanding misses, older x86 ~10), a number neither the latency nor the
// bandwidth test exposes on its own.
template <int K>
static double chaseParallelNs(uint64_t levelBytes)
{
  size_t nLines = (size_t)std::max<uint64_t>(levelBytes / 64, 64);
  std::vector<uint32_t> buf(nLines * 16, 0);

  std::vector<uint32_t> order(nLines);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937 rng(0xC0FFEE);
  for (size_t i = nLines - 1; i > 0; i--)              // Sattolo: single cycle
  {
    size_t j = rng() % i;
    std::swap(order[i], order[j]);
  }
  for (size_t i = 0; i < nLines; i++)
    buf[(size_t)order[i] * 16] = order[(i + 1) % nLines] * 16;

  // K cursors staggered along the one cycle (order[] IS the cycle sequence),
  // so the chains never converge and each is a real dependent-load chain.
  uint32_t cur[K];
  for (int k = 0; k < K; k++)
    cur[k] = order[(size_t)k * nLines / K] * 16;

  using clock = std::chrono::steady_clock;
  auto nsBetween = [](clock::time_point a, clock::time_point b) {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
  };

  auto walk = [&](uint64_t steps) {
    for (uint64_t s = 0; s < steps; s++)
    {
      CPU_UNROLL_FULL_LAT
      for (int k = 0; k < K; k++) cur[k] = buf[cur[k]];
    }
  };

  uint64_t probe = 1ull << 18;
  auto p0 = clock::now();
  walk(probe);
  double nsPerStep = nsBetween(p0, clock::now()) / (double)probe;
  if (nsPerStep < 0.1) nsPerStep = 0.1;

  uint64_t steps = (uint64_t)(300e6 / nsPerStep);
  steps = std::min<uint64_t>(std::max<uint64_t>(steps, 1ull << 18), 2000000000ull);

  double best = 1e30;
  for (int rep = 0; rep < 3; rep++)
  {
    auto t0 = clock::now();
    walk(steps);
    best = std::min(best, nsBetween(t0, clock::now()) / (double)(steps * K));
  }
  uint32_t sink = 0;
  for (int k = 0; k < K; k++) sink ^= cur[k];
  volatile uint32_t keep = sink; (void)keep;
  return best;
}

// TLB / page-walk latency: one node per PAGE across far more pages than any
// L2 TLB holds, but only 64 B touched per page -- the node lines (P x 64 B,
// ~1 MB) stay cache-resident while every hop lands on a new page and misses
// the TLB.  The result is "cache hit + page walk", isolating the walker from
// DRAM latency (the plain DRAM row above mixes both).  Node line offsets are
// scattered within the page so the nodes don't alias to one cache set.
// Resident set = P x pagesize (pages commit on first touch).
static double tlbChaseNs()
{
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  const size_t pageBytes = (size_t)si.dwPageSize;
#else
  long ps = sysconf(_SC_PAGESIZE);
  const size_t pageBytes = ps > 0 ? (size_t)ps : 4096;
#endif
  // 16384 pages = 64 MB span at 4 KB pages / 256 MB at Apple's 16 KB pages;
  // >5x the largest shipping L2 TLB (3072 entries) either way.
  const size_t nPages = 16384;
  const size_t linesPerPage = pageBytes / 64;
  std::vector<uint32_t> buf(nPages * pageBytes / 4, 0);

  std::mt19937 rng(0xC0FFEE);
  std::vector<uint32_t> lineOf(nPages);
  for (size_t p = 0; p < nPages; p++) lineOf[p] = rng() % (uint32_t)linesPerPage;
  auto nodeIdx = [&](size_t p) {
    return (uint32_t)(p * (pageBytes / 4) + (size_t)lineOf[p] * 16);
  };

  std::vector<uint32_t> order(nPages);
  std::iota(order.begin(), order.end(), 0u);
  for (size_t i = nPages - 1; i > 0; i--)              // Sattolo: single cycle
  {
    size_t j = rng() % i;
    std::swap(order[i], order[j]);
  }
  for (size_t i = 0; i < nPages; i++)
    buf[nodeIdx(order[i])] = nodeIdx(order[(i + 1) % nPages]);

  using clock = std::chrono::steady_clock;
  auto nsBetween = [](clock::time_point a, clock::time_point b) {
    return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
  };

  uint32_t cur = nodeIdx(0);
  uint64_t probe = 1ull << 20;
  auto p0 = clock::now();
  for (uint64_t k = 0; k < probe; k++) cur = buf[cur];
  double nsPer = nsBetween(p0, clock::now()) / (double)probe;
  if (nsPer < 0.1) nsPer = 0.1;

  uint64_t steps = (uint64_t)(300e6 / nsPer);
  steps = std::min<uint64_t>(std::max<uint64_t>(steps, 1ull << 20), 2000000000ull);

  double best = 1e30;
  for (int rep = 0; rep < 3; rep++)
  {
    auto t0 = clock::now();
    for (uint64_t k = 0; k < steps; k++) cur = buf[cur];
    best = std::min(best, nsBetween(t0, clock::now()) / (double)steps);
  }
  volatile uint32_t keep = cur; (void)keep;
  return best;
}

int CpuPeak::runMemoryLatency(benchmark_config_t &cfg)
{
  (void)cfg;
  logger::TestSpec spec{"memory_latency", "Memory latency (pointer-chase)", "ns",
                        Category::Latency};
  auto test = currentDeviceScope->beginTest(spec);

  struct Level { const char *name; uint64_t bytes; };
  const Level levels[] = {
    {"L1",   std::max<uint64_t>(info.l1dCacheBytes / 2, 8192)},
    {"L2",   std::max<uint64_t>(info.l2CacheBytes  / 2, 65536)},
    {"L3",   std::max<uint64_t>(info.l3CacheBytes  / 2, 1u << 20)},
    {"DRAM", std::max<uint64_t>(info.l3CacheBytes  * 4, 256ull << 20)},
  };

  // Run pinned on core 0 for a stable measurement.
  std::vector<double> ns(4, -1.0);
  for (int i = 0; i < 4; i++)
  {
    uint64_t bytes = levels[i].bytes;
    pool->run(1, [&, bytes, i](int) { ns[(size_t)i] = chaseLatencyNs(bytes); });
  }

  for (int i = 0; i < 4; i++)
  {
    if (ns[(size_t)i] > 0)
      test.emit(levels[i].name, (float)ns[(size_t)i]);
    else
      test.skip(levels[i].name, ResultStatus::Error, "latency walk failed");
  }

  // Memory-level parallelism: effective ns/access over the DRAM working set
  // at depth 8 and 32.  DRAM / (DRAM x32) is the MLP factor (outstanding
  // misses the core sustains) -- see chaseParallelNs.
  const uint64_t dramBytes = levels[3].bytes;
  double ns8 = -1.0, ns32 = -1.0, nsTlb = -1.0;
  pool->run(1, [&](int) { ns8 = chaseParallelNs<8>(dramBytes); });
  pool->run(1, [&](int) { ns32 = chaseParallelNs<32>(dramBytes); });
  if (ns8 > 0)  test.emit("DRAM x8",  (float)ns8);
  else          test.skip("DRAM x8",  ResultStatus::Error, "latency walk failed");
  if (ns32 > 0) test.emit("DRAM x32", (float)ns32);
  else          test.skip("DRAM x32", ResultStatus::Error, "latency walk failed");

  // TLB miss / page-walk: cache-resident nodes, one per page, across more
  // pages than any L2 TLB -- "cache hit + page walk" (vs the DRAM row, which
  // is "DRAM + page walk"); pointer-heavy object graphs see this cost.
  pool->run(1, [&](int) { nsTlb = tlbChaseNs(); });
  if (nsTlb > 0) test.emit("TLB miss", (float)nsTlb);
  else           test.skip("TLB miss", ResultStatus::Error, "latency walk failed");
  return 0;
}

#endif // ENABLE_CPU
