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
  return 0;
}

#endif // ENABLE_CPU
