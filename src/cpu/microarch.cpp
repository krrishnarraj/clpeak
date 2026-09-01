#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/run_document.h>
#include "cpu_kernels.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

// Microarchitecture probes (Category::Latency): atomic-RMW cost and branch-
// mispredict penalty.  Both report nanoseconds -- these are per-operation
// costs, not throughput peaks, and they are the two numbers synchronization-
// heavy (databases, browsers) and branchy (interpreters, parsers) code lives
// and dies by.

// ---------------------------------------------------------------------------
// Atomic fetch-add: uncontended (1 thread, line stays in its L1, measures the
// core's atomic-RMW cost) and contended (every core hammering ONE cache line,
// measures how gracefully the coherence fabric serializes ownership).  The
// contended row is system-wide: wall time per completed op across all cores.
// Relaxed ordering on purpose -- we measure the RMW itself, not fence cost.
// ---------------------------------------------------------------------------
int CpuPeak::runAtomics(benchmark_config_t &cfg)
{
  logger::TestSpec spec{"atomics", "Atomic fetch-add latency", "s",
                        Category::Latency,
                        "How long one all-or-nothing counter update takes.  This is "
                        "the operation behind locks, reference counts and every "
                        "shared data structure, so it sets the cost of coordinating "
                        "between threads.",
                        TestShape::Heterogeneous, "contention"};
  auto test = currentDeviceScope->beginTest(spec);

  struct alignas(64) PaddedAtomic { std::atomic<uint64_t> v{0}; };
  static PaddedAtomic ctr;

  Workload body = [](int, uint64_t iters) {
    for (uint64_t i = 0; i < iters; i++)
      ctr.v.fetch_add(1, std::memory_order_relaxed);
  };

  const int maxT = pool->maxThreads();
  unsigned int forced = forceIters ? specifiedIters : 0;
  double us1 = runWorkload(1,    body, cfg.targetTimeUs, forced);
  double usN = runWorkload(maxT, body, cfg.targetTimeUs, forced);

  const char *unNote = "One thread updating a counter nobody else touches, so it "
                       "stays in that core's own cache.";
  const char *coNote = "Every core fighting over the same counter, which has to be "
                       "handed from core to core -- the time between completed "
                       "updates anywhere on the chip.";
  if (us1 > 0) test.emit("uncontended ST", (float)(us1 * 1e-6), unNote);
  else         test.skip("uncontended ST", ResultStatus::Error, "workload failed", unNote);
  // Each of the maxT threads completes one op per mean iteration, so the
  // system-wide time between completions is wall / maxT.
  if (usN > 0) test.emit("contended MT", (float)(usN * 1e-6 / (double)maxT), coNote);
  else         test.skip("contended MT", ResultStatus::Error, "workload failed", coNote);
  return 0;
}

// ---------------------------------------------------------------------------
// Branch mispredict penalty: the classic sorted-vs-shuffled data-dependent
// branch.  Same data, same instruction stream -- sorted input predicts
// perfectly, shuffled input is a coin flip the predictor cannot learn (the
// pattern period, N elements, is far beyond any TAGE history), so the time
// delta divided by the mispredict count is the pipeline-refill cost.  That
// cost is the single biggest determinant of interpreter/parser speed.
//
// The taken arm writes through a volatile pointer: a conditional volatile
// store cannot be speculated or if-converted, so the compiler MUST emit a
// real conditional branch (clang/GCC turn a plain two-sided += into
// csel/cmov, which has no misprediction to measure).  The store executes for
// the same elements in both runs (same data), so its cost cancels in the
// delta.  Verify with objdump: the loop body must contain a conditional
// branch, not csel/cmov.
// ---------------------------------------------------------------------------
namespace {

static double branchPassNs(const uint8_t *v, size_t n, uint64_t passes)
{
  using clock = std::chrono::steady_clock;
  uint64_t s1 = 0, s2 = 0;
  volatile uint64_t sink = 0;
  auto t0 = clock::now();
  for (uint64_t p = 0; p < passes; p++)
    for (size_t i = 0; i < n; i++)
    {
      if (v[i] < 128) { s1 += v[i]; sink = s1; }
      else            { s2 += v[i]; }
    }
  double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                  clock::now() - t0).count();
  sink = s1 + s2;
  (void)sink;
  return ns / (double)(passes * n);
}

static void branchPenalty(double &predictedNs, double &randomNs, double &penaltyNs)
{
  const size_t n = 1u << 18;              // 256 K elements: period >> any history
  std::vector<uint8_t> data(n);
  std::mt19937 rng(0xC0FFEE);
  for (size_t i = 0; i < n; i++) data[i] = (uint8_t)(rng() & 0xFF);

  std::vector<uint8_t> sorted = data;
  std::sort(sorted.begin(), sorted.end());

  // Size the run to ~200 ms per variant off a shuffled probe.
  double probeNs = branchPassNs(data.data(), n, 4);
  uint64_t passes = (uint64_t)(200e6 / (probeNs * (double)n));
  passes = std::min<uint64_t>(std::max<uint64_t>(passes, 8), 1u << 20);

  double tSorted = 1e30, tRandom = 1e30;
  for (int rep = 0; rep < 3; rep++)
  {
    tSorted = std::min(tSorted, branchPassNs(sorted.data(), n, passes));
    tRandom = std::min(tRandom, branchPassNs(data.data(), n, passes));
  }

  predictedNs = tSorted;
  randomNs    = tRandom;
  // 50/50 random data against any predictor -> ~0.5 mispredicts per branch.
  penaltyNs   = (tRandom - tSorted) / 0.5;
}

} // anonymous namespace

int CpuPeak::runBranchPenalty(benchmark_config_t &cfg)
{
  (void)cfg;
  logger::TestSpec spec{"branch_mispredict", "Branch mispredict penalty", "s",
                        Category::Latency,
                        "What it costs when the CPU guesses the wrong way at an "
                        "if-statement.  The same loop runs over sorted data (easy to "
                        "guess) and shuffled data (a coin flip), and the difference "
                        "is the price of a wrong guess.",
                        // The third reading is the difference between the other
                        // two, so nothing here is a variant of anything else.
                        TestShape::Heterogeneous};
  auto test = currentDeviceScope->beginTest(spec);

  // Pinned single-thread, like the pointer-chase.
  double pred = -1.0, rnd = -1.0, pen = -1.0;
  pool->run(1, [&](int) { branchPenalty(pred, rnd, pen); });

  const char *penNote = "The extra time one wrong guess costs, worked out from the "
                        "gap between the other two readings.";
  if (pred > 0 && rnd > 0)
  {
    test.emit("predicted", (float)(pred * 1e-9),       // s/branch, sorted input
              "Time per if-statement on sorted data, where the CPU guesses right "
              "every time.");
    test.emit("random", (float)(rnd * 1e-9),           // s/branch, 50/50 input
              "Time per if-statement on shuffled data, where the CPU is wrong "
              "about half the time.");
    test.emit("penalty", (float)(std::max(pen, 0.0) * 1e-9), penNote);  // s per mispredict
  }
  else
  {
    test.skip("penalty", ResultStatus::Error, "branch benchmark failed", penNote);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Store-to-load forwarding: a dependent store -> same-address load -> +1
// chain through one hot cache line, reported as ns per roundtrip (it includes
// the one dependent add).  A core with no fast forwarding path pays its store
// -forward latency every iteration; a core that resolves the dependency
// internally approaches the ~1-cycle cost of the add alone.
//
// Do NOT read a ~1-cycle result as proof of any particular mechanism.  With
// the codegen verified on both, M1 Pro measures 1.01 cyc and Zen 2 1.10 --
// but an address-rotation control separates them completely: alternating the
// chain between two slots costs M1 Pro 4.05 cyc/roundtrip and Zen 2 only
// 1.29.  Two cores reach the same headline number by visibly different means,
// so this row is a cost probe, not a feature detector.  It also means the
// vendor-manual "~7 cycle" forwarding figures describe imperfect matches,
// not the exact-address/exact-width case this loop exercises.
//
// `volatile` alone is NOT enough to keep the load and store separate.  GCC
// folds the whole body into a single memory-destination RMW -- verified on
// x86-64 GCC 13 at -O3, where the hot loop is one `addq $0x1,-0x40(%rsp)` --
// so the row measured read-modify-write throughput there while measuring a
// real roundtrip on ARM/clang (`ldr`/`add`/`str`).  Same row, two different
// quantities, and the x86 number looked like renaming (1.1 cycles on Zen 2)
// when it was nothing of the sort.  Passing the loaded value through an empty
// asm constraint forces it into a general-purpose register between the two
// accesses, which no compiler can fuse back together.
// ---------------------------------------------------------------------------
namespace {

// Force `x` to materialise in a GPR, without emitting an instruction.
#if defined(__GNUC__) || defined(__clang__)
#define CPU_KEEP_IN_REG(x) __asm__ volatile("" : "+r"(x))
#else
// Real MSVC has no GNU inline asm; it also does not fold volatile RMW today.
// If that ever changes, this row needs an equivalent barrier on cl.exe.
#define CPU_KEEP_IN_REG(x) do { } while (0)
#endif

static double storeForwardNs()
{
  using clock = std::chrono::steady_clock;
  alignas(64) volatile uint64_t slot = 1;

  auto roundtrips = [&](uint64_t n) {
    for (uint64_t i = 0; i < n; i++)
    {
      uint64_t v = slot;    // load (forwarded from the store below)
      CPU_KEEP_IN_REG(v);   // ... and it must land in a register, not fuse
      slot = v + 1;         // dependent store back to the same address
    }
  };

  uint64_t probe = 1ull << 22;
  auto p0 = clock::now();
  roundtrips(probe);
  double nsPer = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                     clock::now() - p0).count() / (double)probe;
  if (nsPer < 0.05) nsPer = 0.05;

  uint64_t steps = (uint64_t)(200e6 / nsPer);
  steps = std::min<uint64_t>(std::max<uint64_t>(steps, 1ull << 22), 2000000000ull);

  double best = 1e30;
  for (int rep = 0; rep < 3; rep++)
  {
    auto t0 = clock::now();
    roundtrips(steps);
    double ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
                    clock::now() - t0).count() / (double)steps;
    best = std::min(best, ns);
  }
  return best;
}

} // anonymous namespace

int CpuPeak::runStoreForward(benchmark_config_t &cfg)
{
  (void)cfg;
  logger::TestSpec spec{
      "store_forward", "Store-to-load forwarding", "s",
      Category::Latency,
      "How fast the core can read back a value it just wrote, without the "
      "write having reached the cache yet.",
      TestShape::Homogeneous};
  auto test = currentDeviceScope->beginTest(spec);

  double ns = -1.0;
  pool->run(1, [&](int) { ns = storeForwardNs(); });
  if (ns > 0) test.emit("st->ld ST", (float)(ns * 1e-9));
  else        test.skip("st->ld ST", ResultStatus::Error, "workload failed");
  return 0;
}

// ---------------------------------------------------------------------------
// SMT scaling: the widest fp32 FMA chain run MT twice -- once with ONE thread
// per physical core, once on every logical thread -- and both rates emitted.
// The ratio is the SMT story: conventional SMT gains ~10-30% on a compute
// chain; statically partitioned designs (NVIDIA Vera's "spatial multi-
// threading") sit near 1.0x by construction, and negative scaling exposes
// resource-starved SMT.  Needs one-thread-per-core placement, so it requires
// hard affinity (Linux / Windows) plus sibling topology; skipped elsewhere
// and on CPUs without SMT.
// ---------------------------------------------------------------------------
namespace {

// One logical CPU id per physical core (the lowest-numbered sibling), or
// empty when the topology is unavailable / unpinnable.
static std::vector<int> primaryCpusPerCore(int logicalCores)
{
#if defined(__linux__)
  std::vector<int> primaries;
  std::set<std::pair<int, int>> seen;   // (package, core)
  for (int cpu = 0; cpu < logicalCores; cpu++)
  {
    char path[128];
    int pkg = 0, core = -1;
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    if (FILE *f = std::fopen(path, "r")) { if (std::fscanf(f, "%d", &pkg) != 1) pkg = 0; std::fclose(f); }
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
    if (FILE *f = std::fopen(path, "r")) { if (std::fscanf(f, "%d", &core) != 1) core = -1; std::fclose(f); }
    if (core < 0)
      return {};
    if (seen.insert({pkg, core}).second)
      primaries.push_back(cpu);
  }
  return primaries;
#elif defined(_WIN32)
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || len == 0)
    return {};
  std::vector<char> buf(len);
  auto *info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len))
    return {};
  std::vector<int> primaries;
  for (char *p = buf.data(); p < buf.data() + len;)
  {
    auto *e = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(p);
    if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount >= 1)
    {
      const GROUP_AFFINITY &ga = e->Processor.GroupMask[0];
      // pinToCore() can only address the first 64 logical CPUs of group 0, so
      // give up on machines beyond that rather than mis-pin.
      if (ga.Group != 0)
        return {};
      for (int b = 0; b < 64; b++)
        if (ga.Mask & (1ull << b)) { primaries.push_back(b); break; }
    }
    p += e->Size;
  }
  std::sort(primaries.begin(), primaries.end());
  (void)logicalCores;
  return primaries;
#else
  (void)logicalCores;
  return {};   // macOS: no hard affinity (and Apple Silicon has no SMT)
#endif
}

} // anonymous namespace

int CpuPeak::runSmtScaling(benchmark_config_t &cfg)
{
  logger::TestSpec spec{"smt_scaling", "SMT scaling (fp32 FMA)", "flops",
                        Category::Unknown,
                        "Whether the CPU's extra hardware threads (SMT, or Intel's "
                        "Hyper-Threading) actually add throughput: the same maths "
                        "kernel run once with one thread per physical core, then "
                        "again on every thread the chip offers.",
                        TestShape::Heterogeneous, "threads"};
  auto test = currentDeviceScope->beginTest(spec);

  if (info.logicalCores <= info.physicalCores || info.physicalCores < 1)
  {
    test.skip("1T/core", ResultStatus::Unsupported, "no SMT on this CPU");
    return 0;
  }

  std::vector<int> primaries = primaryCpusPerCore(info.logicalCores);
  if ((int)primaries.size() != info.physicalCores)
  {
    test.skip("1T/core", ResultStatus::Unsupported,
              "SMT sibling topology unavailable (needs Linux sysfs / Windows GLPI + hard affinity)");
    return 0;
  }

  // Widest non-streaming fp32 chain (skip the SSVE row: the SME unit is a
  // shared per-cluster resource, which would measure the wrong thing here).
  const auto &fp32 = clpeak_cpu::kernelMenu().fp32;
  const clpeak_cpu::IsaVariant *chain = nullptr;
  for (auto it = fp32.rbegin(); it != fp32.rend(); ++it)
    if (std::string(it->isa).rfind("SSVE", 0) != 0) { chain = &*it; break; }
  if (!chain)
  {
    test.skip("1T/core", ResultStatus::Error, "no fp32 kernel available");
    return 0;
  }

  const int nPhys = info.physicalCores;
  const int nAll  = pool->maxThreads();
  std::vector<double> sink((size_t)std::max(nPhys, nAll), 0.0);
  Workload body = [&](int tid, uint64_t iters) {
    sink[(size_t)tid] += chain->v.fn(iters);
  };
  unsigned int forced = forceIters ? specifiedIters : 0;

  // One worker per physical core, pinned to the primary sibling.  The main
  // pool pins worker i to logical CPU i, which lands sibling pairs on the
  // same core on most enumerations -- hence the dedicated pool.
  double usPhys;
  {
    CpuThreadPool physPool(nPhys, primaries);
    CpuThreadPool *saved = pool;
    pool = &physPool;                    // runWorkload dispatches via `pool`
    usPhys = runWorkload(nPhys, body, cfg.targetTimeUs, forced);
    pool = saved;
  }
  double usAll = runWorkload(nAll, body, cfg.targetTimeUs, forced);

  volatile double keep = 0.0;
  for (double s : sink) keep += s;
  (void)keep;

  auto giga = [&](int n, double us) {
    return (float)(chain->v.opsPerIter * (double)n / (us * 1e-6));
  };
  const char *physNote = "One thread per physical core, with the extra hardware "
                         "threads left idle.";
  const char *smtNote  = "Every hardware thread busy, including the second thread "
                         "sharing each core.";
  if (usPhys > 0) test.emit("1T/core", giga(nPhys, usPhys), physNote);
  else            test.skip("1T/core", ResultStatus::Error, "workload failed", physNote);
  if (usAll > 0)  test.emit("SMT MT", giga(nAll, usAll), smtNote);
  else            test.skip("SMT MT", ResultStatus::Error, "workload failed", smtNote);
  return 0;
}

#endif // ENABLE_CPU
