#ifdef ENABLE_CPU

#include "cache_probe.h"

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/screen_hold.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#endif
#if defined(__linux__)
#include <sched.h>
#endif

// ── Measuring the cache sizes the OS does not give ──────────────────────────
//
// An Android kernel lists its CPUs' caches from the device tree, and on most
// phones that tree names the levels and leaves out their sizes: sysfs has an
// L1d, an L2 and an L3 with no `size` (Galaxy S24, S24 Ultra, Z Fold8), or,
// where the tree has no cache nodes at all, an L1d alone (CMF Phone 1).  A
// working set is not sized for a cache it can only guess at, so the size is
// measured: a dependent-load walk -- each load's address is the value the
// last one read, so the core waits out every load in full -- gets slower in
// steps as its working set outgrows each cache, and each step is where one
// level ends.  The steps are matched, smallest first, to the levels the OS
// lists, and a level whose step is not found stays unsized.  A level the OS
// does not list is not made up: past an L1d a phone can show a partitioned
// L3, a system cache only its big cores use and prefetch effects, and which
// of those to call an L2 or an L3 would be a guess.
//
// Two walks are timed at every size, and the faster kept:
//
//   * one random across the whole set, which no prefetcher can follow, but
//     which misses the TLB on every load once the set outgrows its reach --
//     a rise of up to ~1.7x inside a Cortex-X4's 2 MB L2 that is a step of
//     its own;
//   * one in runs of eight nodes inside a 4 KB region, the runs in random
//     order, which looks a page up once a run -- but in which a prefetcher
//     that sees a page used densely fetches more of it, and filled a CMF
//     Phone 1's last-level cache with lines the walk did not read: its big
//     cores fell to DRAM at 5 MB on it and at 13 MB on the random walk.
//
// Each costs more, never less, than the cache does; so the faster of the two
// is the cache -- unless the runs walk is under half the random one, which
// only a prefetcher fetching the walk's own lines can do, and then it is the
// random walk.  Where the kernel gives transparent huge pages the buffer asks
// for them, and the TLB is out of it.
//
// What else stays out of the curve:
//
//   * The clock.  A phone changes it under load and heat, and a slower clock
//     is a slower walk at every size, which reads as a step.  So each point
//     is timed against a 4 KB walk -- the L1 -- either side of it, the faster
//     of the two, and the curve is the ratio.  The levels inside the core
//     scale with its clock; the steps between them stay where they are.
//   * Noise.  Each point is the fastest of several ~1 ms slices, the points
//     about each step are timed again and keep their faster pass, a point
//     out of line with both neighbours is set aside, and a step must climb
//     1.4x: the next level out costs 1.5x-10x more than a cache.
//
// Where the size is read off the step depends on how evenly the level's sets
// fill.  An L1d is indexed by virtual address, so its sets fill evenly and
// the walk falls out of it at its size: the L1d is the last working set
// before the rise.  The levels past it are indexed by physical address, and
// a process's 4 KB pages land at random in their sets: some fill before the
// cache does, and the rise spreads across its size -- a CMF Phone 1's 256 KB
// L2 starts at 180 KB and ends past 512 KB, an M1 Pro's 12 MB starts at 8 --
// so the size is where the rise is halfway.  On huge pages, each 2 MB of the
// walk is contiguous, every level fills evenly, and every step is sharp: a
// Threadripper's 512 KB L2 is flat to 512 KB, and halfway would be 584.
//
// A class of core -- the cores of one rank, a phone's cluster -- shares a
// cache layout, so one core of each measures it, pinned there; a cache
// several classes read is measured by the fastest of them.  Each point is a
// job of its own on that core's worker, which re-pins at each job and checks
// where it ran, and between jobs the run's hold waits while the app is off
// screen: a point timed off screen ran on the background's cores and clocks,
// so it is timed again.

namespace {

using Clock = std::chrono::steady_clock;

constexpr uint64_t kLine     = 64;            // one node per line: every line of the set is read
constexpr uint64_t kRegion   = 4096;          // a run stays inside one 4 KB region -- one page
constexpr unsigned kRun      = 8;             // nodes per run, for the page-friendly walk
constexpr uint64_t kGrain    = kLine * kRun;  // working sets are whole runs
constexpr uint64_t kRefBytes = 4096;          // the reference walk: the L1 on any core
constexpr uint64_t kAlign    = 2ull << 20;    // a transparent huge page

// Thresholds of the curve, in log2 of the latency against the reference, per
// octave of working set.
constexpr double kSteep   = 0.8;    // a rise begins this steep
constexpr double kEase    = 0.2;    // ... and goes on while it climbs this much
constexpr double kMinRise = 1.4;    // ... and is a step once it has climbed this far
constexpr double kOnFloor = 1.05;   // the last working set on a level: within this of it

double nsSince(Clock::time_point t0)
{
  return std::chrono::duration<double, std::nano>(Clock::now() - t0).count();
}

// `hops` dependent loads from node `cur`, eight to a loop trip so the loop's
// own instructions are noise beside them.
uint32_t chase(const uint8_t *base, uint32_t cur, uint64_t hops)
{
  for (uint64_t k = 0; k < hops; k += 8)
  {
    cur = *reinterpret_cast<const uint32_t *>(base + cur);
    cur = *reinterpret_cast<const uint32_t *>(base + cur);
    cur = *reinterpret_cast<const uint32_t *>(base + cur);
    cur = *reinterpret_cast<const uint32_t *>(base + cur);
    cur = *reinterpret_cast<const uint32_t *>(base + cur);
    cur = *reinterpret_cast<const uint32_t *>(base + cur);
    cur = *reinterpret_cast<const uint32_t *>(base + cur);
    cur = *reinterpret_cast<const uint32_t *>(base + cur);
  }
  return cur;
}

uint8_t *mapBytes(uint64_t bytes)
{
#if defined(_WIN32)
  return static_cast<uint8_t *>(
      VirtualAlloc(nullptr, (SIZE_T)bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
#else
  void *p = mmap(nullptr, (size_t)bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                 -1, 0);
  return p == MAP_FAILED ? nullptr : static_cast<uint8_t *>(p);
#endif
}

void unmapBytes(uint8_t *p, uint64_t bytes)
{
  if (!p)
    return;
#if defined(_WIN32)
  (void)bytes;
  VirtualFree(p, 0, MEM_RELEASE);
#else
  munmap(p, (size_t)bytes);
#endif
}

// xorshift64*: the shuffles of a large set want tens of millions of draws.
struct FastRng
{
  uint64_t s;
  uint32_t below(uint32_t n)
  {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    return (uint32_t)(((s * 0x2545F4914F6CDD1Dull) >> 32) * n >> 32);
  }
};

void shuffle(uint32_t *v, uint32_t n, FastRng &rng)
{
  for (uint32_t i = n; i > 1; i--)
    std::swap(v[i - 1], v[rng.below(i)]);
}

double median3(double a, double b, double c)
{
  return std::max(std::min(a, b), std::min(std::max(a, b), c));
}

std::string sizeText(double bytes)
{
  char buf[32];
  if (bytes >= 1048576.0)
    std::snprintf(buf, sizeof(buf), "%.3g MB", bytes / 1048576.0);
  else
    std::snprintf(buf, sizeof(buf), "%.3g KB", bytes / 1024.0);
  return buf;
}

// Two significant digits, in KB or MB: what a step spread by scattered pages
// can say.
uint64_t twoDigits(double bytes)
{
  const double unit  = bytes >= 1048576.0 ? 1048576.0 : 1024.0;
  const double v     = bytes / unit;
  const double scale = std::pow(10.0, std::floor(std::log10(v)) - 1);
  return (uint64_t)std::llround(std::round(v / scale) * scale * unit);
}

// The faster of the two walks at one size: their artefacts only ever add.
// A runs walk under half the random one is being prefetched, not cached.
CacheSweepPoint combine(const CacheSweepPoint &runs, const CacheSweepPoint &random)
{
  if (runs.ratio() < 0.5 * random.ratio())
    return random;
  return runs.ratio() <= random.ratio() ? runs : random;
}

} // namespace

// ── The walker ──────────────────────────────────────────────────────────────

CacheWalker::CacheWalker(uint64_t maxBytes) : maxBytes(maxBytes), rng(0xC0FFEE) {}

CacheWalker::~CacheWalker()
{
  unmapBytes(region, regionBytes);
  unmapBytes(ref, kRefBytes);
}

bool CacheWalker::reserve(uint64_t bytes)
{
  if (bytes <= capacity)
    return true;
  unmapBytes(region, regionBytes);
  region = buf = nullptr;
  capacity = regionBytes = 0;
  // Grown in steps, so a sweep that stops at 64 MB never maps a gigabyte.
  const uint64_t want = std::min<uint64_t>(maxBytes, std::max<uint64_t>(bytes, 32ull << 20));
  if (bytes > want)
    return false;
  region = mapBytes(want + kAlign);
  if (!region)
    return false;
  regionBytes = want + kAlign;
  buf = reinterpret_cast<uint8_t *>(((uintptr_t)region + kAlign - 1) & ~(uintptr_t)(kAlign - 1));
#if defined(MADV_HUGEPAGE)
  (void)madvise(buf, (size_t)want, MADV_HUGEPAGE);   // a request; refused where THP is off
#endif
  // Touched now, so no page fault lands inside a timed walk.
  std::memset(buf, 0, (size_t)want);
  capacity = want;
  huge = false;
#if defined(__linux__)
  // Whether the kernel took the request: the mapping's AnonHugePages.
  if (FILE *f = std::fopen("/proc/self/smaps", "r"))
  {
    char line[256];
    bool inside = false;
    while (std::fgets(line, sizeof(line), f))
    {
      unsigned long lo = 0, hi = 0, kb = 0;
      if (std::sscanf(line, "%lx-%lx ", &lo, &hi) == 2)
        inside = (uintptr_t)buf >= lo && (uintptr_t)buf < hi;
      else if (inside && std::sscanf(line, "AnonHugePages: %lu kB", &kb) == 1)
      {
        huge = kb * 1024ull >= want / 2;
        break;
      }
    }
    std::fclose(f);
  }
#endif
  return true;
}

uint32_t CacheWalker::build(uint8_t *base, uint64_t bytes, unsigned run)
{
  const uint32_t nodes     = (uint32_t)(bytes / kLine);
  const uint32_t perRegion = (uint32_t)(kRegion / kLine);
  FastRng fast{rng() | 1};
  order.resize(nodes);
  std::iota(order.begin(), order.end(), 0u);
  if (run > 1)
    for (uint32_t r = 0; r < nodes; r += perRegion)
      shuffle(order.data() + r, std::min(perRegion, nodes - r), fast);
  else
    shuffle(order.data(), nodes, fast);
  // Runs of `run` consecutive entries of `order` -- inside one region when
  // run > 1 -- visited in random order.
  const uint32_t nRuns = (nodes + run - 1) / run;
  runs.resize(nRuns);
  std::iota(runs.begin(), runs.end(), 0u);
  if (run > 1)
    shuffle(runs.data(), nRuns, fast);

  bool any = false;
  uint32_t first = 0, prev = 0;
  for (uint32_t r : runs)
  {
    const uint32_t end = std::min<uint32_t>(nodes, (r + 1) * run);
    for (uint32_t i = r * run; i < end; i++)
    {
      const uint32_t node = order[i];
      if (any)
        *reinterpret_cast<uint32_t *>(base + (uint64_t)prev * kLine) = node * (uint32_t)kLine;
      else
        first = node;
      any  = true;
      prev = node;
    }
  }
  *reinterpret_cast<uint32_t *>(base + (uint64_t)prev * kLine) = first * (uint32_t)kLine;
  return first * (uint32_t)kLine;
}

double CacheWalker::timeWalk(const uint8_t *base, uint64_t bytes, uint32_t from, double capNs,
                             double sliceNs, double minNs)
{
  // Once round the set, so all of it that fits is in the cache it fits in --
  // capped at ~40 ms by the last point's rate, for a set far past any cache.
  const uint64_t nodes = bytes / kLine;
  uint64_t warm = nodes;
  if (capNs > 0.0)
    warm = std::min<uint64_t>(warm, (uint64_t)(40e6 / capNs));
  warm = (std::max<uint64_t>(warm, 512) + 7) & ~7ull;
  auto t0 = Clock::now();
  uint32_t cur = chase(base, from, warm);
  double est = nsSince(t0) / (double)warm;
  if (est <= 0.0)
    est = 1.0;

  // Then slices, the fastest of them: an interrupt spoils one slice.
  const uint64_t slice = (std::min<uint64_t>(std::max<uint64_t>((uint64_t)(sliceNs / est), 1024),
                                             1ull << 24) + 7) & ~7ull;
  double best = 1e30, spent = 0.0;
  for (int k = 0; k < 64 && (k < 3 || spent < minNs); k++)
  {
    t0 = Clock::now();
    cur = chase(base, cur, slice);
    const double ns = nsSince(t0);
    spent += ns;
    best = std::min(best, ns / (double)slice);
  }
  volatile uint32_t keep = cur;
  (void)keep;
  return best;
}

bool CacheWalker::measure(uint64_t bytes, unsigned run, CacheSweepPoint &out)
{
  bytes = std::max<uint64_t>(kGrain, bytes / kGrain * kGrain);
  if (!ref)
  {
    ref = mapBytes(kRefBytes);
    if (!ref)
      return false;
    refStart = build(ref, kRefBytes, 1);
  }
  if (!reserve(bytes))
    return false;

  // The L1 at the clock the core runs at now, either side of the point: the
  // faster of the two, so one the clock dipped under is not the reference.
  const uint32_t from = build(buf, bytes, std::max(run, 1u));
  const double before = timeWalk(ref, kRefBytes, refStart, 0.0, 0.5e6, 1.5e6);
  out.bytes = bytes;
  out.ns    = timeWalk(buf, bytes, from, lastNs, 1e6, 6e6);
  out.refNs = std::min(before, timeWalk(ref, kRefBytes, refStart, 0.0, 0.5e6, 1.5e6));
  lastNs    = out.ns;
  return true;
}

bool CacheWalker::thrash(uint64_t bytes, const std::atomic<bool> &stop)
{
  bytes = std::max<uint64_t>(kGrain, bytes / kGrain * kGrain);
  if (!reserve(bytes))
    return false;
  // A stream, not a walk: one line per load, eight loads in flight, so the
  // lines come in as fast as the core can take them.  A walk brings one in
  // per memory latency, and a cache that keeps its most recent lines keeps
  // the measured core's, which it reads a hundred times as often.
  const uint64_t *p = reinterpret_cast<const uint64_t *>(buf);
  const size_t step = kLine / sizeof(uint64_t), words = (size_t)(bytes / sizeof(uint64_t));
  uint64_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  while (!stop.load(std::memory_order_relaxed))
    for (size_t i = 0; i + 8 * step <= words; i += 8 * step)
      for (int k = 0; k < 8; k++)
        acc[k] += p[i + (size_t)k * step];
  volatile uint64_t keep = acc[0] ^ acc[1] ^ acc[2] ^ acc[3] ^ acc[4] ^ acc[5] ^ acc[6] ^ acc[7];
  (void)keep;
  return true;
}

// ── The steps ───────────────────────────────────────────────────────────────

std::vector<CacheStep> findCacheSteps(const std::vector<CacheSweepPoint> &pts)
{
  std::vector<CacheStep> out;
  const size_t n = pts.size();
  if (n < 4)
    return out;
  std::vector<double> x(n), y(n);
  for (size_t i = 0; i < n; i++)
  {
    x[i] = std::log2((double)pts[i].bytes);
    const double r = pts[i].ratio();
    y[i] = r > 0.0 ? std::log2(r) : 0.0;
  }
  // A point out of line with both neighbours is one an interrupt or a clock
  // change spoiled; a step holds for every point past it.
  std::vector<double> ys(y);
  for (size_t i = 1; i + 1 < n; i++)
    ys[i] = median3(y[i - 1], y[i], y[i + 1]);
  std::vector<double> slope(n - 1);
  for (size_t i = 0; i + 1 < n; i++)
    slope[i] = (ys[i + 1] - ys[i]) / std::max(x[i + 1] - x[i], 1e-9);

  const double minRise = std::log2(kMinRise);
  // One rise from point a to point b; halfway is where its level is halfway.
  auto step = [&](size_t a, size_t b) {
    CacheStep s;
    s.onset  = a;
    s.end    = b;
    s.before = std::exp2(ys[a]);
    s.after  = std::exp2(ys[b]);
    const double mid = 0.5 * (ys[a] + ys[b]);
    size_t k = a;
    while (k + 1 < b && ys[k + 1] < mid)
      k++;
    const double f = std::clamp((mid - ys[k]) / std::max(ys[k + 1] - ys[k], 1e-9), 0.0, 1.0);
    s.half      = k;
    s.halfBytes = std::exp2(x[k] + f * (x[k + 1] - x[k]));
    return s;
  };
  // A climb from point a to point b can be two rises in a row -- an L2 into a
  // system cache only twice its size, and that into DRAM -- with the slope
  // easing between them and steepening again.  Split there when each side
  // still climbs 1.4x, or halfway would be halfway across both levels.
  std::function<void(size_t, size_t)> split = [&](size_t a, size_t b) {
    size_t at = 0;
    double best = 1.0;
    for (size_t m = a + 1; m + 1 < b; m++)
    {
      double peakBefore = 0.0, peakAfter = 0.0;
      for (size_t q = a; q < m; q++) peakBefore = std::max(peakBefore, slope[q]);
      for (size_t q = m + 1; q < b; q++) peakAfter = std::max(peakAfter, slope[q]);
      const double dip = slope[m] / std::max(std::min(peakBefore, peakAfter), 1e-9);
      if (slope[m] < 0.6 * peakBefore && peakAfter > 1.5 * slope[m] && dip < best &&
          ys[m] - ys[a] >= minRise && ys[b] - ys[m + 1] >= minRise)
      {
        best = dip;
        at   = m;
      }
    }
    if (!at)
    {
      out.push_back(step(a, b));
      return;
    }
    split(a, at);
    split(at + 1, b);
  };

  // A rise holds at least one steep interval.  A cache whose sets fill
  // unevenly hands over to the next level across an octave or two, gently
  // at first and with pauses -- a CMF Phone 1's 256 KB L2 starts climbing at
  // 152 KB, steepens at 181, eases at 300 and climbs on to 600 -- so a rise
  // reaches back over the gentle climb that leads into it, and on through
  // gentle intervals and through up to two flat ones where the climb
  // resumes gently.  Where it resumes steeply, that is the next level's.
  size_t i = 0;
  while (i + 1 < n)
  {
    size_t s0 = i;
    while (s0 + 1 < n && slope[s0] < kSteep)
      s0++;
    if (s0 + 1 >= n)
      break;
    size_t a = s0;
    while (a > i && slope[a - 1] >= kEase)
      a--;
    size_t b = s0 + 1, k = s0 + 1;
    while (k + 1 < n)
    {
      if (slope[k] >= kEase && (slope[k] < kSteep || k == b))
      {
        k++;
        b = k;
        continue;
      }
      if (slope[k] >= kSteep)
        break;   // steep again after a gentle stretch: still this rise only if contiguous
      size_t g = k;
      while (g + 1 < n && g < k + 2 && slope[g] < kEase)
        g++;
      if (g + 1 < n && slope[g] >= kEase && slope[g] < kSteep)
      {
        k = g;
        continue;
      }
      break;
    }
    if (ys[b] - ys[a] >= minRise)
      split(a, b);
    i = b;
  }
  return out;
}

uint64_t levelSize(const std::vector<CacheSweepPoint> &pts, const CacheStep &st, bool even,
                   const std::function<bool(uint64_t, CacheSweepPoint &)> &measure)
{
  if (even)
  {
    // The last working set still on the level, refined across the interval
    // the rise begins in: three more sizes, the largest still within 10% of
    // the level.  A size the walk lands on exactly -- a power of two -- is
    // the cache's when it is the last to fit.
    uint64_t b = pts[st.onset].bytes;
    const double lo = (double)pts[st.onset].bytes, hi = (double)pts[st.onset + 1].bytes;
    for (int m = 1; m <= 3; m++)
    {
      CacheSweepPoint pt;
      const uint64_t size = (uint64_t)(lo * std::pow(hi / lo, m / 4.0)) / kGrain * kGrain;
      if (size <= b || !measure(size, pt) || pt.ratio() > st.before * kOnFloor)
        break;
      b = pt.bytes;
    }
    return b == pts[st.onset].bytes ? b : twoDigits((double)b);
  }
  // Halfway up the rise, refined across the interval that is in: three more
  // sizes, and the crossing interpolated between the two either side of it.
  std::vector<CacheSweepPoint> local = {pts[st.half]};
  const double lo = (double)pts[st.half].bytes, hi = (double)pts[st.half + 1].bytes;
  for (int m = 1; m <= 3; m++)
  {
    CacheSweepPoint pt;
    if (measure((uint64_t)(lo * std::pow(hi / lo, m / 4.0)), pt))
      local.push_back(pt);
  }
  local.push_back(pts[st.half + 1]);
  const double mid = 0.5 * (std::log2(st.before) + std::log2(st.after));
  double at = st.halfBytes;
  for (size_t q = 0; q + 1 < local.size(); q++)
  {
    const double y0 = std::log2(local[q].ratio()), y1 = std::log2(local[q + 1].ratio());
    if (y0 < mid && y1 >= mid)
    {
      const double f = (mid - y0) / std::max(y1 - y0, 1e-9);
      const double x0 = std::log2((double)local[q].bytes), x1 = std::log2((double)local[q + 1].bytes);
      at = std::exp2(x0 + f * (x1 - x0));
      break;
    }
  }
  return twoDigits(at);
}

// ── The probe ───────────────────────────────────────────────────────────────

namespace {

// One class of core: the cores of one rank.  `levels` are the levels its
// first core lists, up to the highest it measures; the steps are matched to
// them in order.
struct ProbeClass
{
  size_t   worker = 0;          // the pool worker pinned to its first core
  int      cpu    = -1;
  uint64_t rank   = 0;
  int      cores  = 0;
  int      maxLevel = 0;        // the highest level it measures an instance at
  std::vector<int> levels;
  bool     owns[4]  = {false, false, false, false};   // levels it measures an instance at
  uint64_t size[4]  = {0, 0, 0, 0};
  bool     found[4] = {false, false, false, false};
  std::string curve;            // the sweep, for --verbose
  std::string why;              // why it stopped short, when it did
};

} // namespace

void probeCacheSizes(cpu_device_info_t &info, CpuThreadPool &pool)
{
  std::vector<cpu_cache_instance_t> &inst = info.cacheInstances;
  if (inst.empty() || info.cores.empty() || (int)info.cores.size() != pool.maxThreads())
    return;

  // A size the platform had some other way for the ST core (sysconf) is the
  // OS's: its instance keeps it.
  const cpu_core_t &fast = info.cores.front();
  const uint64_t known[4] = {0, info.l1dCacheBytes, info.l2CacheBytes, info.l3CacheBytes};
  for (int lvl = 1; lvl <= 3; lvl++)
    if (fast.cacheInstance[lvl] >= 0 && !inst[(size_t)fast.cacheInstance[lvl]].bytes && known[lvl])
      inst[(size_t)fast.cacheInstance[lvl]].bytes = known[lvl];

  // Which class measures each unsized instance: the fastest that reads it.
  std::vector<ProbeClass> classes;
  std::vector<int> owner(inst.size(), -1);
  for (size_t w = 0; w < info.cores.size(); w++)
  {
    const cpu_core_t &c = info.cores[w];
    int ci = -1;
    for (size_t k = 0; k < classes.size(); k++)
      if (classes[k].rank == c.rank)
        ci = (int)k;
    if (ci < 0)
    {
      ProbeClass pc;
      pc.worker = w;
      pc.cpu    = c.id;
      pc.rank   = c.rank;
      classes.push_back(pc);
      ci = (int)classes.size() - 1;
    }
    classes[(size_t)ci].cores++;
    for (int lvl = 1; lvl <= 3; lvl++)
    {
      const int i = c.cacheInstance[lvl];
      if (i >= 0 && !inst[(size_t)i].bytes && owner[(size_t)i] < 0)
        owner[(size_t)i] = ci;
    }
  }
  bool any = false;
  for (size_t i = 0; i < inst.size(); i++)
    if (owner[i] >= 0)
    {
      ProbeClass &pc = classes[(size_t)owner[i]];
      pc.maxLevel = std::max(pc.maxLevel, inst[i].level);
      pc.owns[inst[i].level] = true;
      any = true;
    }
  if (!any)
    return;
  info.cacheProbed = true;

  // Far enough for the largest last-level caches there are -- a Grace's
  // 114 MB, a 3D V-Cache die's 96 MB -- before the sweep stops on its own,
  // which on a phone it does near 64 MB.
  const uint64_t cap = std::max<uint64_t>(clpeak::memoryBudget(1ull << 30, 8), 64ull << 20);
  const auto deadline = Clock::now() + std::chrono::seconds(60);

  for (ProbeClass &pc : classes)
  {
    if (!pc.maxLevel)
      continue;
    const cpu_core_t &rep = info.cores[pc.worker];
    for (int lvl = 1; lvl <= pc.maxLevel; lvl++)
      if (rep.cacheInstance[lvl] >= 0)
        pc.levels.push_back(lvl);
    const size_t needed = pc.levels.size();

    CacheWalker walker(cap);
    // One size on the class's core, both walks, the faster kept.  Timed again
    // if the app left the screen meanwhile, or the worker ran elsewhere.
    auto measureOn = [&](uint64_t bytes, CacheSweepPoint &pt) {
      for (int attempt = 0; attempt < 3; attempt++)
      {
        bool ok = false, there = true;
        CacheSweepPoint runs, random;
        pool.run((int)pc.worker + 1, [&](int tid) {
          if ((size_t)tid != pc.worker)
            return;
#if defined(__linux__)
          there = sched_getcpu() == pc.cpu;
#endif
          ok = walker.measure(bytes, kRun, runs) && walker.measure(bytes, 1, random);
#if defined(__linux__)
          there = there && sched_getcpu() == pc.cpu;
#endif
        });
        const bool held = clpeak::holdGate();
        if (clpeak::cancelRequested() || !ok)
          return false;
        if (there && !held)
        {
          pt = combine(runs, random);
          return true;
        }
      }
      pc.why = "its worker did not stay on cpu" + std::to_string(pc.cpu);
      return false;
    };

    // Up by quarter octaves, landing on every power of two, until the step
    // past each level has levelled off.
    std::vector<CacheSweepPoint> pts;
    for (int k = 0;; k++)
    {
      const uint64_t g =
          (uint64_t)std::llround(4096.0 * std::exp2(k / 4.0) / (double)kGrain) * kGrain;
      if (g > cap)
        break;
      if (!pts.empty() && g <= pts.back().bytes)
        continue;
      if (Clock::now() > deadline)
      {
        pc.why = "the sweep ran out of time";
        break;
      }
      CacheSweepPoint pt;
      if (!measureOn(g, pt))
        break;
      pts.push_back(pt);
      const std::vector<CacheStep> steps = findCacheSteps(pts);
      if (steps.size() >= needed && pts.size() >= steps[needed - 1].end + 4)
        break;
    }

    // The points about each step timed again, each keeping its faster pass.
    {
      std::vector<bool> again(pts.size(), false);
      const std::vector<CacheStep> first = findCacheSteps(pts);
      for (size_t s = 0; s < first.size() && s < needed; s++)
        for (size_t q = first[s].onset > 1 ? first[s].onset - 1 : 0;
             q <= first[s].end && q < pts.size(); q++)
          again[q] = true;
      for (size_t q = pts.size(); q-- > 0 && Clock::now() <= deadline;)
      {
        if (!again[q])
          continue;
        CacheSweepPoint pt;
        if (!measureOn(pts[q].bytes, pt))
          break;
        if (pt.ratio() < pts[q].ratio())
          pts[q] = pt;
      }
    }
    const std::vector<CacheStep> steps = findCacheSteps(pts);

    // Matched in order to the levels the core lists.  A step that cannot be
    // the level -- no larger than the one before, or outside any size that
    // level comes in -- ends the matching: the ones past it would be off by
    // one level.
    const uint64_t lowest[4]  = {0, 4ull << 10, 32ull << 10, 128ull << 10};
    const uint64_t highest[4] = {0, 1ull << 20, 256ull << 20, 4ull << 30};
    uint64_t below = 0;
    for (size_t s = 0; s < needed && s < steps.size() && Clock::now() <= deadline; s++)
    {
      const int lvl = pc.levels[s];
      const CacheStep &st = steps[s];
      // An L1d is virtually indexed: its sets fill evenly on any pages.
      const uint64_t b = levelSize(pts, st, lvl == 1 || walker.hugePages(), measureOn);
      if (b < lowest[lvl] || b > highest[lvl] || (below && b < below * 3 / 2))
      {
        pc.why = "a step at " + sizeText((double)b) + " cannot be its L" + std::to_string(lvl);
        break;
      }
      const cpu_cache_instance_t &own = inst[(size_t)rep.cacheInstance[lvl]];
      if (own.bytes)
      {
        // A level the OS sized: the step must agree with it.
        if (b < own.bytes / 2 || b > own.bytes * 2)
        {
          pc.why = "its L" + std::to_string(lvl) + " step at " + sizeText((double)b) +
                   " does not match the OS's " + sizeText((double)own.bytes);
          break;
        }
        below = own.bytes;
        continue;
      }
      pc.size[lvl]  = b;
      pc.found[lvl] = true;
      below = b;
    }

    // A level the OS says is the core's own, or its pair's, is checked to be:
    // a step missed below it -- a phone whose L3 is little slower than its L2
    // -- would have the L3's taken for it, and every level past it with it.
    // Half the size measured for it, timed alone and then while a core that
    // shares the core's last level but not this one walks twice that: a
    // working set in a private cache does not notice.
    if (pc.found[2] && rep.cacheInstance[2] >= 0 && Clock::now() <= deadline)
    {
      const cpu_cache_instance_t &own = inst[(size_t)rep.cacheInstance[2]];
      const int last = rep.cacheInstance[3];
      size_t other = info.cores.size();
      for (size_t w = 0; w < info.cores.size() && other == info.cores.size(); w++)
      {
        const cpu_core_t &c = info.cores[w];
        const bool inL2 =
            std::find(own.cpus.begin(), own.cpus.end(), c.id) != own.cpus.end();
        if (!inL2 && last >= 0 && c.cacheInstance[3] == last)
          other = w;
      }
      if (other < info.cores.size())
      {
        const uint64_t w = pc.size[2] / 2;
        CacheWalker neighbour(cap);
        std::atomic<bool> stop{false};
        bool ok = true;
        CacheSweepPoint alone, beside;
        ok = measureOn(w, alone);
        if (ok)
          pool.run((int)std::max(pc.worker, other) + 1, [&](int tid) {
            if ((size_t)tid == other)
              ok = neighbour.thrash(pc.size[2] * 2, stop) && ok;
            else if ((size_t)tid == pc.worker)
            {
              std::this_thread::sleep_for(std::chrono::milliseconds(5));
              CacheSweepPoint runs, random;
              ok = walker.measure(w, kRun, runs) && walker.measure(w, 1, random) && ok;
              beside = combine(runs, random);
              stop.store(true);
            }
          });
        if (ok && beside.ratio() > alone.ratio() * 1.25)
        {
          pc.why = "the step taken for its L2, at " + sizeText((double)pc.size[2]) +
                   ", is a cache it shares with cpu" + std::to_string(info.cores[other].id) +
                   " (" + std::to_string(beside.ratio() / alone.ratio()).substr(0, 4) +
                   "x slower beside it)";
          for (int lvl = 2; lvl <= 3; lvl++)
            pc.found[lvl] = false;
        }
      }
    }

    for (const CacheSweepPoint &p : pts)
    {
      char cell[48];
      std::snprintf(cell, sizeof(cell), " %s %.2f", sizeText((double)p.bytes).c_str(), p.ns);
      pc.curve += cell;
    }
  }

  for (size_t i = 0; i < inst.size(); i++)
    if (owner[i] >= 0 && classes[(size_t)owner[i]].found[inst[i].level])
    {
      inst[i].bytes    = classes[(size_t)owner[i]].size[inst[i].level];
      inst[i].measured = true;
    }
  refreshCacheSizes(info);

  // What was measured, on which core, in one line the run's log always has,
  // and the sweeps themselves under --verbose.
  static const char *const name[4] = {"", "L1d", "L2", "L3"};
  std::string said;
  for (const ProbeClass &pc : classes)
  {
    if (!pc.maxLevel)
      continue;
    std::string part = "cpu" + std::to_string(pc.cpu);
    if (pc.cores > 1)
      part += " (" + std::to_string(pc.cores) + " cores)";
    for (int lvl = 1; lvl <= 3; lvl++)
      if (pc.owns[lvl])
        part += std::string(" ") + name[lvl] + " " +
                (pc.found[lvl] ? sizeText((double)pc.size[lvl]) : std::string("not found"));
    said += (said.empty() ? "" : "; ") + part;
    CLPEAK_VLOG("[cpu] cache sweep on cpu%d, ns per load:%s\n", pc.cpu, pc.curve.c_str());
    if (!pc.why.empty())
      CLPEAK_VLOG("[cpu] cache sweep on cpu%d stopped short: %s\n", pc.cpu, pc.why.c_str());
  }
  CLPEAK_LOG(Info, "[cpu] the OS gives no size for some caches; measured where a dependent-load "
             "walk slows down: %s", said.c_str());
}

#endif // ENABLE_CPU
