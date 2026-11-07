#ifndef CLPEAK_CPU_CACHE_PROBE_H
#define CLPEAK_CPU_CACHE_PROBE_H

// The latency sweep behind probeCacheSizes() (include/cpu/cpu_peak.h): a
// dependent-load walk timed over working sets from 4 KB up, and the steps in
// it -- where the walk falls out of one cache into the next.  Separate from
// the probe's orchestration so the two halves can be exercised on their own.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <vector>

// One point of the sweep: ns per dependent load over `bytes`, and over a 4 KB
// set timed either side of it -- the core's L1 at the clock it ran at then.
struct CacheSweepPoint
{
  uint64_t bytes = 0;
  double   ns    = 0.0;
  double   refNs = 0.0;

  double ratio() const { return refNs > 0.0 ? ns / refNs : 0.0; }
};

// One rise of the curve, from one level to the next, by sweep point index:
// it starts after `onset` -- the last point still on the level below -- is
// halfway, in log latency, between `half` and `half + 1`, and has levelled
// off by `end`.  `before` / `after` are the latencies against the reference
// either side of it.
struct CacheStep
{
  size_t onset = 0, half = 0, end = 0;
  double halfBytes = 0.0;       // where it is halfway, interpolated
  double before = 0.0, after = 0.0;
};

// Times walks on the calling thread, over one buffer it grows as the sweep
// does.  Not thread-safe; first-touched by the thread that measures, so the
// pages are that core's NUMA node's.
class CacheWalker
{
public:
  explicit CacheWalker(uint64_t maxBytes);
  ~CacheWalker();
  CacheWalker(const CacheWalker &) = delete;
  CacheWalker &operator=(const CacheWalker &) = delete;

  // One point, its nodes visited `run` at a time inside a 4 KB region (1 = a
  // walk random across the whole set); false when its memory could not be had.
  bool measure(uint64_t bytes, unsigned run, CacheSweepPoint &out);

  // Walk `bytes` at random until `stop` -- another core's working set, to see
  // whether the one measured shares a cache with it.  False when its memory
  // could not be had.
  bool thrash(uint64_t bytes, const std::atomic<bool> &stop);

  // Whether the kernel backs the walk with huge pages (Linux transparent huge
  // pages): then every cache's sets fill evenly, and each step is sharp.
  bool hugePages() const { return huge; }

private:
  bool     reserve(uint64_t bytes);
  // Lay a walk over `bytes` of `base`; the offset of one of its nodes.
  uint32_t build(uint8_t *base, uint64_t bytes, unsigned run);
  // ns per load of the walk from node `from`, warmed first -- `capNs` bounds
  // the warm-up by the rate of the point before (0 = none) -- as the fastest
  // of slices of ~`sliceNs`, `minNs` of them in all.
  double   timeWalk(const uint8_t *base, uint64_t bytes, uint32_t from, double capNs,
                    double sliceNs, double minNs);

  uint64_t maxBytes;
  uint8_t *region      = nullptr;  // the mapping, as allocated
  uint8_t *buf         = nullptr;  // `region` aligned to 2 MB
  uint64_t capacity    = 0;
  uint64_t regionBytes = 0;
  uint8_t *ref         = nullptr;  // the 4 KB reference walk
  uint32_t refStart    = 0;
  bool     huge        = false;
  double   lastNs      = 0.0;      // the point before's ns per load
  std::mt19937_64       rng;
  std::vector<uint32_t> order, runs;
};

// The rises of a sweep, smallest working set first: each at least 1.4x, so
// an L1 into an L2 or an L3 into DRAM, never a TLB or a clock's wobble.
std::vector<CacheStep> findCacheSteps(const std::vector<CacheSweepPoint> &pts);

// The size of the level `st` rises out of, refined with three more points
// taken through `measure`.  `even`: the level's sets fill evenly -- an L1d,
// which is virtually indexed, or any level on huge pages -- so it ends where
// the rise begins; otherwise its size is where the rise is halfway.
uint64_t levelSize(const std::vector<CacheSweepPoint> &pts, const CacheStep &st, bool even,
                   const std::function<bool(uint64_t, CacheSweepPoint &)> &measure);

#endif // CLPEAK_CPU_CACHE_PROBE_H
