#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include <common/run_document.h>
#include "cpu_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

// Cache-line-aligned float buffer.  std::vector / malloc only promise 16-byte
// alignment on x86-64, and glibc serves large allocations from mmap as
// page + 16 (the chunk header) -- the worst possible case: with a 16-mod-64
// start, HALF of all 32-byte AVX2 loads straddle a 64-byte cache line, and on
// Zen 2 a line-split load costs two accesses.  That alone caps an L1 read row
// near ~60% of the load-port bound.  64-byte alignment removes the splits for
// every vector width we issue (16/32/64 B).
struct AlignedFloats
{
  float *p = nullptr;
  size_t n = 0;
  AlignedFloats() = default;
  explicit AlignedFloats(size_t count) { alloc(count); }
  AlignedFloats(const AlignedFloats &) = delete;
  AlignedFloats &operator=(const AlignedFloats &) = delete;
  ~AlignedFloats() { free(); }

  void alloc(size_t count)
  {
    free();
    n = count;
    p = static_cast<float *>(::operator new(count * sizeof(float),
                                            std::align_val_t(64)));
  }
  void free()
  {
    if (p)
      ::operator delete(p, std::align_val_t(64));
    p = nullptr;
    n = 0;
  }
  float *data() const { return p; }
};

// The streaming-read kernel is ISA-dispatched (compiled per-ISA in
// cpu_kernels_tu.cpp).  Forward to the selected variant.
static inline uint64_t readBufferChecksum(const float *p, size_t M, uint64_t iters)
{
  return clpeak_cpu::kernels().readsum(p, M, iters);
}

// ---------------------------------------------------------------------------
// Cache bandwidth: read-only streaming.  1T uses one resident working set.
// MT keeps private levels resident per thread, but splits shared levels across
// all threads so the aggregate working set remains inside the target cache.
// ---------------------------------------------------------------------------
int CpuPeak::runCacheBandwidth(benchmark_config_t &cfg)
{
  logger::TestSpec spec{"cache_bandwidth", "Cache bandwidth (read)", "gbps",
                        Category::Bandwidth,
                        "How many bytes per second the CPU can read out of each "
                        "cache, from the tiny fast one next to the core to the big "
                        "slow one shared by all of them.  Each reading uses a working "
                        "set sized to stay inside the cache it names.",
                        TestShape::Heterogeneous, "cache level"};
  auto test = currentDeviceScope->beginTest(spec);

  const int maxT = pool->maxThreads();
  // Is the L2 a per-core private cache, or shared by a cluster?  Asked of the
  // topology, not of the vendor string: Apple is not the only one -- Qualcomm's
  // Oryon shares 12 MB across each cluster of 4, and Intel's E-core modules
  // share one L2 as well.  If every core had its own, the aggregate would be
  // per-core x cores; anything less means cores are sharing, and the MT row has
  // to split the working set or it overflows the level it names.
  const int cores = info.physicalCores > 0 ? info.physicalCores : maxT;
  const bool l2Shared = info.l2TotalBytes < info.l2CacheBytes * (uint64_t)cores;
  const uint64_t cap = 32ull * 1024 * 1024; // bound the per-thread allocation
  // Per-thread buffer must hold the largest single-thread working set we stream.
  // That is usually the L3 set, but on Apple Silicon the per-cluster L2 (e.g.
  // 12 MB) can exceed the reported/last-level cache, so size to the max of the
  // L2 and L3 sets, capped so the NT allocation stays bounded.
  uint64_t largestLevel = std::max<uint64_t>(info.l2CacheBytes / 2, info.l3CacheBytes / 2);
  uint64_t allocBytes = std::min<uint64_t>(std::max<uint64_t>(largestLevel, 65536), cap);
  size_t allocFloats = (size_t)(allocBytes / sizeof(float));
  if (allocFloats < 1024)
    allocFloats = 1024;

  std::vector<AlignedFloats> bufs((size_t)maxT);
  for (auto &b : bufs)
  {
    b.alloc(allocFloats);
    populate(b.data(), allocFloats);
  }

  std::vector<uint64_t> sink((size_t)maxT, 0);

  // Notes ride the table so a level's name and explanation stay adjacent.
  // `bytes` is the ST working set: half of ONE instance of the level, which is
  // all a single core can reach.  `totalBytes` is what the MT row splits across
  // threads -- the AGGREGATE of the level, which on a multi-instance cache is
  // not the same number.  Dividing the per-instance size by every thread in the
  // machine shrinks the slice by the instance count twice over: on a 16C/32T
  // Threadripper (4 CCX x 16 MB L3) it left 256 KB per thread, inside the
  // 512 KB per-core L2, and "L3 MT" came back at 1758 GB/s against "L2 MT" at
  // 1744 -- two different levels reporting the same bandwidth, which is the
  // tell.  0 means the level is absent and the row skips.
  // `mtFloor` is twice one instance of the level BELOW, and it is what keeps a
  // split slice from falling out of the level it names: divide too far and the
  // row quietly re-measures the faster cache underneath.  It also covers the
  // case where the aggregate could not be detected and fell back to the
  // per-instance size, which would otherwise divide a single instance by every
  // thread in the machine.
  struct Level
  {
    const char *name;
    uint64_t bytes;
    uint64_t totalBytes;
    uint64_t mtFloor;
    bool sharedForMt;
    const char *stNote;
    const char *mtNote;
  };
  const Level levels[] = {
      {"L1", std::max<uint64_t>(info.l1dCacheBytes / 2, 4096), info.l1dTotalBytes,
       0, false,
       "One thread reading from the small cache inside its own core.",
       "Every core reading from its own L1 at the same time."},
      {"L2", std::max<uint64_t>(info.l2CacheBytes / 2, 16384), info.l2TotalBytes,
       info.l1dCacheBytes * 2, l2Shared,
       "One thread reading from the mid-level cache, the next step out.",
       "Every core reading from L2 at once; where L2 is shared, the data is "
       "split between them so it still fits."},
      {"L3", info.l3CacheBytes
                 ? std::min<uint64_t>(std::max<uint64_t>(info.l3CacheBytes / 2, 65536), allocBytes)
                 : 0, // 0 = this CPU has no L3; the loop skips the row
       info.l3TotalBytes, info.l2CacheBytes * 2, true,
       "One thread reading from the large cache shared by all cores.",
       "Every core reading from that shared cache at once, each taking a slice "
       "of it."},
  };

  unsigned int forced = forceIters ? specifiedIters : 0;

  for (const auto &lvl : levels)
  {
    // A CPU with no L3 (Apple Silicon, Snapdragon X, most phone SoCs) gets the
    // row as Unsupported rather than a measurement: without a real size the
    // working set falls back to something that still fits in L2, and the row
    // silently reports L2 a second time.
    if (lvl.bytes == 0)
    {
      test.skip(std::string(lvl.name) + " ST", ResultStatus::Unsupported,
                "no L3 on this CPU", lvl.stNote);
      test.skip(std::string(lvl.name) + " MT", ResultStatus::Unsupported,
                "no L3 on this CPU", lvl.mtNote);
      continue;
    }

    size_t M1 = (size_t)(lvl.bytes / sizeof(float));
    if (M1 > allocFloats)
      M1 = allocFloats;
    if (M1 < 64)
      M1 = 64;

    uint64_t mtBytes =
        lvl.sharedForMt
            ? std::max<uint64_t>({lvl.totalBytes / 2 / (uint64_t)maxT, lvl.mtFloor, 4096})
            : lvl.bytes;
    size_t MN = (size_t)(mtBytes / sizeof(float));
    if (MN > allocFloats)
      MN = allocFloats;
    if (MN < 64)
      MN = 64;

    Workload body1 = [&](int tid, uint64_t iters)
    {
      sink[(size_t)tid] ^= readBufferChecksum(bufs[(size_t)tid].data(), M1, iters);
    };
    Workload bodyN = [&](int tid, uint64_t iters)
    {
      sink[(size_t)tid] ^= readBufferChecksum(bufs[(size_t)tid].data(), MN, iters);
    };

    double us1 = runWorkload(1, body1, cfg.targetTimeUs, forced);
    double usN = runWorkload(maxT, bodyN, cfg.targetTimeUs, forced);

    double stPassBytes = (double)M1 * sizeof(float);
    double mtPassBytes = (double)MN * sizeof(float) * (double)maxT;
    auto gbps = [](double bytes, double meanUs) -> float
    {
      return meanUs > 0.0 ? (float)(bytes / (meanUs * 1e3)) : -1.0f;
    };

    if (us1 > 0)
      test.emit(std::string(lvl.name) + " ST", gbps(stPassBytes, us1), lvl.stNote);
    else
      test.skip(std::string(lvl.name) + " ST", ResultStatus::Error, "read failed", lvl.stNote);
    if (usN > 0)
      test.emit(std::string(lvl.name) + " MT", gbps(mtPassBytes, usN), lvl.mtNote);
    else
      test.skip(std::string(lvl.name) + " MT", ResultStatus::Error, "read failed", lvl.mtNote);
  }

  volatile uint64_t keep = 0;
  for (uint64_t s : sink)
    keep ^= s;
  (void)keep;

  // Close the read test BEFORE opening the write/copy one: LoggerText buffers
  // metric rows until TestEnd, and a nested TestBegin clears that buffer --
  // overlapping TestScopes silently drop the first test's rows.
  test.end();

  // ---- L1 write / copy: the store-port side of the story ------------------
  // The read rows above measure the load ports; write and copy expose the
  // store-port width and the load:store split (e.g. NVIDIA Olympus is 4 load
  // + 2 store pipes x 128-bit, so write lands near half of read).  The
  // kernels are the ISA-dispatched vector stores from base_compute.h, NOT
  // libc memset/memcpy: those switch to non-temporal stores above a size
  // threshold and would bypass the very cache under test.
  {
    logger::TestSpec wspec{"l1_write_bandwidth", "L1 bandwidth (write / copy)",
                           "gbps", Category::Bandwidth,
                           "How many bytes per second a core can write into its "
                           "nearest cache, and copy within it.  Cores have fewer "
                           "paths out to memory than in, so writing usually lands "
                           "below the matching read row.",
                           TestShape::Heterogeneous, "operation"};
    auto wtest = currentDeviceScope->beginTest(wspec);

    // A QUARTER of the L1, not half like the read row: on a hybrid chip
    // l1dCacheBytes is the big core's (Apple reports hw.perflevel0), so half
    // of it is the *entire* L1 of the small cores -- 64 KB vs the E-core's
    // 64 KB on M1 Pro -- and the MT row would measure L2 write drain on those
    // threads.  A quarter is resident on both core types.  Reads tolerate the
    // overflow (they scale ~7.7x either way); stores do not.
    size_t wFloats = (size_t)(std::max<uint64_t>(info.l1dCacheBytes / 4, 4096) / sizeof(float));
    if (wFloats > allocFloats)
      wFloats = allocFloats;
    size_t cFloats = wFloats / 2; // src [cFloats, 2*cFloats) + dst [0, cFloats)

    Workload writeBody = [&](int tid, uint64_t iters)
    {
      clpeak_cpu::kernels().writefill(bufs[(size_t)tid].data(), wFloats, iters);
    };
    Workload copyBody = [&](int tid, uint64_t iters)
    {
      float *p = bufs[(size_t)tid].data();
      clpeak_cpu::kernels().copybuf(p, p + cFloats, cFloats, iters);
    };

    auto gbps = [](double bytes, double meanUs) -> float
    {
      return meanUs > 0.0 ? (float)(bytes / (meanUs * 1e3)) : -1.0f;
    };

    const char *wStNote = "One thread storing new values into its own L1.";
    const char *wMtNote = "Every core storing into its own L1 at the same time.";
    const char *cStNote = "One thread copying inside L1 -- one read plus one write "
                          "for every byte moved.";
    const char *cMtNote = "Every core copying inside its own L1 at the same time.";

    double us1 = runWorkload(1, writeBody, cfg.targetTimeUs, forced);
    double usN = runWorkload(maxT, writeBody, cfg.targetTimeUs, forced);
    if (us1 > 0)
      wtest.emit("write ST", gbps((double)wFloats * sizeof(float), us1), wStNote);
    else
      wtest.skip("write ST", ResultStatus::Error, "write failed", wStNote);
    if (usN > 0)
      wtest.emit("write MT", gbps((double)wFloats * sizeof(float) * maxT, usN), wMtNote);
    else
      wtest.skip("write MT", ResultStatus::Error, "write failed", wMtNote);

    us1 = runWorkload(1, copyBody, cfg.targetTimeUs, forced);
    usN = runWorkload(maxT, copyBody, cfg.targetTimeUs, forced);
    if (us1 > 0)
      wtest.emit("copy ST", gbps(2.0 * cFloats * sizeof(float), us1), cStNote);
    else
      wtest.skip("copy ST", ResultStatus::Error, "copy failed", cStNote);
    if (usN > 0)
      wtest.emit("copy MT", gbps(2.0 * cFloats * sizeof(float) * maxT, usN), cMtNote);
    else
      wtest.skip("copy MT", ResultStatus::Error, "copy failed", cMtNote);
  }
  return 0;
}

// Number of floats per STREAM array.  Must exceed *every cache the stream can
// land in*, summed — not the L3 alone.  Two ways that goes wrong if you size
// off L3 by name: on multi-CCX/CCD AMD the per-instance L3 is only a slice, and
// on a chip whose last level IS the L2 (Apple Silicon, Snapdragon X, most ARM
// parts) there is no L3 to size off at all.  A Snapdragon X Elite has 36 MB of
// aggregate L2 and reports no L3, so the old L3-only rule left the array at the
// 64 MB floor — under 2x the cache — and the "DRAM" read row came back at
// 149 GB/s on memory whose theoretical peak is 135.  Under-sizing never fails
// loudly: it just serves part of the read out of cache and reports a number
// above what the DIMMs can physically do.  4x total cache is the classic STREAM
// margin; the cap keeps us from hogging memory.  Even split across threads.
static size_t pickStreamFloats(const cpu_device_info_t &info, int maxT)
{
  uint64_t cache = info.l1dTotalBytes + info.l2TotalBytes +
                   std::max(info.l3TotalBytes, info.l3CacheBytes);
  uint64_t arrayBytes = std::max<uint64_t>(cache * 4, 64ull << 20);
  uint64_t cap = info.totalMemBytes ? std::min<uint64_t>(512ull << 20, info.totalMemBytes / 16)
                                    : (512ull << 20);
  if (cap < cache * 2)
    cap = cache * 2; // always large enough to miss every level
  arrayBytes = std::min(arrayBytes, cap);
  size_t N = (size_t)(arrayBytes / sizeof(float));
  N = (N / (size_t)maxT) * (size_t)maxT;
  if (N < (size_t)maxT)
    N = (size_t)maxT;
  return N;
}

// ---------------------------------------------------------------------------
// DRAM bandwidth: STREAM-style read / copy / triad over shared arrays far
// larger than the LLC, partitioned across all cores.  Arrays are allocated
// untouched and first-touched in parallel so their pages land on the NUMA node
// of the thread that will use them (single-threaded init would place every page
// on one node and cripple bandwidth on multi-socket / multi-CCD systems).
// ---------------------------------------------------------------------------
int CpuPeak::runDramBandwidth(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
      {"global_memory_bandwidth", "DRAM bandwidth", "gbps", Category::Unknown,
       "How many bytes per second all cores together can move to and from main "
       "memory.  The arrays are far too big for any cache, so every access goes "
       "out to RAM.  The two rows that write count only the bytes the program "
       "asked for, the usual STREAM convention; most CPUs must also fetch each "
       "line before overwriting it, so copy and triad move about half again as "
       "much as they count and normally land below the read row.",
       TestShape::Heterogeneous, "operation"});

  const int maxT = pool->maxThreads();
  const size_t N = pickStreamFloats(info, maxT);
  // The one number that decides whether this test measures DRAM at all, so it
  // is worth being able to read it back off a suspicious run: a "DRAM" figure
  // above the memory's rated peak means the array was not big enough.
  CLPEAK_VLOG("[cpu] STREAM array %llu MB x3, %llu MB total cache, %d threads\n",
              (unsigned long long)((uint64_t)N * sizeof(float) >> 20),
              (unsigned long long)((info.l1dTotalBytes + info.l2TotalBytes +
                                    std::max(info.l3TotalBytes, info.l3CacheBytes)) >> 20),
              maxT);

  auto chunk = [&](int tid, size_t &lo, size_t &hi)
  {
    size_t per = N / (size_t)maxT;
    lo = (size_t)tid * per;
    hi = (tid == maxT - 1) ? N : lo + per;
  };

  // `new float[N]` leaves the pages untouched (floats are not value-initialized),
  // so the parallel populate below is the first touch.
  // Aligned like the cache buffers (see AlignedFloats): operator new with an
  // alignment leaves the pages untouched, so the parallel first-touch below is
  // still what places them NUMA-locally.
  AlignedFloats Abuf(N), Bbuf(N), Cbuf(N);
  float *A = Abuf.data();
  float *B = Bbuf.data();
  float *C = Cbuf.data();
  pool->run(maxT, [&](int tid)
            {
    size_t lo, hi; chunk(tid, lo, hi);
    populate(A + lo, hi - lo);
    populate(B + lo, hi - lo);
    populate(C + lo, hi - lo); });

  std::vector<uint64_t> sink((size_t)maxT, 0);
  unsigned int forced = forceIters ? specifiedIters : 0;
  auto gbps = [](double bytes, double meanUs) -> float
  {
    return meanUs > 0.0 ? (float)(bytes / (meanUs * 1e3)) : -1.0f;
  };

  {
    Workload body = [&](int tid, uint64_t iters)
    {
      size_t lo, hi;
      chunk(tid, lo, hi);
      sink[(size_t)tid] ^= readBufferChecksum(A + lo, hi - lo, iters);
    };
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("read", gbps((double)N * sizeof(float), us),
              "Reading one large array straight through, start to end.");
  }
  {
    Workload body = [&](int tid, uint64_t iters)
    {
      size_t lo, hi;
      chunk(tid, lo, hi);
      for (uint64_t it = 0; it < iters; it++)
        std::memcpy(A + lo, C + lo, (hi - lo) * sizeof(float));
    };
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("copy", gbps(2.0 * N * sizeof(float), us),
              "Copying one large array into another -- a read and a write for "
              "every element.");
  }
  {
    const float s = 1.5f;
    Workload body = [&](int tid, uint64_t iters)
    {
      size_t lo, hi;
      chunk(tid, lo, hi);
      for (uint64_t it = 0; it < iters; it++)
        for (size_t i = lo; i < hi; i++)
          A[i] = B[i] + s * C[i];
    };
    double us = runWorkload(maxT, body, cfg.targetTimeUs, forced);
    test.emit("triad", gbps(3.0 * N * sizeof(float), us),
              "Scaling one array, adding a second and storing to a third: two "
              "reads and a write per element, the hardest of the three.");
  }

  volatile uint64_t keep = 0;
  for (uint64_t v : sink)
    keep ^= v;
  (void)keep;
  return 0;
}

#endif // ENABLE_CPU
