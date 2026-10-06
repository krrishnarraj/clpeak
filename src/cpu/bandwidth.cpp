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
  logger::TestSpec spec{"cache_bandwidth", "Cache bandwidth (read)", "bps",
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
    int level;
    uint64_t bytes;
    uint64_t totalBytes;
    uint64_t mtFloor;
    bool sharedForMt;
    const char *stNote;
    const char *mtNote;
  };
  const Level levels[] = {
      {"L1", 1, std::max<uint64_t>(info.l1dCacheBytes / 2, 4096), info.l1dTotalBytes,
       0, false,
       "One thread reading from the small cache inside its own core.",
       "Every core reading from its own L1 at the same time."},
      {"L2", 2, std::max<uint64_t>(info.l2CacheBytes / 2, 16384), info.l2TotalBytes,
       info.l1dCacheBytes * 2, l2Shared,
       "One thread reading from the mid-level cache, the next step out.",
       "Every core reading from L2 at once; where L2 is shared, the data is "
       "split between them so it still fits."},
      {"L3", 3, info.l3CacheBytes
                 ? std::min<uint64_t>(std::max<uint64_t>(info.l3CacheBytes / 2, 65536), cap)
                 : 0, // 0 = this CPU has no L3 (or none with a size); the loop skips the row
       info.l3TotalBytes, info.l2CacheBytes * 2, true,
       "One thread reading from the large cache shared by all cores.",
       "Every core reading from that shared cache at once, each taking a slice "
       "of it."},
  };

  // Every MT worker's working set at `lvl`, sized for the core it is pinned to.
  // One size for every thread was the old rule, and on a chip whose cores
  // differ no single size fits: sized for the fastest core -- the one the
  // header describes -- a little core's L1 or L2 overflows, and sized for the
  // little core, a big core's "L2" slice can sit in its L1.  So each worker
  // takes half its share of the instance its own core reads (the instance over
  // the workers reading it -- an SMT sibling counts, or two threads each fill
  // half of one core's L1 and together all of it), and never less than twice
  // its share of the level below, or the row re-measures that faster cache.
  // Where threads are not pinned (macOS), or the OS gave no size for a core,
  // the uniform rule stands: the table's split above.
  auto mtSlices = [&](const Level &lvl) {
    const uint64_t uniform =
        lvl.sharedForMt
            ? std::max<uint64_t>({lvl.totalBytes / 2 / (uint64_t)maxT, lvl.mtFloor, 4096})
            : lvl.bytes;
    std::vector<uint64_t> out((size_t)maxT, uniform);
    if ((int)info.cores.size() != maxT)
      return out;
    for (int t = 0; t < maxT; t++)
    {
      const cpu_core_t &c = info.cores[(size_t)t];
      const uint64_t share[4] = {0, c.l1dBytes / (uint64_t)c.l1dSharers,
                                 c.l2Bytes / (uint64_t)c.l2Sharers,
                                 c.l3Bytes / (uint64_t)c.l3Sharers};
      if (!share[lvl.level] || (lvl.level > 1 && !share[lvl.level - 1]))
        continue;
      out[(size_t)t] = std::max<uint64_t>(
          {share[lvl.level] / 2, lvl.level > 1 ? 2 * share[lvl.level - 1] : 0, 4096});
    }
    return out;
  };
  std::vector<uint64_t> mtBytes[3];
  for (int i = 0; i < 3; i++)
    mtBytes[i] = mtSlices(levels[i]);

  // Per-thread buffer must hold the largest working set any thread streams.
  // That is usually the L3 set, but on Apple Silicon the per-cluster L2 (e.g.
  // 12 MB) can exceed the reported/last-level cache, so size to the max of the
  // L2 and L3 sets, capped so the NT allocation stays bounded.
  uint64_t largestLevel = std::max<uint64_t>(info.l2CacheBytes / 2, info.l3CacheBytes / 2);
  for (int i = 0; i < 3; i++)
    if (levels[i].bytes)
      for (uint64_t b : mtBytes[i])
        largestLevel = std::max(largestLevel, b);
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
  unsigned int forced = forceIters ? specifiedIters : 0;
  auto floatsIn = [allocFloats](uint64_t bytes) {
    return std::min(std::max<size_t>((size_t)(bytes / sizeof(float)), 64), allocFloats);
  };

  for (int i = 0; i < 3; i++)
  {
    const Level &lvl = levels[i];
    // A CPU with no L3 (Apple Silicon, Snapdragon X, most phone SoCs) gets the
    // row as Unsupported rather than a measurement: without a real size the
    // working set falls back to something that still fits in L2, and the row
    // silently reports L2 a second time.  The same goes for an L3 the OS lists
    // without a size.
    if (lvl.bytes == 0)
    {
      const char *why = info.l3Unsized ? "the OS lists an L3 but gives no size for it"
                                       : "no L3 on this CPU";
      test.skip(std::string(lvl.name) + " ST", ResultStatus::Unsupported, why, lvl.stNote);
      test.skip(std::string(lvl.name) + " MT", ResultStatus::Unsupported, why, lvl.mtNote);
      continue;
    }

    const size_t M1 = floatsIn(lvl.bytes);
    std::vector<size_t> MN((size_t)maxT);
    std::vector<double> mtPassBytes((size_t)maxT);
    for (int t = 0; t < maxT; t++)
    {
      MN[(size_t)t] = floatsIn(mtBytes[i][(size_t)t]);
      mtPassBytes[(size_t)t] = (double)MN[(size_t)t] * sizeof(float);
    }

    Workload body1 = [&](int tid, uint64_t iters)
    {
      sink[(size_t)tid] ^= readBufferChecksum(bufs[(size_t)tid].data(), M1, iters);
    };
    Workload bodyN = [&](int tid, uint64_t iters)
    {
      sink[(size_t)tid] ^= readBufferChecksum(bufs[(size_t)tid].data(), MN[(size_t)tid], iters);
    };

    // Passes per second; the MT one already weighted into bytes per second.
    double ps1 = runWorkload(1, body1, cfg.targetTimeUs, forced);
    double bpsN = runWorkload(maxT, bodyN, cfg.targetTimeUs, forced, &mtPassBytes);

    if (ps1 > 0)
      test.emit(std::string(lvl.name) + " ST", (float)(ps1 * (double)M1 * sizeof(float)),
                lvl.stNote);
    else
      test.skip(std::string(lvl.name) + " ST", ResultStatus::Error, "read failed", lvl.stNote);
    if (bpsN > 0)
      test.emit(std::string(lvl.name) + " MT", (float)bpsN, lvl.mtNote);
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
                           "bps", Category::Bandwidth,
                           "How many bytes per second a core can write into its "
                           "nearest cache, and copy within it.  Cores have fewer "
                           "paths out to memory than in, so writing usually lands "
                           "below the matching read row.",
                           TestShape::Heterogeneous, "operation"};
    auto wtest = currentDeviceScope->beginTest(wspec);

    // A QUARTER of the L1, not half like the read row, and of each worker's own
    // core: stores do not tolerate the overflow reads do (those scale ~7.7x
    // either way), and a quarter stays resident with an SMT sibling writing
    // its own quarter beside it.  Where threads are not pinned the quarter is
    // the ST core's -- on Apple the P-core's, whose quarter (32 KB on M1 Pro)
    // still fits the E-core's 64 KB L1 that half of it would fill.
    auto quarterFloats = [allocFloats](uint64_t l1d) {
      return std::min((size_t)(std::max<uint64_t>(l1d / 4, 4096) / sizeof(float)), allocFloats);
    };
    const size_t wFloats1 = quarterFloats(info.l1dCacheBytes);
    std::vector<size_t> wFloats((size_t)maxT, wFloats1);
    if ((int)info.cores.size() == maxT)
      for (int t = 0; t < maxT; t++)
        if (info.cores[(size_t)t].l1dBytes)
          wFloats[(size_t)t] = quarterFloats(info.cores[(size_t)t].l1dBytes);
    // copy: src [c, 2c) + dst [0, c), with c = half the write set
    std::vector<double> wBytes((size_t)maxT), cBytes((size_t)maxT);
    for (int t = 0; t < maxT; t++)
    {
      wBytes[(size_t)t] = (double)wFloats[(size_t)t] * sizeof(float);
      cBytes[(size_t)t] = 2.0 * (double)(wFloats[(size_t)t] / 2) * sizeof(float);
    }

    // `n` is the set each worker streams: the ST core's quarter for the
    // single-thread rows (worker 0 runs them), each worker's own for MT.
    auto writeBody = [&](const std::vector<size_t> &n) {
      return Workload([&bufs, &n](int tid, uint64_t iters) {
        clpeak_cpu::kernels().writefill(bufs[(size_t)tid].data(), n[(size_t)tid], iters);
      });
    };
    auto copyBody = [&](const std::vector<size_t> &n) {
      return Workload([&bufs, &n](int tid, uint64_t iters) {
        const size_t c = n[(size_t)tid] / 2;
        float *p = bufs[(size_t)tid].data();
        clpeak_cpu::kernels().copybuf(p, p + c, c, iters);
      });
    };
    const std::vector<size_t> wFloatsST((size_t)maxT, wFloats1);

    const char *wStNote = "One thread storing new values into its own L1.";
    const char *wMtNote = "Every core storing into its own L1 at the same time.";
    const char *cStNote = "One thread copying inside L1 -- one read plus one write "
                          "for every byte moved.";
    const char *cMtNote = "Every core copying inside its own L1 at the same time.";

    double ps1  = runWorkload(1, writeBody(wFloatsST), cfg.targetTimeUs, forced);
    double bpsN = runWorkload(maxT, writeBody(wFloats), cfg.targetTimeUs, forced, &wBytes);
    if (ps1 > 0)
      wtest.emit("write ST", (float)(ps1 * (double)wFloats1 * sizeof(float)), wStNote);
    else
      wtest.skip("write ST", ResultStatus::Error, "write failed", wStNote);
    if (bpsN > 0)
      wtest.emit("write MT", (float)bpsN, wMtNote);
    else
      wtest.skip("write MT", ResultStatus::Error, "write failed", wMtNote);

    ps1  = runWorkload(1, copyBody(wFloatsST), cfg.targetTimeUs, forced);
    bpsN = runWorkload(maxT, copyBody(wFloats), cfg.targetTimeUs, forced, &cBytes);
    if (ps1 > 0)
      wtest.emit("copy ST", (float)(ps1 * 2.0 * (double)(wFloats1 / 2) * sizeof(float)), cStNote);
    else
      wtest.skip("copy ST", ResultStatus::Error, "copy failed", cStNote);
    if (bpsN > 0)
      wtest.emit("copy MT", (float)bpsN, cMtNote);
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
      {"global_memory_bandwidth", "DRAM bandwidth", "bps", Category::Unknown,
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
  // Each thread keeps to its own slice -- the pages it first-touched -- and
  // the runner weighs a pass by that slice's bytes, so a core that makes more
  // passes than its neighbours counts for what it moved.  `streams` is how
  // many bytes move per element: 1 read, 2 copy, 3 triad.
  auto sliceBytes = [&](double streams) {
    std::vector<double> w((size_t)maxT);
    for (int t = 0; t < maxT; t++)
    {
      size_t lo, hi;
      chunk(t, lo, hi);
      w[(size_t)t] = streams * (double)(hi - lo) * sizeof(float);
    }
    return w;
  };
  auto emitBps = [&](const char *id, double bps, const char *note) {
    if (bps > 0) test.emit(id, (float)bps, note);
    else         test.skip(id, ResultStatus::Error, "workload failed", note);
  };

  {
    Workload body = [&](int tid, uint64_t iters)
    {
      size_t lo, hi;
      chunk(tid, lo, hi);
      sink[(size_t)tid] ^= readBufferChecksum(A + lo, hi - lo, iters);
    };
    const std::vector<double> w = sliceBytes(1.0);
    emitBps("read", runWorkload(maxT, body, cfg.targetTimeUs, forced, &w),
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
    const std::vector<double> w = sliceBytes(2.0);
    emitBps("copy", runWorkload(maxT, body, cfg.targetTimeUs, forced, &w),
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
    const std::vector<double> w = sliceBytes(3.0);
    emitBps("triad", runWorkload(maxT, body, cfg.targetTimeUs, forced, &w),
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
