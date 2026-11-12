#include <opencl/cl_peak.h>
#include <opencl/cl_utils.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>

#if defined(_WIN32) || defined(__ANDROID__)
#include <malloc.h>
#endif

using xpupeak::Dim; namespace dim = xpupeak::dim;

// Platform-specific aligned alloc/free, for the pageable fallback below.
static float *allocAligned(size_t bytes)
{
#if defined(_WIN32)
  return static_cast<float *>(_aligned_malloc(bytes, 64));
#elif defined(__ANDROID__)
  return static_cast<float *>(memalign(64, bytes));
#else
  return static_cast<float *>(aligned_alloc(64, bytes));
#endif
}

static void freeAligned(float *ptr)
{
  if (!ptr) return;
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif
}

int clPeak::runTransferBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  UNUSED(prog);

  auto run = admit(Benchmark::TransferBW);
  if (!run)
    return 0;

  cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
  unsigned int forced = forceIters ? specifiedIters : 0;

  // Two live allocations of this size (device buffer + host staging buffer),
  // so cap at half the largest single allocation the device admits to.
  uint64_t bytes = cfg.transferBWMaxSize ? cfg.transferBWMaxSize : (1ull << 27);
  if (devInfo.maxAllocSize > 0)
    bytes = std::min<uint64_t>(bytes, devInfo.maxAllocSize / 2);
  bytes &= ~255ull;
  if (bytes == 0)
    bytes = 256;

  auto test = currentDeviceScope->beginTest(
    {"transfer_bandwidth", "Transfer bandwidth", "bps", Category::Unknown,
     "Copy bandwidth between host and device memory.",
     TestShape::Heterogeneous, "direction"});

  // Host staging memory.  The pinned path is a CL_MEM_ALLOC_HOST_PTR buffer
  // mapped once and left mapped: that pointer is host memory the driver may
  // DMA from directly, which is what cuMemAllocHost/hipHostMalloc give the
  // other backends.  Drivers that refuse to map it fall back to an ordinary
  // (pageable) aligned allocation, which measures the same route more slowly
  // -- and is another measurement, so it reads as rows of its own,
  // `memory: pageable`, never as pinned ones.
  cl::Buffer pinnedBuf;
  void *hostPtr = nullptr;
  float *pageable = nullptr;
  bool pinned = true;

  struct Row { const char *id; const char *note; Dims dims; };
  auto rows = [&](dim::Direction direction) -> Row
  {
    const bool h2d = direction == dim::Direction::H2d;
    return {pinned ? (h2d ? "h2d_pinned" : "d2h_pinned")
                   : (h2d ? "h2d_pageable" : "d2h_pageable"),
            pinned ? (h2d ? "Host to device." : "Device to host.")
                   : (h2d ? "Host to device, from pageable memory: the driver would not "
                            "map a pinned buffer."
                          : "Device to host, into pageable memory: the driver would not "
                            "map a pinned buffer."),
            {{Dim::Direction, direction},
             {Dim::Memory, pinned ? dim::Memory::Pinned : dim::Memory::Pageable}}};
  };
  auto skipBoth = [&](const std::string &reason)
  {
    for (dim::Direction direction : {dim::Direction::H2d, dim::Direction::D2h})
    {
      const Row r = rows(direction);
      test.skip(r.id, ResultStatus::Error, reason, r.note, r.dims);
    }
  };

  auto releaseHost = [&]() {
    if (hostPtr && pageable == nullptr)
    {
      try { queue.enqueueUnmapMemObject(pinnedBuf, hostPtr); queue.finish(); }
      catch (cl::Error &) {}
    }
    freeAligned(pageable);
  };

  try
  {
    cl::Buffer devBuf = cl::Buffer(ctx, CL_MEM_READ_WRITE, bytes);

    try
    {
      pinnedBuf = cl::Buffer(ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, bytes);
      hostPtr = queue.enqueueMapBuffer(pinnedBuf, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bytes);
      queue.finish();
    }
    catch (cl::Error &error)
    {
      XPUPEAK_VLOG("Pinned host buffer unavailable (%s, %d); falling back to pageable memory\n",
                   error.what(), error.err());
      hostPtr = nullptr;
    }

    if (!hostPtr)
    {
      pageable = allocAligned(bytes);
      hostPtr = pageable;
    }
    if (!hostPtr)
    {
      // Neither kind of host memory: the pinned rows, which are what the
      // test is for, are the ones missing.
      skipBoth("Out of memory");
      return -1;
    }
    pinned = pageable == nullptr;

    // Pseudo-random contents, to defeat hardware memory compression on both
    // directions.
    populate((float *)hostPtr, bytes / sizeof(float));

    // Warm up, probe one iteration to size the measurement window at
    // ~cfg.targetTimeUs, then run.  The copies are enqueued non-blocking and
    // the batch is drained once, so the driver can keep the link busy -- the
    // same shape as the async memcpy loops in the CUDA/ROCm/oneAPI backends.
    //
    // Always wall-clock: CL profiling events time the device's command
    // processing, which on a unified-memory device is near zero for a copy that
    // moves nothing (Apple M1 reports ~70x the real rate).  The host clock over
    // a ~targetTimeUs window is both accurate enough and the number a caller
    // actually pays.  oneAPI times this the same way.
    auto runTransfer = [&](std::function<void()> op) -> float
    {
      for (unsigned int w = 0; w < warmupCount; w++)
      {
        op();
        queue.finish();
      }

      auto runBatch = [&](unsigned int n) -> float {
        auto t1 = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < n; i++)
          op();
        queue.finish();
        auto t2 = std::chrono::high_resolution_clock::now();
        return (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      };

      float probeUs = runBatch(1);
      unsigned int iters = pickIters((double)probeUs, cfg.targetTimeUs, forced);
      float timed = runBatch(iters) / static_cast<float>(iters);
      if (timed <= 0.0f)
        return -1.0f;
      return (float)bytes / timed * 1e6f;
    };

    float bw;

    const Row h2d = rows(dim::Direction::H2d);
    bw = runTransfer(
      [&]() { queue.enqueueWriteBuffer(devBuf, CL_FALSE, 0, bytes, hostPtr); });
    if (bw > 0.0f) test.emit(h2d.id, bw, h2d.note, h2d.dims);
    else           test.skip(h2d.id, ResultStatus::Error, "timer returned zero", h2d.note, h2d.dims);

    const Row d2h = rows(dim::Direction::D2h);
    bw = runTransfer(
      [&]() { queue.enqueueReadBuffer(devBuf, CL_FALSE, 0, bytes, hostPtr); });
    if (bw > 0.0f) test.emit(d2h.id, bw, d2h.note, d2h.dims);
    else           test.skip(d2h.id, ResultStatus::Error, "timer returned zero", d2h.note, d2h.dims);

    releaseHost();
  }
  catch (cl::Error &error)
  {
    skipBoth(std::string(error.what()) + " (" + std::to_string(error.err()) + ")");

    releaseHost();
    return -1;
  }

  return 0;
}
