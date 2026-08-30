#include <opencl/cl_peak.h>
#include <opencl/cl_utils.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>

#if defined(_WIN32) || defined(__ANDROID__)
#include <malloc.h>
#endif

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

  if (!isAllowed(Benchmark::TransferBW))
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
    {"transfer_bandwidth", "Transfer bandwidth", "gbps", Category::Unknown,
     "How fast data crosses between the host's memory and the device's.  On a "
     "discrete card that means the PCIe link, which is far narrower than "
     "either side's own memory and is what makes moving data to the device "
     "worth avoiding; where the two share one pool of memory the numbers are much "
     "higher.  Both readings use pinned host memory, the fast path.",
     TestShape::Heterogeneous, "direction"});

  const char *h2dNote = "Host to device: sending data across to the device.";
  const char *d2hNote = "Device to host: reading results back.  Often a little "
                        "slower than the other direction.";

  // Host staging memory.  The pinned path is a CL_MEM_ALLOC_HOST_PTR buffer
  // mapped once and left mapped: that pointer is host memory the driver may
  // DMA from directly, which is what cuMemAllocHost/hipHostMalloc give the
  // other backends.  Drivers that refuse to map it fall back to an ordinary
  // (pageable) aligned allocation, which measures the same route more slowly.
  cl::Buffer pinnedBuf;
  void *hostPtr = nullptr;
  float *pageable = nullptr;

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
      CLPEAK_VLOG("Pinned host buffer unavailable (%s, %d); falling back to pageable memory\n",
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
      test.skip("h2d_pinned", ResultStatus::Error, "Out of memory", h2dNote);
      test.skip("d2h_pinned", ResultStatus::Error, "Out of memory", d2hNote);
      return -1;
    }

    // Pseudo-random contents, to defeat hardware memory compression on both
    // directions.
    populate((float *)hostPtr, bytes / sizeof(float));

    // Warm up, probe one iteration to size the measurement window at
    // ~cfg.targetTimeUs, then run.  The copies are enqueued non-blocking and
    // the batch is drained once, so the driver can keep the link busy -- the
    // same shape as the async memcpy loops in the CUDA/ROCm/oneAPI backends.
    //
    // Always wall-clock, even under --use-event-timer: CL profiling events
    // time the device's command processing, which on a unified-memory device
    // is near zero for a copy that moves nothing (Apple M1 reports ~70x the
    // real rate).  The host clock over a ~targetTimeUs window is both accurate
    // enough and the number a caller actually pays.  oneAPI times this the
    // same way.
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
      return (float)bytes / timed / 1e3f;
    };

    float bw;

    bw = runTransfer(
      [&]() { queue.enqueueWriteBuffer(devBuf, CL_FALSE, 0, bytes, hostPtr); });
    if (bw > 0.0f) test.emit("h2d_pinned", bw, h2dNote);
    else           test.skip("h2d_pinned", ResultStatus::Error, "timer returned zero", h2dNote);

    bw = runTransfer(
      [&]() { queue.enqueueReadBuffer(devBuf, CL_FALSE, 0, bytes, hostPtr); });
    if (bw > 0.0f) test.emit("d2h_pinned", bw, d2hNote);
    else           test.skip("d2h_pinned", ResultStatus::Error, "timer returned zero", d2hNote);

    releaseHost();
  }
  catch (cl::Error &error)
  {
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    test.skip("h2d_pinned", ResultStatus::Error, reason, h2dNote);
    test.skip("d2h_pinned", ResultStatus::Error, reason, d2hNote);

    releaseHost();
    return -1;
  }

  return 0;
}
