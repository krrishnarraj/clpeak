#include <clpeak.h>
#include <cstdlib>
#include <cstring>
#include <functional>

// If map/unmap bandwidth exceeds this multiplier of the peak real-transfer
// bandwidth measured earlier in the same run, it's a zero-copy / shared-memory
// operation with no actual data movement.  5x is conservative: on M1 the ratio
// is ~15000x, on Mali ~10x; on discrete GPUs the ratio is ~1x.
static const float ZERO_COPY_MULTIPLIER = 5.0f;

// Absolute floor: if no real-transfer baselines were measured (e.g. user ran
// only the map sub-test), fall back to this.  Well above any real hardware.
static const float ZERO_COPY_ABSOLUTE_GBPS = 10000.0f;

#if defined(_WIN32) || defined(__ANDROID__)
#include <malloc.h>
#endif

// Platform-specific aligned alloc/free
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

  if (!isTestEnabled(Benchmark::TransferBW))
    return 0;

  cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
  unsigned int iters = cfg.transferBWIters;
  float *arr = nullptr;

  uint64_t maxItems = devInfo.maxAllocSize / sizeof(float) / 2;
  uint64_t numItems = roundToMultipleOf(maxItems, devInfo.maxWGSize, cfg.transferBWMaxSize);
  size_t bytes = numItems * sizeof(float);

  // Track the peak bandwidth from real-transfer tests (write/read) so that
  // map/unmap can be compared against it to detect zero-copy.
  float peakRealTransferBW = 0;

  // Helper: run a timed transfer test with warmup, return measured gbps
  auto runTransfer = [&](const std::string &label, const std::string &xmlName,
                         std::function<void(cl::Event *)> op, bool forceWallClock = false) -> float
  {
    if (forceTest && specifiedTestName != xmlName)
      return 0;

    log->print(label);

    // Warmup
    for (unsigned int w = 0; w < warmupCount; w++)
    {
      op(nullptr);
      queue.finish();
    }

    float timed = 0;

    if (useEventTimer && !forceWallClock)
    {
      for (unsigned int i = 0; i < iters; i++)
      {
        cl::Event timeEvent;
        op(&timeEvent);
        queue.finish();
        timed += timeInUS(timeEvent);
      }
    }
    else
    {
      Timer timer;
      timer.start();
      for (unsigned int i = 0; i < iters; i++)
      {
        op(nullptr);
      }
      queue.finish();
      timed = timer.stopAndTime();
    }
    timed /= static_cast<float>(iters);

    float gbps = (float)bytes / timed / 1e3f;
    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord(xmlName, gbps);
    return gbps;
  };

  // Helper: check if a bandwidth value looks like zero-copy
  auto isZeroCopy = [&](float gbps) -> bool
  {
    if (peakRealTransferBW > 0)
      return gbps > peakRealTransferBW * ZERO_COPY_MULTIPLIER;
    return gbps > ZERO_COPY_ABSOLUTE_GBPS;
  };

  // Helper: report a map/unmap result, detecting zero-copy
  auto reportMapUnmap = [&](float gbps, const std::string &xmlName)
  {
    if (isZeroCopy(gbps))
    {
      log->print("inf (zero-copy)");
      log->print(NEWLINE);
      log->xmlRecord(xmlName, (float)0);
    }
    else
    {
      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord(xmlName, gbps);
    }
  };

  try
  {
    arr = allocAligned(bytes);
    if (!arr)
    {
      log->print(TAB TAB TAB "Out of memory, tests skipped" NEWLINE);
      return -1;
    }
    memset(arr, 0, bytes);
    cl::Buffer clBuffer = cl::Buffer(ctx, (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR), bytes);

    log->print(NEWLINE TAB TAB "Transfer bandwidth (GBPS)" NEWLINE);
    log->xmlOpenTag("transfer_bandwidth");
    log->xmlAppendAttribs("unit", "gbps");

    // enqueueWriteBuffer (blocking)
    float bw;
    bw = runTransfer(TAB TAB TAB "enqueueWriteBuffer              : ", "enqueuewritebuffer",
      [&](cl::Event *ev) { queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, bytes, arr, nullptr, ev); });
    if (bw > peakRealTransferBW) peakRealTransferBW = bw;

    // enqueueReadBuffer (blocking)
    bw = runTransfer(TAB TAB TAB "enqueueReadBuffer               : ", "enqueuereadbuffer",
      [&](cl::Event *ev) { queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, bytes, arr, nullptr, ev); });
    if (bw > peakRealTransferBW) peakRealTransferBW = bw;

    // enqueueWriteBuffer non-blocking
    bw = runTransfer(TAB TAB TAB "enqueueWriteBuffer non-blocking : ", "enqueuewritebuffer_nonblocking",
      [&](cl::Event *ev) { queue.enqueueWriteBuffer(clBuffer, CL_FALSE, 0, bytes, arr, nullptr, ev); });
    if (bw > peakRealTransferBW) peakRealTransferBW = bw;

    // enqueueReadBuffer non-blocking
    bw = runTransfer(TAB TAB TAB "enqueueReadBuffer non-blocking  : ", "enqueuereadbuffer_nonblocking",
      [&](cl::Event *ev) { queue.enqueueReadBuffer(clBuffer, CL_FALSE, 0, bytes, arr, nullptr, ev); });
    if (bw > peakRealTransferBW) peakRealTransferBW = bw;

    // enqueueMapBuffer(for read)
    // Always use wall-clock for map/unmap: CL events measure only GPU command
    // processing time, which is zero on unified-memory platforms (Apple Silicon),
    // causing division-by-zero / inf with --use-event-timer.
    if (!forceTest || specifiedTestName == "enqueuemapbuffer")
    {
      log->print(TAB TAB TAB "enqueueMapBuffer(for read)      : ");
      queue.finish();

      float timed = 0;
      for (unsigned int i = 0; i < iters; i++)
      {
        Timer timer;
        timer.start();
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, bytes);
        queue.finish();
        timed += timer.stopAndTime();

        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      }
      timed /= static_cast<float>(iters);

      float gbps = (float)bytes / timed / 1e3f;
      reportMapUnmap(gbps, "enqueuemapbuffer");
    }

    // memcpy from mapped ptr
    if (!forceTest || specifiedTestName == "memcpy_from_mapped_ptr")
    {
      log->print(TAB TAB TAB TAB "memcpy from mapped ptr        : ");
      queue.finish();

      float timed = 0;
      for (unsigned int i = 0; i < iters; i++)
      {
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, bytes);
        queue.finish();

        Timer timer;
        timer.start();
        memcpy(arr, mapPtr, bytes);
        timed += timer.stopAndTime();

        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      }
      timed /= static_cast<float>(iters);

      float gbps = (float)bytes / timed / 1e3f;
      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord("memcpy_from_mapped_ptr", gbps);
    }

    // enqueueUnmap(after write)
    if (!forceTest || specifiedTestName == "enqueueunmap")
    {
      log->print(TAB TAB TAB "enqueueUnmap(after write)       : ");
      queue.finish();

      float timed = 0;
      for (unsigned int i = 0; i < iters; i++)
      {
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, bytes);
        queue.finish();

        Timer timer;
        timer.start();
        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
        timed += timer.stopAndTime();
      }
      timed /= static_cast<float>(iters);

      float gbps = (float)bytes / timed / 1e3f;
      reportMapUnmap(gbps, "enqueueunmap");
    }

    // memcpy to mapped ptr
    if (!forceTest || specifiedTestName == "memcpy_to_mapped_ptr")
    {
      log->print(TAB TAB TAB TAB "memcpy to mapped ptr          : ");
      queue.finish();

      float timed = 0;
      for (unsigned int i = 0; i < iters; i++)
      {
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, bytes);
        queue.finish();

        Timer timer;
        timer.start();
        memcpy(mapPtr, arr, bytes);
        timed += timer.stopAndTime();

        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      }
      timed /= static_cast<float>(iters);

      float gbps = (float)bytes / timed / 1e3f;
      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord("memcpy_to_mapped_ptr", gbps);
    }

    log->xmlCloseTag(); // transfer_bandwidth

    freeAligned(arr);
  }
  catch (cl::Error &error)
  {
    std::stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());

    freeAligned(arr);
    return -1;
  }

  return 0;
}
