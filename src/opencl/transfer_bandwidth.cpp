#include <opencl/cl_peak.h>
#include <opencl/cl_utils.h>
#include <cstdlib>
#include <cstring>
#include <functional>

// If map/unmap bandwidth exceeds this multiplier of the peak real-transfer
// bandwidth measured earlier in the same run, it's a zero-copy / shared-memory
// operation with no actual data movement.  3x is conservative: on M1 the ratio
// is ~15000x, on Mali ~10x; on discrete GPUs the ratio is ~1x.
static const float ZERO_COPY_MULTIPLIER = 3.0f;

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

  if (!isAllowed(Benchmark::TransferBW))
    return 0;

  cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
  unsigned int forced = forceIters ? specifiedIters : 0;
  float *arr = nullptr;

  uint64_t maxItems = devInfo.maxAllocSize / sizeof(float) / 2;
  uint64_t numItems = roundToMultipleOf(maxItems, devInfo.maxWGSize, cfg.transferBWMaxSize / sizeof(float));
  size_t bytes = numItems * sizeof(float);

  // Track the peak bandwidth from real-transfer tests (write/read) so that
  // map/unmap can be compared against it to detect zero-copy.
  float peakRealTransferBW = 0;

  // Helper: run a timed transfer test with warmup + calibration, return
  // measured gbps.  Calibrates iters from a one-iteration timed probe so the
  // measurement window lands at ~cfg.targetTimeUs regardless of bus speed.
  auto runTransfer = [&](std::function<void(cl::Event *)> op, bool forceWallClock = false) -> float
  {
    // Phase 1: untimed warmup
    for (unsigned int w = 0; w < warmupCount; w++)
    {
      op(nullptr);
      queue.finish();
    }

    auto runBatch = [&](unsigned int n) -> float {
      float total = 0;
      if (useEventTimer && !forceWallClock)
      {
        for (unsigned int i = 0; i < n; i++)
        {
          cl::Event timeEvent;
          op(&timeEvent);
          queue.finish();
          total += timeInUS(timeEvent);
        }
        return total;
      }
      auto t1 = std::chrono::high_resolution_clock::now();
      for (unsigned int i = 0; i < n; i++)
        op(nullptr);
      queue.finish();
      auto t2 = std::chrono::high_resolution_clock::now();
      return (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    };

    // Phase 2: timed probe -> per-iter time -> calibrated iters.
    unsigned int probeIters = 1;
    float probeUs = runBatch(probeIters);
    double per_iter_us = (double)probeUs / (double)probeIters;
    unsigned int iters = pickIters(per_iter_us, cfg.targetTimeUs, forced);

    // Phase 3: real timed run.
    float timed = runBatch(iters) / static_cast<float>(iters);

    float gbps = (float)bytes / timed / 1e3f;
    return gbps;
  };

  // Helper: check if a bandwidth value looks like zero-copy
  auto isZeroCopy = [&](float gbps) -> bool
  {
    if (peakRealTransferBW > 0)
      return gbps > peakRealTransferBW * ZERO_COPY_MULTIPLIER;
    return gbps > ZERO_COPY_ABSOLUTE_GBPS;
  };

  auto test = currentDeviceScope->beginTest(
    {"transfer_bandwidth", "Transfer bandwidth", "gbps"});

  // Helper: report a map/unmap result, detecting zero-copy
  auto reportMapUnmap = [&](float gbps, const std::string &resultName,
                            logger::EmitOptions opts = {})
  {
    if (isZeroCopy(gbps))
      test.emit(resultName, 0.0f, opts);
    else
      test.emit(resultName, gbps, opts);
  };

  try
  {
    arr = allocAligned(bytes);
    if (!arr)
    {
      test.skip("enqueuewritebuffer", ResultStatus::Error, "Out of memory");
      test.skip("enqueuereadbuffer", ResultStatus::Error, "Out of memory");
      test.skip("enqueuewritebuffer_nonblocking", ResultStatus::Error, "Out of memory");
      test.skip("enqueuereadbuffer_nonblocking", ResultStatus::Error, "Out of memory");
      test.skip("enqueuemapbuffer", ResultStatus::Error, "Out of memory");
      test.skip("memcpy_from_mapped_ptr", ResultStatus::Error, "Out of memory");
      test.skip("enqueueunmap", ResultStatus::Error, "Out of memory");
      test.skip("memcpy_to_mapped_ptr", ResultStatus::Error, "Out of memory");
      return -1;
    }
    populate(arr, bytes / sizeof(float));
    cl::Buffer clBuffer = cl::Buffer(ctx, (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR), bytes);

    // enqueueWriteBuffer (blocking)
    float bw;
    bw = runTransfer(
      [&](cl::Event *ev) { queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, bytes, arr, nullptr, ev); });
    test.emit("enqueuewritebuffer", bw);
    if (bw > peakRealTransferBW) peakRealTransferBW = bw;

    // enqueueReadBuffer (blocking)
    bw = runTransfer(
      [&](cl::Event *ev) { queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, bytes, arr, nullptr, ev); });
    test.emit("enqueuereadbuffer", bw);
    if (bw > peakRealTransferBW) peakRealTransferBW = bw;

    // enqueueWriteBuffer non-blocking
    bw = runTransfer(
      [&](cl::Event *ev) { queue.enqueueWriteBuffer(clBuffer, CL_FALSE, 0, bytes, arr, nullptr, ev); });
    test.emit("enqueuewritebuffer_nonblocking", bw);
    if (bw > peakRealTransferBW) peakRealTransferBW = bw;

    // enqueueReadBuffer non-blocking
    bw = runTransfer(
      [&](cl::Event *ev) { queue.enqueueReadBuffer(clBuffer, CL_FALSE, 0, bytes, arr, nullptr, ev); });
    test.emit("enqueuereadbuffer_nonblocking", bw);
    if (bw > peakRealTransferBW) peakRealTransferBW = bw;

    // Helper: calibrate iter count for the open-coded map/unmap loops below.
    // Each loop runs `iter()` per iteration (one full setup+timed-segment+
    // teardown) but only times a sub-segment.  We probe full-iter wall-clock
    // so calibration sizes total runtime to ~cfg.targetTimeUs.
    auto calibrateMapIters = [&](std::function<void()> iter) -> unsigned int {
      for (unsigned int w = 0; w < warmupCount; w++) iter();
      unsigned int probe = 1;
      auto t1 = std::chrono::high_resolution_clock::now();
      for (unsigned int i = 0; i < probe; i++) iter();
      auto t2 = std::chrono::high_resolution_clock::now();
      float probeUs = (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      return pickIters((double)probeUs / (double)probe, cfg.targetTimeUs, forced);
    };

    // enqueueMapBuffer(for read)
    // Always use wall-clock for map/unmap: CL events measure only GPU command
    // processing time, which is zero on unified-memory platforms (Apple Silicon),
    // causing division-by-zero / inf with --use-event-timer.
    {
      queue.finish();

      unsigned int iters = calibrateMapIters([&]() {
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, bytes);
        queue.finish();
        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      });

      float timed = 0;
      for (unsigned int i = 0; i < iters; i++)
      {
        auto t1 = std::chrono::high_resolution_clock::now();
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, bytes);
        queue.finish();
        auto t2 = std::chrono::high_resolution_clock::now();
        timed += (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      }
      timed /= static_cast<float>(iters);

      float gbps = (float)bytes / timed / 1e3f;
      reportMapUnmap(gbps, "enqueuemapbuffer");
    }

    // memcpy from mapped ptr
    {
      queue.finish();

      unsigned int iters = calibrateMapIters([&]() {
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, bytes);
        queue.finish();
        memcpy(arr, mapPtr, bytes);
        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      });

      float timed = 0;
      for (unsigned int i = 0; i < iters; i++)
      {
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, bytes);
        queue.finish();

        auto t1 = std::chrono::high_resolution_clock::now();
        memcpy(arr, mapPtr, bytes);
        auto t2 = std::chrono::high_resolution_clock::now();
        timed += (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      }
      timed /= static_cast<float>(iters);

      float gbps = (float)bytes / timed / 1e3f;
      test.emit("memcpy_from_mapped_ptr", gbps, {true});
    }

    // enqueueUnmap(after write)
    {
      queue.finish();

      unsigned int iters = calibrateMapIters([&]() {
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, bytes);
        queue.finish();
        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      });

      float timed = 0;
      for (unsigned int i = 0; i < iters; i++)
      {
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, bytes);
        queue.finish();

        auto t1 = std::chrono::high_resolution_clock::now();
        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
        auto t2 = std::chrono::high_resolution_clock::now();
        timed += (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
      }
      timed /= static_cast<float>(iters);

      float gbps = (float)bytes / timed / 1e3f;
      reportMapUnmap(gbps, "enqueueunmap");
    }

    // memcpy to mapped ptr
    {
      queue.finish();

      unsigned int iters = calibrateMapIters([&]() {
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, bytes);
        queue.finish();
        memcpy(mapPtr, arr, bytes);
        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      });

      float timed = 0;
      for (unsigned int i = 0; i < iters; i++)
      {
        void *mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, bytes);
        queue.finish();

        auto t1 = std::chrono::high_resolution_clock::now();
        memcpy(mapPtr, arr, bytes);
        auto t2 = std::chrono::high_resolution_clock::now();
        timed += (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      }
      timed /= static_cast<float>(iters);

      float gbps = (float)bytes / timed / 1e3f;
      test.emit("memcpy_to_mapped_ptr", gbps, {true});
    }

    freeAligned(arr);
  }
  catch (cl::Error &error)
  {
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    test.skip("enqueuewritebuffer", ResultStatus::Error, reason);
    test.skip("enqueuereadbuffer", ResultStatus::Error, reason);
    test.skip("enqueuewritebuffer_nonblocking", ResultStatus::Error, reason);
    test.skip("enqueuereadbuffer_nonblocking", ResultStatus::Error, reason);
    test.skip("enqueuemapbuffer", ResultStatus::Error, reason);
    test.skip("memcpy_from_mapped_ptr", ResultStatus::Error, reason);
    test.skip("enqueueunmap", ResultStatus::Error, reason);
    test.skip("memcpy_to_mapped_ptr", ResultStatus::Error, reason);

    freeAligned(arr);
    return -1;
  }

  return 0;
}
