#include <opencl/cl_peak.h>

int clPeak::runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  float timed_lo, timed_go, timed, gbps;
  cl::NDRange globalSize, localSize;
  float *arr = nullptr;

  if (!isAllowed(Benchmark::GlobalBW))
    return 0;

  cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
  unsigned int forced = forceIters ? specifiedIters : 0;

  uint64_t maxItems = devInfo.maxAllocSize / sizeof(float) / 2;
  uint64_t numItems = roundToMultipleOf(maxItems, (devInfo.maxWGSize * FETCH_PER_WI * 16), cfg.globalBWMaxSize / sizeof(float));

  // Opened before the sizing diagnostic below, so that line lands under
  // this test's header rather than under the previous test's readings.
  auto test = currentDeviceScope->beginTest(
    {"global_memory_bandwidth", "Global memory bandwidth", "gbps",
     Category::Unknown,
     "How many bytes per second the device can stream out of its main memory, "
     "reading a buffer far too large to cache.  Each reading fetches a "
     "different number of values per instruction, since wider fetches usually "
     "pull more through before the memory system saturates.",
     TestShape::Homogeneous, "vector width"});

  // The one number that decides whether this test measured memory or cache:
  // the timed phase re-reads the same buffer, so a working set that fits behind
  // the last-level cache reports the cache.  benchmark_config_t::forDevice
  // sizes globalBWMaxSize to clear the cache the device reported; print both so
  // an implausible reading can be checked against them without a rebuild.
  CLPEAK_VLOG("global_memory_bandwidth: working set %llu MB, device cache %llu MB\n",
              (unsigned long long)(numItems * sizeof(float) >> 20),
              (unsigned long long)(devInfo.globalMemCacheSize >> 20));

  try
  {
    arr = new float[numItems];
    populate(arr, numItems);

    // Every kernel writes one float per work-item, and the widest launch is the
    // float one at numItems / FETCH_PER_WI work-items (the vector kernels launch
    // fewer, and run_kernel's work-group clamp only ever shrinks the range), so
    // the output needs a sixteenth of the input -- worth spelling out now that
    // the input is sized to outrun a 96 MB+ last-level cache.
    cl::Buffer inputBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, ((numItems / FETCH_PER_WI) * sizeof(float)));
    queue.enqueueWriteBuffer(inputBuf, CL_TRUE, 0, (numItems * sizeof(float)), arr);
    // Blocking write, so the staging copy is dead the moment it returns.  At a
    // working set sized to clear the last-level cache that is half a gigabyte
    // or more of host memory not to be holding through the timed run -- which
    // on a CPU device is the same memory the benchmark is measuring.
    delete[] arr;
    arr = nullptr;

    cl::Kernel kernel_v1_lo(prog, "global_bandwidth_v1_local_offset");
    kernel_v1_lo.setArg(0, inputBuf), kernel_v1_lo.setArg(1, outputBuf);

    cl::Kernel kernel_v2_lo(prog, "global_bandwidth_v2_local_offset");
    kernel_v2_lo.setArg(0, inputBuf), kernel_v2_lo.setArg(1, outputBuf);

    cl::Kernel kernel_v4_lo(prog, "global_bandwidth_v4_local_offset");
    kernel_v4_lo.setArg(0, inputBuf), kernel_v4_lo.setArg(1, outputBuf);

    cl::Kernel kernel_v8_lo(prog, "global_bandwidth_v8_local_offset");
    kernel_v8_lo.setArg(0, inputBuf), kernel_v8_lo.setArg(1, outputBuf);

    cl::Kernel kernel_v16_lo(prog, "global_bandwidth_v16_local_offset");
    kernel_v16_lo.setArg(0, inputBuf), kernel_v16_lo.setArg(1, outputBuf);

    cl::Kernel kernel_v1_go(prog, "global_bandwidth_v1_global_offset");
    kernel_v1_go.setArg(0, inputBuf), kernel_v1_go.setArg(1, outputBuf);

    cl::Kernel kernel_v2_go(prog, "global_bandwidth_v2_global_offset");
    kernel_v2_go.setArg(0, inputBuf), kernel_v2_go.setArg(1, outputBuf);

    cl::Kernel kernel_v4_go(prog, "global_bandwidth_v4_global_offset");
    kernel_v4_go.setArg(0, inputBuf), kernel_v4_go.setArg(1, outputBuf);

    cl::Kernel kernel_v8_go(prog, "global_bandwidth_v8_global_offset");
    kernel_v8_go.setArg(0, inputBuf), kernel_v8_go.setArg(1, outputBuf);

    cl::Kernel kernel_v16_go(prog, "global_bandwidth_v16_global_offset");
    kernel_v16_go.setArg(0, inputBuf), kernel_v16_go.setArg(1, outputBuf);

    cl::Kernel *lo_kernels[] = {&kernel_v1_lo, &kernel_v2_lo, &kernel_v4_lo, &kernel_v8_lo, &kernel_v16_lo};
    cl::Kernel *go_kernels[] = {&kernel_v1_go, &kernel_v2_go, &kernel_v4_go, &kernel_v8_go, &kernel_v16_go};
    const int widths[] = {1, 2, 4, 8, 16};
    const char *labels[] = {"float", "float2", "float4", "float8", "float16"};

    for (int w = 0; w < 5; w++)
    {
      // Reset each iteration: run_kernel may clamp global/local for a kernel
      // whose work-group limit is below the device max.
      globalSize = numItems / widths[w] / FETCH_PER_WI;
      localSize = devInfo.maxWGSize;

      timed_lo = run_kernel(queue, *lo_kernels[w], globalSize, localSize, cfg.targetTimeUs, forced);
      timed_go = run_kernel(queue, *go_kernels[w], globalSize, localSize, cfg.targetTimeUs, forced);
      timed = (timed_lo < timed_go) ? timed_lo : timed_go;

      // Bytes actually moved = effective work-items * per-WI fetch.
      uint64_t movedFloats = ndRangeTotal(globalSize) * widths[w] * FETCH_PER_WI;
      gbps = ((float)movedFloats * sizeof(float)) / timed / 1e3f;

      // OpenCL is the only backend carrying both offset shapes -- the other
      // five implement the local-offset one alone -- so this race is the only
      // evidence anywhere of whether the grid-stride shape ever wins.  Print
      // both: if it never does by more than noise, the second family can go and
      // every backend measures the same thing; if it does, the shape has to be
      // ported to the other five instead.
      CLPEAK_VLOG("global_memory_bandwidth %s: local-offset %.1f, global-offset %.1f gbps\n",
                  labels[w],
                  ((float)movedFloats * sizeof(float)) / timed_lo / 1e3f,
                  ((float)movedFloats * sizeof(float)) / timed_go / 1e3f);

      test.emit(labels[w], gbps, clWidthNote(widths[w]));
     }
  }
  catch (cl::Error &error)
  {
    const char *labels[] = {"float", "float2", "float4", "float8", "float16"};
    const int widths[] = {1, 2, 4, 8, 16};
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    for (int w = 0; w < 5; w++)
      test.skip(labels[w], ResultStatus::Error, reason, clWidthNote(widths[w]));

    delete[] arr;
    return -1;
  }
  catch (std::bad_alloc &)
  {
    const char *labels[] = {"float", "float2", "float4", "float8", "float16"};
    const int widths[] = {1, 2, 4, 8, 16};
    for (int w = 0; w < 5; w++)
      test.skip(labels[w], ResultStatus::Error, "Out of memory", clWidthNote(widths[w]));
    delete[] arr;
    return -1;
  }

  return 0;
}
