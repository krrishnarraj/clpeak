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

  auto test = currentDeviceScope->beginTest(
    {"global_memory_bandwidth", "Global memory bandwidth", "gbps"});

  try
  {
    arr = new float[numItems];
    populate(arr, numItems);

    cl::Buffer inputBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (numItems * sizeof(float)));
    queue.enqueueWriteBuffer(inputBuf, CL_TRUE, 0, (numItems * sizeof(float)), arr);

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

      test.emit(labels[w], gbps);
     }

    delete[] arr;
  }
  catch (cl::Error &error)
  {
    const char *labels[] = {"float", "float2", "float4", "float8", "float16"};
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    for (int w = 0; w < 5; w++)
      test.skip(labels[w], ResultStatus::Error, reason);

    delete[] arr;
    return -1;
  }
  catch (std::bad_alloc &)
  {
    const char *labels[] = {"float", "float2", "float4", "float8", "float16"};
    for (int w = 0; w < 5; w++)
      test.skip(labels[w], ResultStatus::Error, "Out of memory");
    delete[] arr;
    return -1;
  }

  return 0;
}
