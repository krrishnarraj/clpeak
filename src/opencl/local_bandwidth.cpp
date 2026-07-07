#include <opencl/cl_peak.h>

int clPeak::runLocalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  float timed, gbps;
  cl::NDRange globalSize, localSize;

  if (!isAllowed(Benchmark::LocalBW))
    return 0;

  unsigned int forced = forceIters ? specifiedIters : 0;

  uint64_t globalWIs = (uint64_t)devInfo.numCUs * cfg.computeWgsPerCU * devInfo.maxWGSize;

  auto test = currentDeviceScope->beginTest(
    {"local_memory_bandwidth", "Local memory bandwidth", "gbps"});

  try
  {
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_float)));

    const int widths[] = {1, 2, 4, 8};
    const char *knames[] = {"local_bandwidth_v1", "local_bandwidth_v2", "local_bandwidth_v4", "local_bandwidth_v8"};
    const char *labels[] = {"float", "float2", "float4", "float8"};

    cl::Kernel kernels[4];
    for (int w = 0; w < 4; w++)
    {
      kernels[w] = cl::Kernel(prog, knames[w]);
      kernels[w].setArg(0, outputBuf);
      kernels[w].setArg(1, cl::Local(devInfo.maxWGSize * widths[w] * sizeof(cl_float)));
    }

    for (int w = 0; w < 4; w++)
    {
      // float8 requires enough local memory
      if (widths[w] == 8 && devInfo.localMemSize < devInfo.maxWGSize * 8 * sizeof(cl_float))
        continue;

      // Reset each iteration: run_kernel may clamp global/local for a kernel
      // whose work-group limit is below the device max.
      globalSize = globalWIs;
      localSize  = devInfo.maxWGSize;

      timed = run_kernel(queue, kernels[w], globalSize, localSize, cfg.targetTimeUs, forced);

      // Each rep: 1 write + 1 read per WI = 2 * width * sizeof(float) bytes per WI
      uint64_t bytesPerCall = (uint64_t)LMEM_REPS * 2 * widths[w] * sizeof(cl_float) * ndRangeTotal(globalSize);
      gbps = (float)bytesPerCall / timed / 1e3f;

      test.emit(labels[w], gbps);
    }
  }
  catch (cl::Error &error)
  {
    const char *labels[] = {"float", "float2", "float4", "float8"};
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    for (int w = 0; w < 4; w++)
      test.skip(labels[w], ResultStatus::Error, reason);
    return -1;
  }

  return 0;
}
