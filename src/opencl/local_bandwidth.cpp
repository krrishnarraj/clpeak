#include <opencl/cl_peak.h>

int clPeak::runLocalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  float timed, bps;
  cl::NDRange globalSize, localSize;

  if (!isAllowed(Benchmark::LocalBW))
    return 0;

  unsigned int forced = forceIters ? specifiedIters : 0;

  uint64_t globalWIs = (uint64_t)devInfo.numCUs * cfg.computeWgsPerCU * devInfo.maxWGSize;

  auto test = currentDeviceScope->beginTest(
    {"local_memory_bandwidth", "Local memory bandwidth", "bps",
     Category::Unknown,
     "How many bytes per second the device moves through local memory -- the "
     "small on-chip scratchpad a group of work-items passes data through, "
     "which never goes out to main memory.",
     TestShape::Homogeneous, "vector width"});

  const int widths[] = {1, 2, 4, 8};
  const char *labels[] = {"float", "float2", "float4", "float8"};

  // CL_DEVICE_LOCAL_MEM_TYPE == CL_GLOBAL: the device has no scratchpad and the
  // runtime carves __local out of ordinary global memory -- every CPU runtime,
  // and some GPUs that back local memory with cache.  There is nothing here for
  // this test to measure: the ping-pong kernel's two barriers per rep then time
  // the runtime's work-item serialization against DRAM, which is why the same
  // CPU reads 79 GBPS under Intel's runtime and 170 under pocl.  Reporting
  // either under "local memory bandwidth" is reporting a number for a memory
  // that isn't there.
  if (!devInfo.localMemDedicated)
  {
    test.skipAll({labels[0], labels[1], labels[2], labels[3]}, ResultStatus::Unsupported,
                 "Device has no dedicated local memory (CL_DEVICE_LOCAL_MEM_TYPE is not CL_LOCAL)");
    return 0;
  }

  try
  {
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_float)));

    const char *knames[] = {"local_bandwidth_v1", "local_bandwidth_v2", "local_bandwidth_v4", "local_bandwidth_v8"};

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
      {
        test.skip(labels[w], ResultStatus::Unsupported,
                  "Local memory too small for a float8 scratchpad at this work-group size",
                  clWidthNote(widths[w]));
        continue;
      }

      // Reset each iteration: run_kernel may clamp global/local for a kernel
      // whose work-group limit is below the device max.
      globalSize = globalWIs;
      localSize  = devInfo.maxWGSize;

      timed = run_kernel(queue, kernels[w], globalSize, localSize, cfg.targetTimeUs, forced);

      // Each rep: 1 write + 1 read per WI = 2 * width * sizeof(float) bytes per WI
      uint64_t bytesPerCall = (uint64_t)LMEM_REPS * 2 * widths[w] * sizeof(cl_float) * ndRangeTotal(globalSize);
      bps = (float)bytesPerCall / timed * 1e6f;

      test.emit(labels[w], bps, clWidthNote(widths[w]));
    }
  }
  catch (cl::Error &error)
  {
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    for (int w = 0; w < 4; w++)
      test.skip(labels[w], ResultStatus::Error, reason, clWidthNote(widths[w]));
    return -1;
  }

  return 0;
}
