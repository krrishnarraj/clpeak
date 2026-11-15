#include <opencl/cl_peak.h>

using xpupeak::Dim;

int clPeak::runLocalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  float timed, bps;
  cl::NDRange globalSize, localSize;

  auto run = admit(Benchmark::LocalBW);
  if (!run)
    return 0;

  unsigned int forced = forceIters ? specifiedIters : 0;

  uint64_t globalWIs = (uint64_t)devInfo.numCUs * cfg.computeWgsPerCU * devInfo.maxWGSize;

  auto test = currentDeviceScope->beginTest(
    {"local_memory_bandwidth", "Local memory bandwidth", "bps",
     Category::Unknown,
     "Bandwidth of local memory, the scratchpad a work-group shares.",
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

  // Each width gets exactly one reading.  The sweep records them in order, so
  // `w` is also the first width with no reading yet: a failure outside the
  // sweep (the buffer, the kernels) skips only the widths from there on, never
  // one already measured, whose id would then hold two readings.
  int w = 0;

  try
  {
    // A program that failed to build is a null handle, and an older ICD
    // loader dereferences one rather than returning CL_INVALID_PROGRAM.
    if (!prog())
      throw cl::Error(CL_INVALID_PROGRAM, "clCreateKernel");
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_float)));

    const char *knames[] = {"local_bandwidth_v1", "local_bandwidth_v2", "local_bandwidth_v4", "local_bandwidth_v8"};

    cl::Kernel kernels[4];
    for (int k = 0; k < 4; k++)
    {
      kernels[k] = cl::Kernel(prog, knames[k]);
      kernels[k].setArg(0, outputBuf);
      kernels[k].setArg(1, cl::Local(devInfo.maxWGSize * widths[k] * sizeof(cl_float)));
    }

    // A width that fails records its own error, and the next one still runs.
    for (; w < 4; w++)
    {
      // float8 requires enough local memory
      if (widths[w] == 8 && devInfo.localMemSize < devInfo.maxWGSize * 8 * sizeof(cl_float))
      {
        test.skip(labels[w], ResultStatus::Unsupported,
                  "Local memory too small for a float8 scratchpad at this work-group size",
                  clWidthNote(widths[w]), {{Dim::Width, widths[w]}});
        continue;
      }

      try
      {
        // Reset each iteration: run_kernel may clamp global/local for a kernel
        // whose work-group limit is below the device max.
        globalSize = globalWIs;
        localSize  = devInfo.maxWGSize;

        timed = run_kernel(queue, kernels[w], globalSize, localSize, cfg.targetTimeUs, forced);

        // Each rep: 1 write + 1 read per WI = 2 * width * sizeof(float) bytes per WI
        uint64_t bytesPerCall = (uint64_t)LMEM_REPS * 2 * widths[w] * sizeof(cl_float) * ndRangeTotal(globalSize);
        bps = (float)bytesPerCall / timed * 1e6f;

        test.emit(labels[w], bps, clWidthNote(widths[w]), {{Dim::Width, widths[w]}});
      }
      catch (cl::Error &error)
      {
        std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
        test.skip(labels[w], ResultStatus::Error, reason, clWidthNote(widths[w]),
                  {{Dim::Width, widths[w]}});
      }
    }
  }
  catch (cl::Error &error)
  {
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    for (; w < 4; w++)
      test.skip(labels[w], ResultStatus::Error, reason, clWidthNote(widths[w]),
                {{Dim::Width, widths[w]}});
    return -1;
  }

  return 0;
}
