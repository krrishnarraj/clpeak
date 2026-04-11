#include <clpeak.h>

int clPeak::runLocalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  float timed, gbps;
  cl::NDRange globalSize, localSize;

  if (!isTestEnabled(Benchmark::LocalBW))
    return 0;

  unsigned int iters = cfg.localBWIters;

  uint64_t globalWIs = (uint64_t)devInfo.numCUs * cfg.computeWgsPerCU * devInfo.maxWGSize;

  try
  {
    log->print(NEWLINE TAB TAB "Local memory bandwidth (GBPS)" NEWLINE);
    log->xmlOpenTag("local_memory_bandwidth");
    log->xmlAppendAttribs("unit", "gbps");

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_float)));

    globalSize = globalWIs;
    localSize  = devInfo.maxWGSize;

    const int widths[] = {1, 2, 4, 8};
    const char *knames[] = {"local_bandwidth_v1", "local_bandwidth_v2", "local_bandwidth_v4", "local_bandwidth_v8"};
    const char *labels[] = {"float", "float2", "float4", "float8"};
    const char *display[] = {"float   ", "float2  ", "float4  ", "float8  "};

    cl::Kernel kernels[4];
    for (int w = 0; w < 4; w++)
    {
      kernels[w] = cl::Kernel(prog, knames[w]);
      kernels[w].setArg(0, outputBuf);
      kernels[w].setArg(1, cl::Local(devInfo.maxWGSize * widths[w] * sizeof(cl_float)));
    }

    for (int w = 0; w < 4; w++)
    {
      if (forceTest && specifiedTestName != labels[w])
        continue;

      // float8 requires enough local memory
      if (widths[w] == 8 && devInfo.localMemSize < devInfo.maxWGSize * 8 * sizeof(cl_float))
        continue;

      log->print(TAB TAB TAB + std::string(display[w]) + ": ");

      timed = run_kernel(queue, kernels[w], globalSize, localSize, iters);

      // Each rep: 1 write + 1 read per WI = 2 * width * sizeof(float) bytes per WI
      uint64_t bytesPerCall = (uint64_t)LMEM_REPS * 2 * widths[w] * sizeof(cl_float) * globalWIs;
      gbps = (float)bytesPerCall / timed / 1e3f;

      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord(labels[w], gbps);
    }

    log->xmlCloseTag(); // local_memory_bandwidth
  }
  catch (cl::Error &error)
  {
    std::stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    return -1;
  }

  return 0;
}
