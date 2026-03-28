#include <clpeak.h>

// Must match the rep count hardcoded in local_bandwidth_kernels.cl
static const uint LMEM_REPS = 64;

int clPeak::runLocalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
  float timed, gbps;
  cl::NDRange globalSize, localSize;

  if (!isLocalBW)
    return 0;

  uint iters = devInfo.localBWIters;

  uint64_t globalWIs = (uint64_t)devInfo.numCUs * devInfo.computeWgsPerCU * devInfo.maxWGSize;

  try
  {
    log->print(NEWLINE TAB TAB "Local memory bandwidth (GBPS)" NEWLINE);
    log->xmlOpenTag("local_memory_bandwidth");
    log->xmlAppendAttribs("unit", "gbps");

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_float)));

    globalSize = globalWIs;
    localSize  = devInfo.maxWGSize;

    cl::Kernel kernel_v1(prog, "local_bandwidth_v1");
    kernel_v1.setArg(0, outputBuf);
    kernel_v1.setArg(1, cl::Local(devInfo.maxWGSize * 1 * sizeof(cl_float)));

    cl::Kernel kernel_v2(prog, "local_bandwidth_v2");
    kernel_v2.setArg(0, outputBuf);
    kernel_v2.setArg(1, cl::Local(devInfo.maxWGSize * 2 * sizeof(cl_float)));

    cl::Kernel kernel_v4(prog, "local_bandwidth_v4");
    kernel_v4.setArg(0, outputBuf);
    kernel_v4.setArg(1, cl::Local(devInfo.maxWGSize * 4 * sizeof(cl_float)));

    cl::Kernel kernel_v8(prog, "local_bandwidth_v8");
    kernel_v8.setArg(0, outputBuf);
    kernel_v8.setArg(1, cl::Local(devInfo.maxWGSize * 8 * sizeof(cl_float)));

    ///////////////////////////////////////////////////////////////////////////
    // float
    if (!forceTest || strcmp(specifiedTestName, "float") == 0)
    {
      log->print(TAB TAB TAB "float   : ");

      timed = run_kernel(queue, kernel_v1, globalSize, localSize, iters);

      // Each rep: 1 write + 1 read per WI = 2 * sizeof(float) bytes per WI
      uint64_t bytesPerCall = (uint64_t)LMEM_REPS * 2 * 1 * sizeof(cl_float) * globalWIs;
      gbps = (float)bytesPerCall / timed / 1e3f;

      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord("float", gbps);
    }
    ///////////////////////////////////////////////////////////////////////////

    // float2
    if (!forceTest || strcmp(specifiedTestName, "float2") == 0)
    {
      log->print(TAB TAB TAB "float2  : ");

      timed = run_kernel(queue, kernel_v2, globalSize, localSize, iters);

      uint64_t bytesPerCall = (uint64_t)LMEM_REPS * 2 * 2 * sizeof(cl_float) * globalWIs;
      gbps = (float)bytesPerCall / timed / 1e3f;

      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord("float2", gbps);
    }
    ///////////////////////////////////////////////////////////////////////////

    // float4
    if (!forceTest || strcmp(specifiedTestName, "float4") == 0)
    {
      log->print(TAB TAB TAB "float4  : ");

      timed = run_kernel(queue, kernel_v4, globalSize, localSize, iters);

      uint64_t bytesPerCall = (uint64_t)LMEM_REPS * 2 * 4 * sizeof(cl_float) * globalWIs;
      gbps = (float)bytesPerCall / timed / 1e3f;

      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord("float4", gbps);
    }
    ///////////////////////////////////////////////////////////////////////////

    // float8 — requires 32 KB local memory per workgroup; skip if insufficient
    if (!forceTest || strcmp(specifiedTestName, "float8") == 0)
    {
      if (devInfo.localMemSize >= devInfo.maxWGSize * 8 * sizeof(cl_float))
      {
        log->print(TAB TAB TAB "float8  : ");

        timed = run_kernel(queue, kernel_v8, globalSize, localSize, iters);

        uint64_t bytesPerCall = (uint64_t)LMEM_REPS * 2 * 8 * sizeof(cl_float) * globalWIs;
        gbps = (float)bytesPerCall / timed / 1e3f;

        log->print(gbps);
        log->print(NEWLINE);
        log->xmlRecord("float8", gbps);
      }
    }
    ///////////////////////////////////////////////////////////////////////////

    log->xmlCloseTag(); // local_memory_bandwidth
  }
  catch (cl::Error &error)
  {
    stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    return -1;
  }

  return 0;
}
