#include <clpeak.h>

// Must match ATOMIC_REPS defined in atomic_throughput_kernels.cl
static const uint ATOMIC_REPS = 512;

int clPeak::runAtomicThroughputTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
  float timed, giops;
  cl::NDRange globalSize, localSize;

  if (!isAtomicThroughput)
    return 0;

  uint iters = devInfo.computeIters;

  uint64_t globalWIs = (uint64_t)devInfo.numCUs * devInfo.computeWgsPerCU * devInfo.maxWGSize;
  uint64_t numWGs    = globalWIs / devInfo.maxWGSize;

  try
  {
    log->print(NEWLINE TAB TAB "Atomic throughput (GIOPS)" NEWLINE);
    log->xmlOpenTag("atomic_throughput");
    log->xmlAppendAttribs("unit", "giops");

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    globalSize = globalWIs;
    localSize  = devInfo.maxWGSize;

    ///////////////////////////////////////////////////////////////////////////
    // Global atomics — independent per-WI counters (no cross-WI contention)
    if (!forceTest || strcmp(specifiedTestName, "global") == 0)
    {
      log->print(TAB TAB TAB "global  : ");

      cl::Buffer globalBuf = cl::Buffer(ctx, CL_MEM_READ_WRITE, globalWIs * sizeof(cl_int));
      cl_int zero = 0;
      queue.enqueueFillBuffer(globalBuf, zero, 0, globalWIs * sizeof(cl_int));
      queue.finish();

      cl::Kernel kernel_global(prog, "atomic_throughput_global");
      kernel_global.setArg(0, globalBuf);

      timed = run_kernel(queue, kernel_global, globalSize, localSize, iters);

      giops = (static_cast<float>(globalWIs) * static_cast<float>(ATOMIC_REPS)) / timed / 1e3f;
      log->print(giops);
      log->print(NEWLINE);
      log->xmlRecord("global", giops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Local atomics — all WIs in a WG contend on one shared counter
    if (!forceTest || strcmp(specifiedTestName, "local") == 0)
    {
      log->print(TAB TAB TAB "local   : ");

      cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, numWGs * sizeof(cl_int));

      cl::Kernel kernel_local(prog, "atomic_throughput_local");
      kernel_local.setArg(0, outputBuf);
      kernel_local.setArg(1, cl::Local(sizeof(cl_int)));

      timed = run_kernel(queue, kernel_local, globalSize, localSize, iters);

      giops = (static_cast<float>(globalWIs) * static_cast<float>(ATOMIC_REPS)) / timed / 1e3f;
      log->print(giops);
      log->print(NEWLINE);
      log->xmlRecord("local", giops);
    }
    ///////////////////////////////////////////////////////////////////////////

    log->xmlCloseTag(); // atomic_throughput
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
