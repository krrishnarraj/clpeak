#include <clpeak.h>

int clPeak::runAtomicThroughputTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  float timed, gops;
  cl::NDRange globalSize, localSize;

  if (!gating.isAllowed(Benchmark::AtomicThroughput))
    return 0;

  unsigned int iters = cfg.computeIters;

  uint64_t globalWIs = (uint64_t)devInfo.numCUs * cfg.computeWgsPerCU * devInfo.maxWGSize;
  uint64_t numWGs    = globalWIs / devInfo.maxWGSize;

  try
  {
    log->print(NEWLINE TAB TAB "Atomic throughput (GOPS)" NEWLINE);
    log->resultScopeBegin("atomic_throughput");
    log->resultScopeAttribute("unit", "gops");

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    globalSize = globalWIs;
    localSize  = devInfo.maxWGSize;

    ///////////////////////////////////////////////////////////////////////////
    // Global atomics -- independent per-WI counters (no cross-WI contention)
    if (!forceTest || specifiedTestName == "global")
    {
      log->print(TAB TAB TAB "global  : ");

      cl::Buffer globalBuf = cl::Buffer(ctx, CL_MEM_READ_WRITE, globalWIs * sizeof(cl_int));
      cl_int zero = 0;
      queue.enqueueFillBuffer(globalBuf, zero, 0, globalWIs * sizeof(cl_int));
      queue.finish();

      cl::Kernel kernel_global(prog, "atomic_throughput_global");
      kernel_global.setArg(0, globalBuf);

      timed = run_kernel(queue, kernel_global, globalSize, localSize, iters);

      gops = (static_cast<float>(globalWIs) * static_cast<float>(ATOMIC_REPS)) / timed / 1e3f;
      log->print(gops);
      log->print(NEWLINE);
      log->resultRecord("global", gops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Local atomics -- all WIs in a WG contend on one shared counter
    if (!forceTest || specifiedTestName == "local")
    {
      log->print(TAB TAB TAB "local   : ");

      cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, numWGs * sizeof(cl_int));

      cl::Kernel kernel_local(prog, "atomic_throughput_local");
      kernel_local.setArg(0, outputBuf);
      kernel_local.setArg(1, cl::Local(sizeof(cl_int)));

      timed = run_kernel(queue, kernel_local, globalSize, localSize, iters);

      gops = (static_cast<float>(globalWIs) * static_cast<float>(ATOMIC_REPS)) / timed / 1e3f;
      log->print(gops);
      log->print(NEWLINE);
      log->resultRecord("local", gops);
    }
    ///////////////////////////////////////////////////////////////////////////

    log->resultScopeEnd(); // atomic_throughput
  }
  catch (cl::Error &error)
  {
    std::stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    log->recordSkip("global", ResultStatus::Error, reason);
    log->recordSkip("local", ResultStatus::Error, reason);
    // Close the resultScopeBegin pushed above so subsequent tests don't nest under
    // a leaked parent -- manifests on Android as later tests collapsing into
    // this test's result card.
    log->resultScopeEnd();
    return -1;
  }

  return 0;
}
