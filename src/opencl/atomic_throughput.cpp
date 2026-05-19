#include <opencl/cl_peak.h>

int clPeak::runAtomicThroughputTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  float timed, gops;
  cl::NDRange globalSize, localSize;

  if (!isAllowed(Benchmark::AtomicThroughput))
    return 0;

  unsigned int forced = forceIters ? specifiedIters : 0;

  uint64_t globalWIs = (uint64_t)devInfo.numCUs * cfg.computeWgsPerCU * devInfo.maxWGSize;
  uint64_t numWGs    = globalWIs / devInfo.maxWGSize;

  auto test = currentDeviceScope->beginTest(
    {"atomic_throughput", "Atomic throughput", "gops"});

  try
  {
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    globalSize = globalWIs;
    localSize  = devInfo.maxWGSize;

    ///////////////////////////////////////////////////////////////////////////
    // Global atomics -- independent per-WI counters (no cross-WI contention)
    {
      cl::Buffer globalBuf = cl::Buffer(ctx, CL_MEM_READ_WRITE, globalWIs * sizeof(cl_int));
      cl_int zero = 0;
      queue.enqueueFillBuffer(globalBuf, zero, 0, globalWIs * sizeof(cl_int));
      queue.finish();

      cl::Kernel kernel_global(prog, "atomic_throughput_global");
      kernel_global.setArg(0, globalBuf);

      timed = run_kernel(queue, kernel_global, globalSize, localSize, cfg.targetTimeUs, forced);

      gops = (static_cast<float>(globalWIs) * static_cast<float>(ATOMIC_REPS)) / timed / 1e3f;
      test.emit("int_global", gops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Local atomics -- all WIs in a WG contend on one shared counter
    {
      cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, numWGs * sizeof(cl_int));

      cl::Kernel kernel_local(prog, "atomic_throughput_local");
      kernel_local.setArg(0, outputBuf);
      kernel_local.setArg(1, cl::Local(sizeof(cl_int)));

      timed = run_kernel(queue, kernel_local, globalSize, localSize, cfg.targetTimeUs, forced);

      gops = (static_cast<float>(globalWIs) * static_cast<float>(ATOMIC_REPS)) / timed / 1e3f;
      test.emit("int_local", gops);
    }
    ///////////////////////////////////////////////////////////////////////////
  }
  catch (cl::Error &error)
  {
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    test.skip("int_global", ResultStatus::Error, reason);
    test.skip("int_local", ResultStatus::Error, reason);
    return -1;
  }

  return 0;
}
