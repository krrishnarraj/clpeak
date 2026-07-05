#include <opencl/cl_peak.h>
#include <chrono>

// ---------------------------------------------------------------------------
// Kernel launch latency (OpenCL) -- two distinct metrics:
//
//   dispatch  : CL_PROFILING_COMMAND_QUEUED -> CL_PROFILING_COMMAND_START.
//               One-way host-enqueue -> device-execution-start latency.
//               This is the historical clpeak number.
//
//   roundtrip : std::chrono around enqueueNDRangeKernel + finish().
//               Full host submit -> GPU complete -> signal back.
//
// Both are reported so users can compare apples-to-apples across backends.
// ---------------------------------------------------------------------------

int clPeak::runKernelLatency(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  if (!isAllowed(Benchmark::KernelLatency))
    return 0;

  cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
  cl_uint numItems = (devInfo.maxWGSize) * (devInfo.numCUs) * FETCH_PER_WI;
  cl::NDRange globalSize = (numItems / FETCH_PER_WI);
  cl::NDRange localSize  = devInfo.maxWGSize;
  unsigned int iters = forceIters ? specifiedIters : cfg.kernelLatencyIters;

  auto test = currentDeviceScope->beginTest(
    {"kernel_launch_latency", "Kernel launch latency", "us"});

  try
  {
    cl::Buffer inputBuf  = cl::Buffer(ctx, CL_MEM_READ_ONLY,  (numItems * sizeof(float)));
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (numItems * sizeof(float)));

    cl::Kernel kernel(prog, "global_bandwidth_v1_local_offset");
    kernel.setArg(0, inputBuf);
    kernel.setArg(1, outputBuf);

    // Direct enqueue site (not via run_kernel): clamp explicitly so a kernel
    // work-group limit below the device max does not fail with -54.
    clampToKernelWG(queue.getInfo<CL_QUEUE_DEVICE>(), kernel, globalSize, localSize);

    // Warmup
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.finish();

    double totalDispatchUs  = 0;
    double totalRoundtripUs = 0;
    for (unsigned int i = 0; i < iters; i++)
    {
      cl::Event timeEvent;
      auto t0 = std::chrono::high_resolution_clock::now();
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &timeEvent);
      queue.finish();
      auto t1 = std::chrono::high_resolution_clock::now();

      cl_ulong qd = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
      cl_ulong st = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      totalDispatchUs  += (double)(st - qd) / 1000.0; // ns -> us
      totalRoundtripUs += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    }

    float dispatchUs  = (float)(totalDispatchUs  / iters);
    float roundtripUs = (float)(totalRoundtripUs / iters);

    test.emit("dispatch",  dispatchUs);
    test.emit("roundtrip", roundtripUs);
  }
  catch (cl::Error &error)
  {
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    test.skip("dispatch", ResultStatus::Error, reason);
    test.skip("roundtrip", ResultStatus::Error, reason);
    return -1;
  }

  return 0;
}
