#include <opencl/cl_peak.h>
#include <chrono>

using xpupeak::Dim; namespace dim = xpupeak::dim;

// ---------------------------------------------------------------------------
// Kernel launch latency (OpenCL) -- two distinct metrics:
//
//   dispatch  : CL_PROFILING_COMMAND_QUEUED -> CL_PROFILING_COMMAND_START.
//               One-way host-enqueue -> device-execution-start latency.
//               This is the historical XpuPeak number.
//
//   roundtrip : std::chrono around enqueueNDRangeKernel + finish().
//               Full host submit -> GPU complete -> signal back.
//
// Both launch one work-item of an empty kernel
// (kernels/kernel_latency_kernels.cl), as the other backends' latency tests
// do, so the roundtrip carries no kernel run time and both readings compare
// across backends.
// ---------------------------------------------------------------------------

int clPeak::runKernelLatency(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  auto run = admit(Benchmark::KernelLatency);
  if (!run)
    return 0;

  unsigned int iters = forceIters ? specifiedIters : cfg.kernelLatencyIters;

  logger::TestSpec spec = {
    "kernel_launch_latency", "Kernel launch latency", "s", Category::Unknown,
    "Time to launch an empty kernel, the fixed cost every dispatch pays.",
    TestShape::Heterogeneous};
  // Revision 2: since it first shipped, the launch has become an empty kernel
  // at one work-item, as in every other backend, where it was global
  // bandwidth's float kernel over numCUs x maxWGSize work-items and the
  // roundtrip carried its run time (4.0.0).
  spec.revision = 2;
  auto test = currentDeviceScope->beginTest(spec);

  const char *dispatchNote = "Host submit to kernel start, one way.";
  const char *roundtripNote = "Host submit to completion seen by the host.";
  const Dims dispatchDims  = {{Dim::Op, dim::Op::Dispatch}};
  const Dims roundtripDims = {{Dim::Op, dim::Op::Roundtrip}};

  try
  {
    // A program that failed to build is a null handle, and an older ICD
    // loader dereferences one rather than returning CL_INVALID_PROGRAM.
    if (!prog())
      throw cl::Error(CL_INVALID_PROGRAM, "clCreateKernel");
    cl::Kernel kernel(prog, "kernel_latency_noop");
    const cl::NDRange one(1);

    // Warmup
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, one, one);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, one, one);
    queue.finish();

    double totalDispatchUs  = 0;
    double totalRoundtripUs = 0;
    for (unsigned int i = 0; i < iters; i++)
    {
      cl::Event timeEvent;
      auto t0 = std::chrono::high_resolution_clock::now();
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, one, one, nullptr, &timeEvent);
      queue.finish();
      auto t1 = std::chrono::high_resolution_clock::now();

      cl_ulong qd = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
      cl_ulong st = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
      totalDispatchUs  += (double)(st - qd) / 1000.0; // ns -> us
      totalRoundtripUs += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    }

    float dispatchUs  = (float)(totalDispatchUs  / iters);
    float roundtripUs = (float)(totalRoundtripUs / iters);

    test.emit("dispatch", (float)(dispatchUs * 1e-6), dispatchNote, dispatchDims);
    test.emit("roundtrip", (float)(roundtripUs * 1e-6), roundtripNote, roundtripDims);
  }
  catch (cl::Error &error)
  {
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    test.skip("dispatch", ResultStatus::Error, reason, dispatchNote, dispatchDims);
    test.skip("roundtrip", ResultStatus::Error, reason, roundtripNote, roundtripDims);
    return -1;
  }

  return 0;
}
