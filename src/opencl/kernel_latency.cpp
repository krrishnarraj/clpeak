#include <opencl/cl_peak.h>
#include <algorithm>
#include <chrono>

using xpupeak::Dim; namespace dim = xpupeak::dim;

// ---------------------------------------------------------------------------
// Kernel launch latency (OpenCL) -- two distinct metrics:
//
//   dispatch  : CL_PROFILING_COMMAND_QUEUED -> CL_PROFILING_COMMAND_START.
//               One-way host-enqueue -> device-execution-start latency.
//               This is the historical XpuPeak number.
//
//   roundtrip : the host clock around enqueueNDRangeKernel + finish().
//               Full host submit -> GPU complete -> signal back.
//
// Both launch one work-item of an empty kernel
// (kernels/kernel_latency_kernels.cl), as the other backends' latency tests
// do, so the roundtrip carries no kernel run time and both readings compare
// across backends.
//
// The event timer's unit is measured, because a driver need not count the
// nanoseconds the spec says.  Apple's OpenCL returns raw mach_absolute_time()
// ticks, which are nanoseconds only where the timebase is 1/1 (Intel Macs):
// on Apple silicon (125/3) each is 41.67 ns, and nothing the driver reports
// says so -- CL_DEVICE_PROFILING_TIMER_RESOLUTION is 1000.  M1 Pro, macOS
// 27.2.1: all four stamps of each of 4000 launches lie between
// mach_absolute_time() read before the enqueue and after finish(), and 2000
// launches' first and last QUEUED stamps are 9411239 apart over 392.1 ms of
// host clock, 41.668 ns each.  Read as nanoseconds, dispatch was 2.35 us
// against a 212 us roundtrip; timed, it is 92 us.
//
// Each launch's QUEUED stamp is taken between the host's t0 and t1 around
// that launch, so the first and last launches bound the host time between
// their stamps.  Where the stamps' difference, read as nanoseconds, fits that
// bound, the timer counts the spec's nanoseconds and the stamps are used as
// they are (NVIDIA, PoCL, Intel's CPU runtime); where it does not, one unit is
// the host time between those two launches' t0 over the stamps' difference.
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
  // roundtrip carried its run time; and dispatch has read the event stamps in
  // the unit the host clock times them in, where it took them as nanoseconds
  // and so read Apple's OpenCL 41.7x short (4.0.0).
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

    // Monotonic, since the event timer is timed against it.
    using Clock = std::chrono::steady_clock;
    auto ns = [](Clock::duration d) {
      return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
    };

    // The readings average the first iters launches.  The event timer is timed
    // over every launch, and over at least 200, so the first and last stamps
    // lie 199 launches apart and pin its unit to 1% even if each sat anywhere
    // within its own launch.
    const unsigned int launches = std::max(iters, 200u);
    double totalDispatchTicks = 0;
    double totalRoundtripUs = 0;
    cl_ulong firstQueued = 0, lastQueued = 0;
    Clock::time_point first0, first1, last0, last1;
    for (unsigned int i = 0; i < launches; i++)
    {
      cl::Event timeEvent;
      auto t0 = Clock::now();
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, one, one, nullptr, &timeEvent);
      queue.finish();
      auto t1 = Clock::now();

      cl_ulong qd = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
      if (i == 0)
      {
        firstQueued = qd;
        first0 = t0;
        first1 = t1;
      }
      lastQueued = qd;
      last0 = t0;
      last1 = t1;
      if (i < iters)
      {
        cl_ulong st = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        totalDispatchTicks += (double)(st - qd);
        totalRoundtripUs   += ns(t1 - t0) / 1000.0;
      }
    }

    // Nanoseconds per event-timer unit (the comment above this function).
    if (lastQueued <= firstQueued)
      test.skip("dispatch", ResultStatus::Error,
                "The event timer did not advance across the launches", dispatchNote,
                dispatchDims);
    else
    {
      const double ticks = (double)(lastQueued - firstQueued);
      const double lo = ns(last0 - first1), hi = ns(last1 - first0);
      double tickNs = 1.0;
      if (ticks < lo || ticks > hi)
        tickNs = ns(last0 - first0) / ticks;
      XPUPEAK_VLOG("kernel_launch_latency: event timer %.4f ns per unit (QUEUED stamps "
                   "%.0f apart, %.0f-%.0f ns of host clock, %u launches)\n",
                   tickNs, ticks, lo, hi, launches);
      float dispatchUs = (float)(totalDispatchTicks * tickNs / 1000.0 / iters);
      test.emit("dispatch", (float)(dispatchUs * 1e-6), dispatchNote, dispatchDims);
    }

    float roundtripUs = (float)(totalRoundtripUs / iters);
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
