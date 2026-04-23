#include <clpeak.h>

int clPeak::runKernelLatency(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  if (!isTestEnabled(Benchmark::KernelLatency))
    return 0;

  cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
  cl_uint numItems = (devInfo.maxWGSize) * (devInfo.numCUs) * FETCH_PER_WI;
  cl::NDRange globalSize = (numItems / FETCH_PER_WI);
  cl::NDRange localSize = devInfo.maxWGSize;
  unsigned int iters = cfg.kernelLatencyIters;
  float latency;

  try
  {
    log->print(NEWLINE TAB TAB "Kernel launch latency : ");
    log->xmlOpenTag("kernel_launch_latency");
    log->xmlAppendAttribs("unit", "us");

    cl::Buffer inputBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (numItems * sizeof(float)));

    cl::Kernel kernel_v1(prog, "global_bandwidth_v1_local_offset");
    kernel_v1.setArg(0, inputBuf), kernel_v1.setArg(1, outputBuf);

    // Dummy calls
    queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
    queue.finish();

    latency = 0;
    for (unsigned int i = 0; i < iters; i++)
    {
      cl::Event timeEvent;
      queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize, nullptr, &timeEvent);
      queue.finish();
      cl_ulong start = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() / 1000;
      cl_ulong end = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
      latency += (float)(end - start);
    }
    latency /= static_cast<float>(iters);

    log->print(latency);
    log->print(" us" NEWLINE);
    log->xmlSetContent(latency);
    log->xmlCloseTag();
  }
  catch (cl::Error &error)
  {
    std::stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    // Close the xmlOpenTag pushed above so subsequent tests don't nest under
    // a leaked parent -- manifests on Android as later tests collapsing into
    // this test's result card.
    log->xmlCloseTag();
    return -1;
  }

  return 0;
}
