#include <clpeak.h>

int clPeak::runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  float timed_lo, timed_go, timed, gbps;
  cl::NDRange globalSize, localSize;
  float *arr = nullptr;

  if (!isAllowed(Benchmark::GlobalBW))
    return 0;

  cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
  unsigned int iters = cfg.globalBWIters;

  uint64_t maxItems = devInfo.maxAllocSize / sizeof(float) / 2;
  uint64_t numItems = roundToMultipleOf(maxItems, (devInfo.maxWGSize * FETCH_PER_WI * 16), cfg.globalBWMaxSize);

  try
  {
    arr = new float[numItems];
    populate(arr, numItems);

    log->print(NEWLINE TAB TAB "Global memory bandwidth (GBPS)" NEWLINE);
    log->resultScopeBegin("global_memory_bandwidth");
    log->resultScopeAttribute("unit", "gbps");

    cl::Buffer inputBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (numItems * sizeof(float)));
    queue.enqueueWriteBuffer(inputBuf, CL_TRUE, 0, (numItems * sizeof(float)), arr);

    cl::Kernel kernel_v1_lo(prog, "global_bandwidth_v1_local_offset");
    kernel_v1_lo.setArg(0, inputBuf), kernel_v1_lo.setArg(1, outputBuf);

    cl::Kernel kernel_v2_lo(prog, "global_bandwidth_v2_local_offset");
    kernel_v2_lo.setArg(0, inputBuf), kernel_v2_lo.setArg(1, outputBuf);

    cl::Kernel kernel_v4_lo(prog, "global_bandwidth_v4_local_offset");
    kernel_v4_lo.setArg(0, inputBuf), kernel_v4_lo.setArg(1, outputBuf);

    cl::Kernel kernel_v8_lo(prog, "global_bandwidth_v8_local_offset");
    kernel_v8_lo.setArg(0, inputBuf), kernel_v8_lo.setArg(1, outputBuf);

    cl::Kernel kernel_v16_lo(prog, "global_bandwidth_v16_local_offset");
    kernel_v16_lo.setArg(0, inputBuf), kernel_v16_lo.setArg(1, outputBuf);

    cl::Kernel kernel_v1_go(prog, "global_bandwidth_v1_global_offset");
    kernel_v1_go.setArg(0, inputBuf), kernel_v1_go.setArg(1, outputBuf);

    cl::Kernel kernel_v2_go(prog, "global_bandwidth_v2_global_offset");
    kernel_v2_go.setArg(0, inputBuf), kernel_v2_go.setArg(1, outputBuf);

    cl::Kernel kernel_v4_go(prog, "global_bandwidth_v4_global_offset");
    kernel_v4_go.setArg(0, inputBuf), kernel_v4_go.setArg(1, outputBuf);

    cl::Kernel kernel_v8_go(prog, "global_bandwidth_v8_global_offset");
    kernel_v8_go.setArg(0, inputBuf), kernel_v8_go.setArg(1, outputBuf);

    cl::Kernel kernel_v16_go(prog, "global_bandwidth_v16_global_offset");
    kernel_v16_go.setArg(0, inputBuf), kernel_v16_go.setArg(1, outputBuf);

    cl::Kernel *lo_kernels[] = {&kernel_v1_lo, &kernel_v2_lo, &kernel_v4_lo, &kernel_v8_lo, &kernel_v16_lo};
    cl::Kernel *go_kernels[] = {&kernel_v1_go, &kernel_v2_go, &kernel_v4_go, &kernel_v8_go, &kernel_v16_go};
    const int widths[] = {1, 2, 4, 8, 16};
    const char *labels[] = {"float", "float2", "float4", "float8", "float16"};
    const char *display[] = {"float   ", "float2  ", "float4  ", "float8  ", "float16 "};

    localSize = devInfo.maxWGSize;

    for (int w = 0; w < 5; w++)
    {
      if (forceTest && specifiedTestName != labels[w])
        continue;

      log->print(TAB TAB TAB + std::string(display[w]) + ": ");

      globalSize = numItems / widths[w] / FETCH_PER_WI;

      timed_lo = run_kernel(queue, *lo_kernels[w], globalSize, localSize, iters);
      timed_go = run_kernel(queue, *go_kernels[w], globalSize, localSize, iters);
      timed = (timed_lo < timed_go) ? timed_lo : timed_go;

      gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

      log->print(gbps);
      log->print(NEWLINE);
      log->resultRecord(labels[w], gbps);
    }

    log->resultScopeEnd(); // global_memory_bandwidth

    delete[] arr;
  }
  catch (cl::Error &error)
  {
    std::stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    const char *labels[] = {"float", "float2", "float4", "float8", "float16"};
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    for (int w = 0; w < 5; w++)
      log->recordSkip(labels[w], ResultStatus::Error, reason);

    // Close the resultScopeBegin pushed above so subsequent tests don't nest under
    // a leaked parent -- manifests on Android as later tests collapsing into
    // this test's result card.
    log->resultScopeEnd();
    delete[] arr;
    return -1;
  }
  catch (std::bad_alloc &)
  {
    log->print(TAB TAB TAB "Out of memory, tests skipped" NEWLINE);
    const char *labels[] = {"float", "float2", "float4", "float8", "float16"};
    for (int w = 0; w < 5; w++)
      log->recordSkip(labels[w], ResultStatus::Error, "Out of memory");
    log->resultScopeEnd();
    delete[] arr;
    return -1;
  }

  return 0;
}
