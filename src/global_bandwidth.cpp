#include <clpeak.h>

#define FETCH_PER_WI 16

int clPeak::runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
  float timed_lo, timed_go, timed, gbps;
  cl::NDRange globalSize, localSize;
  float *arr = NULL;

  if (!isGlobalBW)
    return 0;

  cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
  uint iters = devInfo.gloalBWIters;

  uint64_t maxItems = devInfo.maxAllocSize / sizeof(float) / 2;
  uint64_t numItems = roundToMultipleOf(maxItems, (devInfo.maxWGSize * FETCH_PER_WI * 16), devInfo.globalBWMaxSize);

  try
  {
    arr = new float[numItems];
    populate(arr, numItems);

    log->print(NEWLINE TAB TAB "Global memory bandwidth (GBPS)" NEWLINE);
    log->xmlOpenTag("global_memory_bandwidth");
    log->xmlAppendAttribs("unit", "gbps");

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

    localSize = devInfo.maxWGSize;

    ///////////////////////////////////////////////////////////////////////////
    // Vector width 1
    log->print(TAB TAB TAB "float   : ");

    globalSize = numItems / FETCH_PER_WI;

    // Run 2 kind of bandwidth kernel
    // lo -- local_size offset - subsequent fetches at local_size offset
    // go -- global_size offset
    timed_lo = run_kernel(queue, kernel_v1_lo, globalSize, localSize, iters);
    timed_go = run_kernel(queue, kernel_v1_go, globalSize, localSize, iters);
    timed = (timed_lo < timed_go) ? timed_lo : timed_go;

    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("float", gbps);
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 2
    log->print(TAB TAB TAB "float2  : ");

    globalSize = (numItems / 2 / FETCH_PER_WI);

    timed_lo = run_kernel(queue, kernel_v2_lo, globalSize, localSize, iters);
    timed_go = run_kernel(queue, kernel_v2_go, globalSize, localSize, iters);
    timed = (timed_lo < timed_go) ? timed_lo : timed_go;

    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("float2", gbps);
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 4
    log->print(TAB TAB TAB "float4  : ");

    globalSize = (numItems / 4 / FETCH_PER_WI);

    timed_lo = run_kernel(queue, kernel_v4_lo, globalSize, localSize, iters);
    timed_go = run_kernel(queue, kernel_v4_go, globalSize, localSize, iters);
    timed = (timed_lo < timed_go) ? timed_lo : timed_go;

    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("float4", gbps);
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 8
    log->print(TAB TAB TAB "float8  : ");

    globalSize = (numItems / 8 / FETCH_PER_WI);

    timed_lo = run_kernel(queue, kernel_v8_lo, globalSize, localSize, iters);
    timed_go = run_kernel(queue, kernel_v8_go, globalSize, localSize, iters);
    timed = (timed_lo < timed_go) ? timed_lo : timed_go;

    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("float8", gbps);
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 16
    log->print(TAB TAB TAB "float16 : ");
    globalSize = (numItems / 16 / FETCH_PER_WI);

    timed_lo = run_kernel(queue, kernel_v16_lo, globalSize, localSize, iters);
    timed_go = run_kernel(queue, kernel_v16_go, globalSize, localSize, iters);
    timed = (timed_lo < timed_go) ? timed_lo : timed_go;

    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("float16", gbps);
    ///////////////////////////////////////////////////////////////////////////
    log->xmlCloseTag(); // global_memory_bandwidth

    if (arr)
    {
      delete[] arr;
    }
  }
  catch (cl::Error &error)
  {
    stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());

    if (arr)
    {
      delete[] arr;
    }
    return -1;
  }

  return 0;
}
