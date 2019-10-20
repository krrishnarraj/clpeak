#include <clpeak.h>

int clPeak::runComputeDP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
  float timed, gflops;
  cl_uint workPerWI;
  cl::NDRange globalSize, localSize;
  cl_double A = 1.3f;
  uint iters = devInfo.computeIters;

  if (!isComputeDP)
    return 0;

  if (!devInfo.doubleSupported)
  {
    log->print(NEWLINE TAB TAB "No double precision support! Skipped" NEWLINE);
    return 0;
  }

  try
  {
    log->print(NEWLINE TAB TAB "Double-precision compute (GFLOPS)" NEWLINE);
    log->xmlOpenTag("double_precision_compute");
    log->xmlAppendAttribs("unit", "gflops");

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    uint64_t globalWIs = (devInfo.numCUs) * (devInfo.computeDPWgsPerCU) * (devInfo.maxWGSize);
    uint64_t t = MIN((globalWIs * sizeof(cl_double)), devInfo.maxAllocSize) / sizeof(cl_double);
    globalWIs = roundToMultipleOf(t, devInfo.maxWGSize);

    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_double)));

    globalSize = globalWIs;
    localSize = devInfo.maxWGSize;

    cl::Kernel kernel_v1(prog, "compute_dp_v1");
    kernel_v1.setArg(0, outputBuf), kernel_v1.setArg(1, A);

    cl::Kernel kernel_v2(prog, "compute_dp_v2");
    kernel_v2.setArg(0, outputBuf), kernel_v2.setArg(1, A);

    cl::Kernel kernel_v4(prog, "compute_dp_v4");
    kernel_v4.setArg(0, outputBuf), kernel_v4.setArg(1, A);

    cl::Kernel kernel_v8(prog, "compute_dp_v8");
    kernel_v8.setArg(0, outputBuf), kernel_v8.setArg(1, A);

    cl::Kernel kernel_v16(prog, "compute_dp_v16");
    kernel_v16.setArg(0, outputBuf), kernel_v16.setArg(1, A);

    ///////////////////////////////////////////////////////////////////////////
    // Vector width 1
    log->print(TAB TAB TAB "double   : ");

    workPerWI = 4096; // Indicates flops executed per work-item

    timed = run_kernel(queue, kernel_v1, globalSize, localSize, iters);

    gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

    log->print(gflops);
    log->print(NEWLINE);
    log->xmlRecord("double", gflops);
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 2
    log->print(TAB TAB TAB "double2  : ");

    workPerWI = 4096;

    timed = run_kernel(queue, kernel_v2, globalSize, localSize, iters);

    gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

    log->print(gflops);
    log->print(NEWLINE);
    log->xmlRecord("double2", gflops);
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 4
    log->print(TAB TAB TAB "double4  : ");

    workPerWI = 4096;

    timed = run_kernel(queue, kernel_v4, globalSize, localSize, iters);

    gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

    log->print(gflops);
    log->print(NEWLINE);
    log->xmlRecord("double4", gflops);
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 8
    log->print(TAB TAB TAB "double8  : ");
    workPerWI = 4096;

    timed = run_kernel(queue, kernel_v8, globalSize, localSize, iters);

    gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

    log->print(gflops);
    log->print(NEWLINE);
    log->xmlRecord("double8", gflops);
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 16
    log->print(TAB TAB TAB "double16 : ");

    workPerWI = 4096;

    timed = run_kernel(queue, kernel_v16, globalSize, localSize, iters);

    gflops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

    log->print(gflops);
    log->print(NEWLINE);
    log->xmlRecord("double16", gflops);
    ///////////////////////////////////////////////////////////////////////////
    log->xmlCloseTag(); // double_precision_compute
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
