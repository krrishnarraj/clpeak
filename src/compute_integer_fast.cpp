#include <clpeak.h>

int clPeak::runComputeIntFast(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
  float timed, giops, peak_timed, peak_giops;
  cl_uint workPerWI;
  cl::NDRange globalSize, localSize;
  cl_int A = 4;
  uint iters = devInfo.computeIters;

  if (!isComputeIntFast)
    return 0;

  try
  {
    log->print(NEWLINE TAB TAB "Integer compute Fast 24bit (GIOPS)" NEWLINE);
    log->xmlOpenTag("integer_compute_fast");
    log->xmlAppendAttribs("unit", "giops");

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    uint64_t globalWIs = (devInfo.numCUs) * (devInfo.computeWgsPerCU) * (devInfo.maxWGSize);
    uint64_t t = std::min((globalWIs * sizeof(cl_int)), devInfo.maxAllocSize) / sizeof(cl_int);
    globalWIs = roundToMultipleOf(t, devInfo.maxWGSize);

    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_int)));

    globalSize = globalWIs;
    localSize = devInfo.maxWGSize;

    cl::Kernel kernel_v1(prog, "compute_intfast_v1");
    kernel_v1.setArg(0, outputBuf), kernel_v1.setArg(1, A);

    cl::Kernel kernel_v2(prog, "compute_intfast_v2");
    kernel_v2.setArg(0, outputBuf), kernel_v2.setArg(1, A);

    cl::Kernel kernel_v4(prog, "compute_intfast_v4");
    kernel_v4.setArg(0, outputBuf), kernel_v4.setArg(1, A);

    cl::Kernel kernel_v8(prog, "compute_intfast_v8");
    kernel_v8.setArg(0, outputBuf), kernel_v8.setArg(1, A);

    cl::Kernel kernel_v16(prog, "compute_intfast_v16");
    kernel_v16.setArg(0, outputBuf), kernel_v16.setArg(1, A);

    ///////////////////////////////////////////////////////////////////////////
    // Vector width 1
    if (!forceTest || strcmp(specifiedTestName, "int") == 0)
    {
      log->print(TAB TAB TAB "int   : ");

      workPerWI = 2048; // Indicates integer operations executed per work-item

      timed = run_kernel(queue, kernel_v1, globalSize, localSize, iters, &peak_timed);

      giops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;
      peak_giops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / peak_timed / 1e3f;

      log->print(giops);
      log->print(" (peak: "); log->print(peak_giops); log->print(")");
      log->print(NEWLINE);
      log->xmlRecord("int", giops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 2
    if (!forceTest || strcmp(specifiedTestName, "int2") == 0)
    {
      log->print(TAB TAB TAB "int2  : ");

      workPerWI = 2048;

      timed = run_kernel(queue, kernel_v2, globalSize, localSize, iters, &peak_timed);

      giops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;
      peak_giops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / peak_timed / 1e3f;

      log->print(giops);
      log->print(" (peak: "); log->print(peak_giops); log->print(")");
      log->print(NEWLINE);
      log->xmlRecord("int2", giops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 4
    if (!forceTest || strcmp(specifiedTestName, "int4") == 0)
    {
      log->print(TAB TAB TAB "int4  : ");

      workPerWI = 2048;

      timed = run_kernel(queue, kernel_v4, globalSize, localSize, iters, &peak_timed);

      giops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;
      peak_giops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / peak_timed / 1e3f;

      log->print(giops);
      log->print(" (peak: "); log->print(peak_giops); log->print(")");
      log->print(NEWLINE);
      log->xmlRecord("int4", giops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 8
    if (!forceTest || strcmp(specifiedTestName, "int8") == 0)
    {
      log->print(TAB TAB TAB "int8  : ");

      workPerWI = 2048;

      timed = run_kernel(queue, kernel_v8, globalSize, localSize, iters, &peak_timed);

      giops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;
      peak_giops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / peak_timed / 1e3f;

      log->print(giops);
      log->print(" (peak: "); log->print(peak_giops); log->print(")");
      log->print(NEWLINE);
      log->xmlRecord("int8", giops);
    }
    ///////////////////////////////////////////////////////////////////////////

    // Vector width 16
    if (!forceTest || strcmp(specifiedTestName, "int16") == 0)
    {
      log->print(TAB TAB TAB "int16 : ");

      workPerWI = 2048;

      timed = run_kernel(queue, kernel_v16, globalSize, localSize, iters, &peak_timed);

      giops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;
      peak_giops = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / peak_timed / 1e3f;

      log->print(giops);
      log->print(" (peak: "); log->print(peak_giops); log->print(")");
      log->print(NEWLINE);
      log->xmlRecord("int16", giops);
    }
    ///////////////////////////////////////////////////////////////////////////
    log->xmlCloseTag(); // integer_compute
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
