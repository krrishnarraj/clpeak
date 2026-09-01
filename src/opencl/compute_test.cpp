#include <opencl/cl_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// Unified compute benchmark -- replaces compute_sp/hp/dp/integer/intfast/char/short
// ---------------------------------------------------------------------------

int clPeak::runComputeTest(cl::CommandQueue &queue, cl::Program &prog,
                           device_info_t &devInfo, benchmark_config_t &cfg,
                           Benchmark which,
                           const std::string &displayName, const std::string &resultTag,
                           const std::string &kernelPrefix, const std::string &typeName,
                           const std::string &unit, const std::string &description,
                           unsigned int workPerWI,
                           unsigned int wgsPerCU, size_t elemSize)
{
  if (!isAllowed(which))
    return 0;

  // Vector width suffixes and display labels
  const int widths[] = {1, 2, 4, 8, 16};
  const char *suffixes[] = {"_v1", "_v2", "_v4", "_v8", "_v16"};

  // Build display names: "float", "float2", ... or "int", "int2", ...
  std::string labels[5];
  for (int w = 0; w < 5; w++)
  {
    labels[w] = typeName;
    if (widths[w] > 1)
      labels[w] += std::to_string(widths[w]);
  }

  auto test = currentDeviceScope->beginTest(
    {resultTag, displayName, unit, Category::Unknown, description,
     // Every test routed through here is one kernel at five vector widths.
     TestShape::Homogeneous, "vector width"});

  // Feature gates
  if (which == Benchmark::ComputeHP && !devInfo.halfSupported)
  {
    test.skipAll({labels[0], labels[1], labels[2], labels[3], labels[4]},
                 ResultStatus::Unsupported, "No half precision support");
    return 0;
  }
  if (which == Benchmark::ComputeMP && !devInfo.halfSupported)
  {
    test.skipAll({labels[0], labels[1], labels[2], labels[3], labels[4]},
                 ResultStatus::Unsupported, "No half precision support");
    return 0;
  }
  if (which == Benchmark::ComputeDP && !devInfo.doubleSupported)
  {
    test.skipAll({labels[0], labels[1], labels[2], labels[3], labels[4]},
                 ResultStatus::Unsupported, "No double precision support");
    return 0;
  }
  if (which == Benchmark::ComputeInt8DP && !devInfo.int8DotProductSupported)
  {
    test.skipAll({labels[0], labels[1], labels[2], labels[3], labels[4]},
                 ResultStatus::Unsupported,
                 "cl_khr_integer_dot_product not supported");
    return 0;
  }

  try
  {
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    uint64_t globalWIs = (uint64_t)devInfo.numCUs * wgsPerCU * devInfo.maxWGSize;
    uint64_t t = std::min(globalWIs * elemSize, devInfo.maxAllocSize) / elemSize;
    globalWIs = roundToMultipleOf(t, devInfo.maxWGSize);

    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, globalWIs * elemSize);

    // Create kernels and set arguments.  Each width also looks for an
    // affine-chain twin (compute_*_alt_v*, see kernels/mad_chain.cl); families
    // that do not define one simply race nothing.
    cl::Kernel kernels[5];
    cl::Kernel altKernels[5];
    bool hasAlt[5] = {false, false, false, false, false};
    for (int w = 0; w < 5; w++)
    {
      std::string kname = kernelPrefix + suffixes[w];
      kernels[w] = cl::Kernel(prog, kname.c_str());
      kernels[w].setArg(0, outputBuf);
      try
      {
        altKernels[w] = cl::Kernel(prog, (kernelPrefix + "_alt" + suffixes[w]).c_str());
        altKernels[w].setArg(0, outputBuf);
        hasAlt[w] = true;
      }
      catch (cl::Error &)
      {
        hasAlt[w] = false;
      }
      // Arg 1: scalar constant -- type depends on the test
      auto setScalarArg = [&](cl::Kernel &k) {
        if (which == Benchmark::ComputeDP)
        {
          cl_double A = 1.3;
          k.setArg(1, A);
        }
        else if (which == Benchmark::ComputeChar || which == Benchmark::ComputeInt8DP)
        {
          cl_char A = 4;
          k.setArg(1, A);
        }
        else if (which == Benchmark::ComputeShort)
        {
          cl_short A = 4;
          k.setArg(1, A);
        }
        else if (which == Benchmark::ComputeInt || which == Benchmark::ComputeIntFast)
        {
          cl_int A = 4;
          k.setArg(1, A);
        }
        else
        {
          // SP and HP both take cl_float
          cl_float A = 1.3f;
          k.setArg(1, A);
        }
      };
      setScalarArg(kernels[w]);
      if (hasAlt[w]) setScalarArg(altKernels[w]);
    }

    // Run each vector width. run_kernel clamps the local size to each kernel's
    // own work-group limit (wide vector widths can be capped by register
    // pressure), so widths run at whatever size the kernel actually supports.
    // Isolate per-width failures so one constrained width does not mark the
    // whole group as errored.
    for (int w = 0; w < 5; w++)
    {
      try
      {
        cl::NDRange globalSize = globalWIs;
        cl::NDRange localSize = devInfo.maxWGSize;

        float timed = run_kernel(queue, kernels[w], globalSize, localSize,
                                 cfg.targetTimeUs, forceIters ? specifiedIters : 0);
        float throughput = (static_cast<float>(ndRangeTotal(globalSize)) * static_cast<float>(workPerWI)) / timed * 1e6f;

        // Race the affine chain and keep the faster reading.  A failure here
        // is not an error: the squaring chain already produced one.
        if (hasAlt[w])
        {
          try
          {
            float altTimed = run_kernel(queue, altKernels[w], globalSize, localSize,
                                        cfg.targetTimeUs, forceIters ? specifiedIters : 0);
            float altThroughput = (static_cast<float>(ndRangeTotal(globalSize)) * static_cast<float>(workPerWI)) / altTimed * 1e6f;
            CLPEAK_VLOG("%s %s: squaring chain %.1f, alt chain %.1f %s\n",
                        resultTag.c_str(), labels[w].c_str(), throughput,
                        altThroughput, unit.c_str());
            if (altThroughput > throughput * MAX_ALT_CHAIN_RATIO)
              CLPEAK_VLOG("%s %s: alt chain %.1fx faster -- rejecting it as a "
                          "compiler fold\n", resultTag.c_str(), labels[w].c_str(),
                          altThroughput / throughput);
            else if (altThroughput > throughput)
              throughput = altThroughput;
          }
          catch (cl::Error &)
          {
          }
        }

        test.emit(labels[w], throughput, clWidthNote(widths[w]));
      }
      catch (cl::Error &error)
      {
        std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
        test.skip(labels[w], ResultStatus::Error, reason, clWidthNote(widths[w]));
      }
    }
  }
  catch (cl::Error &error)
  {
    // A missing kernel is a capability fact, not a failure: the device's
    // OpenCL compiler declined to provide the builtin the kernel needs, so
    // the whole family was preprocessed out.  int8_dp hits this on devices
    // that advertise cl_khr_integer_dot_product and report the 4x8-bit
    // capability but whose compiler defines neither the extension macro nor
    // the OpenCL 3.0 feature macro.  Reporting five errors there is noise.
    if (error.err() == CL_INVALID_KERNEL_NAME || error.err() == CL_INVALID_PROGRAM)
    {
      for (int w = 0; w < 5; w++)
        test.skip(labels[w], ResultStatus::Unsupported,
                  "device's OpenCL compiler did not build these kernels",
                  clWidthNote(widths[w]));
      return 0;
    }
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    for (int w = 0; w < 5; w++)
      test.skip(labels[w], ResultStatus::Error, reason, clWidthNote(widths[w]));
    return -1;
  }
  catch (std::exception &e)
  {
    for (int w = 0; w < 5; w++)
      test.skip(labels[w], ResultStatus::Error, e.what(), clWidthNote(widths[w]));
    return -1;
  }

  return 0;
}
