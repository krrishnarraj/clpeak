#include <opencl/cl_peak.h>
#include <common/common.h>
#include <common/form_race.h>
#include <cstdio>
#include <string>

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
  if (which == Benchmark::ComputeInt8DP &&
      !devInfo.int8DotProductSupported && !devInfo.int8DotProductPackedSupported)
  {
    test.skipAll({labels[0], labels[1], labels[2], labels[3], labels[4]},
                 ResultStatus::Unsupported,
                 "integer dot product (4x8) not supported");
    return 0;
  }

  try
  {
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    uint64_t globalWIs = (uint64_t)devInfo.numCUs * wgsPerCU * devInfo.maxWGSize;
    uint64_t t = std::min(globalWIs * elemSize, devInfo.maxAllocSize) / elemSize;
    globalWIs = roundToMultipleOf(t, devInfo.maxWGSize);

    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, globalWIs * elemSize);

    // Create kernels and set arguments.  Each width also looks for its
    // second-shape twins (see kernels/mad_chain.cl): compute_*_alt_v*, and for
    // the float families on a device that offers it compute_*_alt_sg16_v*,
    // the same affine chain pinned to sub-group 16.  A family that defines
    // neither simply races nothing.
    cl::Kernel kernels[5];
    cl::Kernel altKernels[5];
    cl::Kernel sg16Kernels[5];
    bool hasAlt[5] = {false, false, false, false, false};
    bool hasSg16[5] = {false, false, false, false, false};
    // Looked up by name first: some runtimes (Apple's) print an error for
    // every clCreateKernel on a name the program lacks, and most families lack
    // the twins.
    std::string programKernels;
    try
    {
      programKernels = ";" + prog.getInfo<CL_PROGRAM_KERNEL_NAMES>() + ";";
    }
    catch (cl::Error &)
    {
    }
    auto findTwin = [&](const std::string &name, cl::Kernel &k) -> bool
    {
      if (programKernels.size() > 2 &&
          programKernels.find(";" + name + ";") == std::string::npos)
        return false;
      try
      {
        k = cl::Kernel(prog, name.c_str());
        k.setArg(0, outputBuf);
        return true;
      }
      catch (cl::Error &)
      {
        return false;
      }
    };
    for (int w = 0; w < 5; w++)
    {
      std::string kname = kernelPrefix + suffixes[w];
      kernels[w] = cl::Kernel(prog, kname.c_str());
      kernels[w].setArg(0, outputBuf);
      hasAlt[w] = findTwin(kernelPrefix + "_alt" + suffixes[w], altKernels[w]);
      hasSg16[w] = findTwin(kernelPrefix + "_alt_sg16" + suffixes[w], sg16Kernels[w]);
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
      if (hasSg16[w]) setScalarArg(sg16Kernels[w]);
    }

    // Which width the compiler built a kernel at.  On Intel GPUs the preferred
    // multiple is the SIMD width it chose, and the chain shapes' rates depend
    // on it (the MAD chain block in include/common/common.h), so the --verbose
    // race line carries it.
    cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();
    auto simdHint = [&](cl::Kernel &k) -> size_t
    {
      try
      {
        return k.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);
      }
      catch (cl::Error &)
      {
        return 0;
      }
    };

    // The affine chain at the compiler's sub-group size and at 16 race as a
    // FormRace that drops only a clear loser: a tie at one vector width says
    // nothing about the next, because the register allocator lays every width
    // out afresh -- that is how Alchemist's mixed-precision chain tied at width
    // 4 and then read 16-28% low at 8 and 16.  See kernels/mad_chain.cl.
    clpeak::FormRace widthRace;

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

        // Race the second shapes and keep the fastest reading.  A failure
        // here is not an error: the squaring chain already produced one.
        auto timeTwin = [&](cl::Kernel &k) -> float
        {
          try
          {
            float twinTimed = run_kernel(queue, k, globalSize, localSize,
                                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
            return (static_cast<float>(ndRangeTotal(globalSize)) * static_cast<float>(workPerWI)) / twinTimed * 1e6f;
          }
          catch (cl::Error &)
          {
            return 0.0f;
          }
        };
        double twin[2] = {0.0, 0.0};
        if (hasAlt[w] && widthRace.runs(false))
          twin[0] = timeTwin(altKernels[w]);
        if (hasSg16[w] && widthRace.runs(true))
          twin[1] = timeTwin(sg16Kernels[w]);
        if (clpeak::verboseEnabled() && (twin[0] > 0.0 || twin[1] > 0.0))
        {
          std::string alts;
          for (int f = 0; f < 2; f++)
          {
            if (twin[f] <= 0.0)
              continue;
            char reading[64];
            snprintf(reading, sizeof(reading), "%s%.1f%s", alts.empty() ? "" : ", ",
                     twin[f], !hasSg16[w] ? "" : f ? " (sub-group 16)" : " (compiler's sub-group)");
            alts += reading;
          }
          std::string multiples = std::to_string(simdHint(kernels[w])) + "/" +
                                  std::to_string(simdHint(altKernels[w]));
          if (hasSg16[w])
            multiples += "/" + std::to_string(simdHint(sg16Kernels[w]));
          CLPEAK_VLOG("%s %s: squaring chain %.1f, alt chain %s %s; work-group "
                      "multiples %s\n",
                      resultTag.c_str(), labels[w].c_str(), throughput, alts.c_str(),
                      unit.c_str(), multiples.c_str());
        }
        // The fold guard first, so a reading it rejects cannot decide the race.
        const float squaring = throughput;
        double raced[2] = {twin[0], twin[1]};
        for (int f = 0; f < 2; f++)
        {
          if (twin[f] > squaring * MAX_ALT_CHAIN_RATIO)
          {
            CLPEAK_VLOG("%s %s: alt chain %.1fx faster -- rejecting it as a "
                        "compiler fold\n", resultTag.c_str(), labels[w].c_str(),
                        twin[f] / squaring);
            raced[f] = 0.0;
          }
          else if (twin[f] > throughput)
            throughput = (float)twin[f];
        }
        if (hasSg16[w])
        {
          if (widthRace.runs(true) && twin[1] <= 0.0)
            widthRace.drop(true);
          if (widthRace.runs(false) && widthRace.runs(true))
          {
            widthRace.dropTrailing(raced);
            if (!widthRace.runs(false) || !widthRace.runs(true))
              CLPEAK_VLOG("%s %s: alt chain at the %s sub-group trails by %.1fx -- "
                          "dropped from here\n",
                          resultTag.c_str(), labels[w].c_str(),
                          widthRace.runs(true) ? "compiler's" : "16",
                          widthRace.runs(true) ? twin[1] / twin[0] : twin[0] / twin[1]);
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
