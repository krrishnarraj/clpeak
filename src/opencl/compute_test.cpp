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
                           const std::string &unit, unsigned int workPerWI,
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

  auto test = currentDeviceScope->beginTest({resultTag, displayName, unit});

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

    cl::NDRange globalSize = globalWIs;
    cl::NDRange localSize = devInfo.maxWGSize;

    // Create kernels and set arguments
    cl::Kernel kernels[5];
    for (int w = 0; w < 5; w++)
    {
      std::string kname = kernelPrefix + suffixes[w];
      kernels[w] = cl::Kernel(prog, kname.c_str());
      kernels[w].setArg(0, outputBuf);
      // Arg 1: scalar constant -- type depends on the test
      if (which == Benchmark::ComputeDP)
      {
        cl_double A = 1.3;
        kernels[w].setArg(1, A);
      }
      else if (which == Benchmark::ComputeChar || which == Benchmark::ComputeInt8DP)
      {
        cl_char A = 4;
        kernels[w].setArg(1, A);
      }
      else if (which == Benchmark::ComputeShort)
      {
        cl_short A = 4;
        kernels[w].setArg(1, A);
      }
      else if (which == Benchmark::ComputeInt || which == Benchmark::ComputeIntFast)
      {
        cl_int A = 4;
        kernels[w].setArg(1, A);
      }
      else
      {
        // SP and HP both take cl_float
        cl_float A = 1.3f;
        kernels[w].setArg(1, A);
      }
    }

    // Run each vector width
    for (int w = 0; w < 5; w++)
    {
      float timed = run_kernel(queue, kernels[w], globalSize, localSize,
                               cfg.targetTimeUs, forceIters ? specifiedIters : 0);
      float throughput = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      test.emit(labels[w], throughput);
    }
  }
  catch (cl::Error &error)
  {
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    for (int w = 0; w < 5; w++)
      test.skip(labels[w], ResultStatus::Error, reason);
    return -1;
  }
  catch (std::exception &e)
  {
    for (int w = 0; w < 5; w++)
      test.skip(labels[w], ResultStatus::Error, e.what());
    return -1;
  }

  return 0;
}
