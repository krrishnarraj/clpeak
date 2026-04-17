#include <clpeak.h>
#include <cstring>

#define MSTRINGIFY(...) #__VA_ARGS__

static const std::string stringifiedKernels =
#include "global_bandwidth_kernels.cl"
#include "local_bandwidth_kernels.cl"
#include "atomic_throughput_kernels.cl"
#include "compute_sp_kernels.cl"
#include "compute_hp_kernels.cl"
#include "compute_dp_kernels.cl"
#include "compute_int24_kernels.cl"
#include "compute_integer_kernels.cl"
#include "compute_char_kernels.cl"
#include "compute_short_kernels.cl"
    ;

// Image kernels live in a separate program to avoid contaminating the main
// program with image/sampler resources on drivers that budget them globally
// (e.g. NVIDIA CUDA-OpenCL), which can cause CL_OUT_OF_RESOURCES for
// register-heavy kernels like global_bandwidth_v16.
static const std::string stringifiedImageKernels =
#include "image_bandwidth_kernels.cl"
    ;

#ifdef USE_STUB_OPENCL
// Prototype
extern "C"
{
  void stubOpenclReset();
}
#endif

clPeak::clPeak() : forcePlatform(false), forcePlatformName(false), forceDevice(false),
                   forceDeviceName(false), forceTest(false), forceIters(false), useEventTimer(false),
                   specifiedPlatform(0), specifiedDevice(0),
                   specifiedIters(0),
                   warmupCount(2),
                   enableJson(false), enableCsv(false),
                   listDevices(false)
{
  enableAll(); // all tests on by default
}

int clPeak::runAll()
{
  try
  {
#ifdef USE_STUB_OPENCL
    stubOpenclReset();
#endif
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // --list-devices mode: print platforms/devices and exit
    if (listDevices)
    {
      for (size_t p = 0; p < platforms.size(); p++)
      {
        std::string platformName = platforms[p].getInfo<CL_PLATFORM_NAME>();
        trimString(platformName);
        std::cout << "Platform " << p << ": " << platformName << "\n";

        cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platforms[p])(),
            0};
        cl::Context ctx(CL_DEVICE_TYPE_ALL, cps);
        std::vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();

        for (size_t d = 0; d < devices.size(); d++)
        {
          device_info_t info = getDeviceInfo(devices[d]);
          const char *typeStr = (info.deviceType & CL_DEVICE_TYPE_CPU) ? "CPU" :
                                (info.deviceType & CL_DEVICE_TYPE_GPU) ? "GPU" : "Other";
          std::cout << "  Device " << d << ": " << info.deviceName
                    << " [" << typeStr << "]"
                    << "\n";
          std::cout << "    Driver    : " << info.driverVersion << "\n";
          std::cout << "    CUs       : " << info.numCUs << "\n";
          std::cout << "    Clock     : " << info.maxClockFreq << " MHz\n";
          std::cout << "    Global mem: " << (info.maxGlobalSize / (1024*1024)) << " MB\n";
          std::cout << "    Max alloc : " << (info.maxAllocSize / (1024*1024)) << " MB\n";
          std::cout << "    FP16      : " << (info.halfSupported ? "yes" : "no") << "\n";
          std::cout << "    FP64      : " << (info.doubleSupported ? "yes" : "no") << "\n";
        }
      }
      return 0;
    }

    log->xmlOpenTag("clpeak");
    log->xmlAppendAttribs("os", OS_NAME);
    for (size_t p = 0; p < platforms.size(); p++)
    {
      if (forcePlatform && (p != specifiedPlatform))
        continue;

      std::string platformName = platforms[p].getInfo<CL_PLATFORM_NAME>();
      trimString(platformName);

      if (forcePlatformName && specifiedPlatformName != platformName)
        continue;

      log->print(NEWLINE "Platform: " + platformName + NEWLINE);
      log->xmlOpenTag("platform");
      log->xmlAppendAttribs("name", platformName);

      cl_context_properties cps[3] = {
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)(platforms[p])(),
          0};

      cl::Context ctx(CL_DEVICE_TYPE_ALL, cps);
      std::vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();

      for (size_t d = 0; d < devices.size(); d++)
      {
        if (forceDevice && (d != specifiedDevice))
          continue;

        device_info_t devInfo = getDeviceInfo(devices[d]);
        benchmark_config_t cfg = benchmark_config_t::forDevice(devInfo.deviceType);

        if (forceIters)
        {
          cfg.computeIters = specifiedIters;
          cfg.globalBWIters = specifiedIters;
          cfg.localBWIters = specifiedIters;
          cfg.imageBWIters = specifiedIters;
          cfg.transferBWIters = specifiedIters;
          cfg.kernelLatencyIters = specifiedIters;
        }

        if (forceDeviceName && specifiedDeviceName != devInfo.deviceName)
          continue;

        log->print(TAB "Device: " + devInfo.deviceName + NEWLINE);
        log->print(TAB TAB "Driver version  : ");
        log->print(devInfo.driverVersion);
        log->print(" (" OS_NAME ")" NEWLINE);
        log->print(TAB TAB "Compute units   : ");
        log->print(devInfo.numCUs);
        log->print(NEWLINE);
        log->print(TAB TAB "Clock frequency : ");
        log->print(devInfo.maxClockFreq);
        log->print(" MHz" NEWLINE);
        if (useEventTimer)
          log->print(TAB TAB "Note: --use-event-timer accuracy depends on platform OpenCL profiling implementation" NEWLINE);
        log->xmlOpenTag("device");
        log->xmlAppendAttribs("name", devInfo.deviceName);
        log->xmlAppendAttribs("driver_version", devInfo.driverVersion);
        log->xmlAppendAttribs("compute_units", devInfo.numCUs);
        log->xmlAppendAttribs("clock_frequency", devInfo.maxClockFreq);
        log->xmlAppendAttribs("clock_frequency_unit", "MHz");

        cl::Program::Sources source(1, stringifiedKernels);
        cl::Program prog = cl::Program(ctx, source);
        try
        {
          std::vector<cl::Device> dev = {devices[d]};
          prog.build(dev, BUILD_OPTIONS);
        }
        catch (cl::Error &error)
        {
          UNUSED(error);
          log->print(TAB TAB "Build Log: " + prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[d]) + NEWLINE NEWLINE);
          continue;
        }

        // Build a separate program for image kernels so that image/sampler
        // resources are not budgeted against the main program on drivers
        // that do global resource accounting (e.g. NVIDIA CUDA-OpenCL).
        cl::Program imgProg;
        bool imageProgReady = false;
        if (devInfo.imageSupported)
        {
          try
          {
            cl::Program::Sources imgSrc(1, stringifiedImageKernels);
            imgProg = cl::Program(ctx, imgSrc);
            std::vector<cl::Device> dev = {devices[d]};
            imgProg.build(dev, BUILD_OPTIONS);
            imageProgReady = true;
          }
          catch (cl::Error &)
          {
            log->print(TAB TAB "Image kernel build failed, image bandwidth test skipped" NEWLINE);
          }
        }

        cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d], CL_QUEUE_PROFILING_ENABLE);

        runGlobalBandwidthTest(queue, prog, devInfo, cfg);
        runLocalBandwidthTest(queue, prog, devInfo, cfg);
        {
          cl::Program &imgProgRef = imageProgReady ? imgProg : prog;
          runImageBandwidthTest(queue, imgProgRef, devInfo, cfg);
        }

        // Compute tests via unified helper
        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeSP,
                       "Single-precision compute (GFLOPS)", "single_precision_compute",
                       "compute_sp", "float", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_float));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeHP,
                       "Half-precision compute (GFLOPS)", "half_precision_compute",
                       "compute_hp", "half", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_half));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeDP,
                       "Double-precision compute (GFLOPS)", "double_precision_compute",
                       "compute_dp", "double", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeDPWgsPerCU, sizeof(cl_double));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeInt,
                       "Integer compute (GIOPS)", "integer_compute",
                       "compute_integer", "int", "giops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeIntFast,
                       "Integer compute Fast 24bit (GIOPS)", "integer_compute_fast",
                       "compute_intfast", "int", "giops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeChar,
                       "Integer char (8bit) compute (GIOPS)", "integer_compute_char",
                       "compute_char", "char", "giops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_char));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeShort,
                       "Integer short (16bit) compute (GIOPS)", "integer_compute_short",
                       "compute_short", "short", "giops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_short));

        runAtomicThroughputTest(queue, prog, devInfo, cfg);
        runTransferBandwidthTest(queue, prog, devInfo, cfg);
        runKernelLatency(queue, prog, devInfo, cfg);

        log->print(NEWLINE);
        log->xmlCloseTag(); // device
      }
      log->xmlCloseTag(); // platform
    }
    log->xmlCloseTag(); // clpeak
  }
  catch (cl::Error &error)
  {
    std::stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE;

    log->print(ss.str());

    // skip error for no platform
    if (error.err() == CL_INVALID_VALUE || error.err() == CL_PLATFORM_NOT_FOUND_KHR)
    {
      log->print("no platforms found" NEWLINE);
    }
    else
    {
      return -1;
    }
  }

  return 0;
}

float clPeak::run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, unsigned int iters)
{
  float timed = 0;

  // Warm-up runs
  for (unsigned int w = 0; w < warmupCount; w++)
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
  queue.finish();

  if (useEventTimer)
  {
    for (unsigned int i = 0; i < iters; i++)
    {
      cl::Event timeEvent;

      queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &timeEvent);
      queue.finish();
      timed += timeInUS(timeEvent);
    }
  }
  else // std timer
  {
    for (unsigned int i = 0; i < iters; i++)
    {
      Timer timer;
      timer.start();
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
      queue.finish();
      timed += timer.stopAndTime();
    }
  }

  return (timed / static_cast<float>(iters));
}

// ---------------------------------------------------------------------------
// Unified compute benchmark -- replaces compute_sp/hp/dp/integer/intfast/char/short
// ---------------------------------------------------------------------------

int clPeak::runComputeTest(cl::CommandQueue &queue, cl::Program &prog,
                           device_info_t &devInfo, benchmark_config_t &cfg,
                           Benchmark which,
                           const std::string &displayName, const std::string &xmlTag,
                           const std::string &kernelPrefix, const std::string &typeName,
                           const std::string &unit, unsigned int workPerWI,
                           unsigned int wgsPerCU, size_t elemSize)
{
  if (!isTestEnabled(which))
    return 0;

  // Feature gates
  if (which == Benchmark::ComputeHP && !devInfo.halfSupported)
  {
    log->print(NEWLINE TAB TAB "No half precision support! Skipped" NEWLINE);
    return 0;
  }
  if (which == Benchmark::ComputeDP && !devInfo.doubleSupported)
  {
    log->print(NEWLINE TAB TAB "No double precision support! Skipped" NEWLINE);
    return 0;
  }

  unsigned int iters = cfg.computeIters;

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

  try
  {
    log->print(NEWLINE TAB TAB + displayName + NEWLINE);
    log->xmlOpenTag(xmlTag);
    log->xmlAppendAttribs("unit", unit);

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
      else if (which == Benchmark::ComputeChar)
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
      if (forceTest && specifiedTestName != labels[w])
        continue;

      // Padding for aligned output
      std::string padded = labels[w];
      while (padded.size() < 8) padded += ' ';
      log->print(TAB TAB TAB + padded + ": ");

      float timed = run_kernel(queue, kernels[w], globalSize, localSize, iters);
      float throughput = (static_cast<float>(globalWIs) * static_cast<float>(workPerWI)) / timed / 1e3f;

      log->print(throughput);
      log->print(NEWLINE);
      log->xmlRecord(labels[w], throughput);
    }

    log->xmlCloseTag();
  }
  catch (cl::Error &error)
  {
    std::stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    return -1;
  }
  catch (std::exception &e)
  {
    std::stringstream ss;
    ss << "Exception: " << e.what() << NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    return -1;
  }

  return 0;
}
