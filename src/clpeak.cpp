#include <clpeak.h>
#include <options.h>
#include <cstring>

#define MSTRINGIFY(...) #__VA_ARGS__

// Main program: only kernels without __local pointer arguments.
// Keeping this set identical to what shipped in master ensures that
// NVIDIA CUDA-OpenCL's module compiler does not reserve dynamic shared-memory
// resources, which would shrink the register budget and break the v16 kernels
// (global_bandwidth_v16 / compute_dp_v16) with CL_OUT_OF_RESOURCES.
static const std::string stringifiedKernels =
#include "global_bandwidth_kernels.cl"
#include "compute_sp_kernels.cl"
#include "compute_hp_kernels.cl"
#include "compute_mp_kernels.cl"
#include "compute_dp_kernels.cl"
#include "compute_int24_kernels.cl"
#include "compute_integer_kernels.cl"
#include "compute_char_kernels.cl"
#include "compute_short_kernels.cl"
#include "compute_int4_packed_kernels.cl"
#ifdef CLPEAK_HAS_OPENCL_30
#include "compute_int8_dp_kernels.cl"
#endif
    ;

// Separate programs for kernels that use __local pointer arguments or
// image/sampler types.  On NVIDIA CUDA-OpenCL these force module-level
// resource reservations that spill into every other kernel in the same
// program, so they must be isolated from the main benchmark kernels.
static const std::string stringifiedLocalKernels =
#include "local_bandwidth_kernels.cl"
    ;

static const std::string stringifiedAtomicKernels =
#include "atomic_throughput_kernels.cl"
    ;

static const std::string stringifiedImageKernels =
#include "image_bandwidth_kernels.cl"
    ;

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

void clPeak::applyOptions(const CliOptions &opts)
{
  forcePlatform = opts.forcePlatform;
  specifiedPlatform = opts.platformIndex;
  forcePlatformName = opts.forcePlatformName;
  specifiedPlatformName = opts.platformName;
  forceDevice = opts.forceDevice;
  specifiedDevice = opts.deviceIndex;
  forceDeviceName = opts.forceDeviceName;
  specifiedDeviceName = opts.deviceName;

  forceIters = opts.forceIters;
  specifiedIters = opts.iters;
  warmupCount = opts.warmupCount;

  useEventTimer = opts.useEventTimer;
  listDevices = opts.listDevices;

  // Test selection: copy bitset directly.
  enabledTests = opts.enabledTests;
  forceTest = false;
  specifiedTestName.clear();

  enableJson = opts.enableJson;
  jsonFileName = opts.jsonFile;
  enableCsv = opts.enableCsv;
  csvFileName = opts.csvFile;
  compareFileName = opts.compareFile;

  log.reset(new logger(opts.enableXml, opts.xmlFile,
                       opts.enableJson, opts.jsonFile,
                       opts.enableCsv, opts.csvFile,
                       opts.compareFile));
}

int clPeak::runAll()
{
  try
  {
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
          const char *typeStr = (info.deviceType & CL_DEVICE_TYPE_CPU) ? "CPU" : (info.deviceType & CL_DEVICE_TYPE_GPU) ? "GPU"
                                                                                                                        : "Other";
          std::cout << "  Device " << d << ": " << info.deviceName
                    << " [" << typeStr << "]"
                    << "\n";
          std::cout << "    Driver    : " << info.driverVersion << "\n";
          std::cout << "    CUs       : " << info.numCUs << "\n";
          std::cout << "    Clock     : " << info.maxClockFreq << " MHz\n";
          std::cout << "    Global mem: " << (info.maxGlobalSize / (1024 * 1024)) << " MB\n";
          std::cout << "    Max alloc : " << (info.maxAllocSize / (1024 * 1024)) << " MB\n";
          std::cout << "    FP16      : " << (info.halfSupported ? "yes" : "no") << "\n";
          std::cout << "    FP64      : " << (info.doubleSupported ? "yes" : "no") << "\n";
        }
      }
      return 0;
    }

    log->print(NEWLINE "=== OpenCL backend ===" NEWLINE);
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
      log->xmlAppendAttribs("backend", "OpenCL");

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

        // Helper: build an auxiliary program, silently skip on failure.
        auto buildAuxProg = [&](const std::string &src, const std::string &label) -> cl::Program
        {
          cl::Program p;
          try
          {
            cl::Program::Sources s(1, src);
            p = cl::Program(ctx, s);
            std::vector<cl::Device> dev = {devices[d]};
            p.build(dev, BUILD_OPTIONS);
          }
          catch (cl::Error &)
          {
            log->print(TAB TAB + label + " kernel build failed, test skipped" NEWLINE);
            p = cl::Program(); // return empty/invalid program
          }
          return p;
        };

        // Local-BW and atomic kernels use __local pointer arguments.
        // Image kernels use image2d_t / sampler_t.
        // All three cause NVIDIA CUDA-OpenCL to reserve module-level resources
        // that compress the register budget for every other kernel in the same
        // program, triggering CL_OUT_OF_RESOURCES on the v16 kernels.
        // Each gets its own isolated program object.
        cl::Program localProg = buildAuxProg(stringifiedLocalKernels, "Local bandwidth");
        cl::Program atomicProg = buildAuxProg(stringifiedAtomicKernels, "Atomic throughput");
        cl::Program imgProg;
        if (devInfo.imageSupported)
          imgProg = buildAuxProg(stringifiedImageKernels, "Image bandwidth");

        cl_command_queue_properties supportedQueueProps = devices[d].getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
        bool supportsProfilingQueue = (supportedQueueProps & CL_QUEUE_PROFILING_ENABLE) != 0;

        cl_command_queue_properties queueCreateProps = supportsProfilingQueue ? CL_QUEUE_PROFILING_ENABLE : 0;
        cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d], queueCreateProps);

        bool savedUseEventTimer = useEventTimer;
        if (!supportsProfilingQueue)
        {
          if (useEventTimer)
          {
            log->print(TAB TAB "NOTE: Device does not support profiling queue, --use-event-timer disabled" NEWLINE);
          }
          useEventTimer = false;
        }

        runGlobalBandwidthTest(queue, prog, devInfo, cfg);
        runLocalBandwidthTest(queue, localProg, devInfo, cfg);
        runImageBandwidthTest(queue, imgProg, devInfo, cfg);

        // Compute tests via unified helper (main program — no __local args)
        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeSP,
                       "Single-precision compute (GFLOPS)", "single_precision_compute",
                       "compute_sp", "float", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_float));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeHP,
                       "Half-precision compute (GFLOPS)", "half_precision_compute",
                       "compute_hp", "half", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_half));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeMP,
                       "Mixed-precision compute fp16xfp16+fp32 (GFLOPS)", "mixed_precision_compute",
                       "compute_mp", "mp", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_float));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeDP,
                       "Double-precision compute (GFLOPS)", "double_precision_compute",
                       "compute_dp", "double", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeDPWgsPerCU, sizeof(cl_double));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeInt,
                       "Integer compute (GOPS)", "integer_compute",
                       "compute_integer", "int", "gops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeIntFast,
                       "Integer compute Fast 24bit (GOPS)", "integer_compute_fast",
                       "compute_intfast", "int", "gops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeChar,
                       "Integer char (8bit) compute (GOPS)", "integer_compute_char",
                       "compute_char", "char", "gops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_char));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeShort,
                       "Integer short (16bit) compute (GOPS)", "integer_compute_short",
                       "compute_short", "short", "gops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_short));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeInt4Packed,
                       "Packed INT4 compute (emulated) (GOPS)", "int4_packed_compute",
                       "compute_int4_packed", "int4_packed", "gops",
                       COMPUTE_INT4_PACKED_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_char));

#ifdef CLPEAK_HAS_OPENCL_30
        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeInt8DP,
                       "INT8 dot-product compute (GOPS)", "integer_compute_int8_dp",
                       "compute_int8_dp", "int8_dp", "gops",
                       COMPUTE_INT8_DP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));
#endif

        runAtomicThroughputTest(queue, atomicProg, devInfo, cfg);
        runTransferBandwidthTest(queue, prog, devInfo, cfg);
        if (supportsProfilingQueue)
          runKernelLatency(queue, prog, devInfo, cfg);
        else if (isTestEnabled(Benchmark::KernelLatency))
          log->print(NEWLINE TAB TAB "Kernel launch latency         : Skipped (no profiling queue support)" NEWLINE);

        useEventTimer = savedUseEventTimer;

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
  if (which == Benchmark::ComputeMP && !devInfo.halfSupported)
  {
    log->print(NEWLINE TAB TAB "Mixed-precision compute fp16xfp16+fp32 (GFLOPS)" NEWLINE);
    log->print(TAB TAB TAB "No half precision support! Skipped" NEWLINE);
    return 0;
  }
  if (which == Benchmark::ComputeDP && !devInfo.doubleSupported)
  {
    log->print(NEWLINE TAB TAB "No double precision support! Skipped" NEWLINE);
    return 0;
  }
  if (which == Benchmark::ComputeInt8DP && !devInfo.int8DotProductSupported)
  {
    log->print(NEWLINE TAB TAB + displayName + NEWLINE);
    log->print(TAB TAB TAB "cl_khr_integer_dot_product not supported! Skipped" NEWLINE);
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
      else if (which == Benchmark::ComputeChar || which == Benchmark::ComputeInt8DP ||
               which == Benchmark::ComputeInt4Packed)
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
      while (padded.size() < 8)
        padded += ' ';
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
    // xmlOpenTag was already pushed above; close it so subsequent tests
    // don't nest under a leaked parent (manifests on Android as all later
    // tests collapsing into this test's result card).
    log->xmlCloseTag();
    return -1;
  }
  catch (std::exception &e)
  {
    std::stringstream ss;
    ss << "Exception: " << e.what() << NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
    log->xmlCloseTag();
    return -1;
  }

  return 0;
}
