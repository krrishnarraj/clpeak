#include <opencl/cl_peak.h>
#include <common/inventory.h>
#include <common/options.h>
#include <common/common.h>
#include <algorithm>
#include <cstring>

// Kernel strings live in cl_kernels.cpp — see clGetMainKernels(), etc.
// Benchmark methods live in separate files:
//   cl_kernels.cpp        compute_test.cpp
//   global_bandwidth.cpp  local_bandwidth.cpp  image_bandwidth.cpp
//   transfer_bandwidth.cpp kernel_latency.cpp
//   cl_common.cpp         cl_utils.cpp

clPeak::clPeak() : useEventTimer(false)
{
}

void clPeak::applyOptions(const CliOptions &opts)
{
    // Common fields handled by base class
    Peak::applyOptions(opts);

    // OpenCL-specific device selection
    platformIndices = opts.platformIndices;
    deviceIndices   = opts.deviceIndices;
    useEventTimer   = opts.useEventTimer;
}

int clPeak::runAll()
{
  auto backendScope = log->beginBackend("OpenCL");
  try
  {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (size_t p = 0; p < platforms.size(); p++)
    {
      if (!platformIndices.empty() &&
          std::find(platformIndices.begin(), platformIndices.end(),
                    static_cast<unsigned long>(p)) == platformIndices.end())
        continue;

      std::string platformName = platforms[p].getInfo<CL_PLATFORM_NAME>();
      trimString(platformName);

      cl_context_properties cps[3] = {
          CL_CONTEXT_PLATFORM,
          (cl_context_properties)(platforms[p])(),
          0};

      cl::Context ctx;
      std::vector<cl::Device> devices;
      try
      {
        ctx = cl::Context(CL_DEVICE_TYPE_ALL, cps);
        devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
      }
      catch (cl::Error &error)
      {
        log->note("  Platform \"" + platformName + "\": " + error.what() + " (" + std::to_string(error.err()) + ") — no devices, skipping\n");
        continue;
      }

      for (size_t d = 0; d < devices.size(); d++)
      {
        if (!deviceIndices.empty() &&
            std::find(deviceIndices.begin(), deviceIndices.end(),
                      static_cast<unsigned long>(d)) == deviceIndices.end())
          continue;

        device_info_t devInfo = getDeviceInfo(devices[d]);
        benchmark_config_t cfg = benchmark_config_t::forDevice(devInfo.deviceType);
        cfg.targetTimeUs = targetTimeUs;
        if (forceIters)
          cfg.kernelLatencyIters = specifiedIters;

        if (useEventTimer)
          log->note("  Note: --use-event-timer accuracy depends on platform OpenCL profiling implementation\n");

        auto deviceScope = backendScope.beginDevice({
          devInfo.deviceName,
          platformName,
          devInfo.driverVersion,
          {
            {"Compute units", std::to_string(devInfo.numCUs)},
            {"Clock frequency", std::to_string(devInfo.maxClockFreq) + " MHz"},
          },
          static_cast<int>(p),
          static_cast<int>(d)
        });
        currentDeviceScope = &deviceScope;

        cl::Program::Sources source(1, clGetMainKernels());
        cl::Program prog = cl::Program(ctx, source);
        try
        {
          std::vector<cl::Device> dev = {devices[d]};
          prog.build(dev, BUILD_OPTIONS);
        }
        catch (cl::Error &error)
        {
          UNUSED(error);
          CLPEAK_VLOG("  Build Log: %s\n\n",
                      prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[d]).c_str());
          currentDeviceScope = nullptr;
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
            CLPEAK_VLOG("  %s kernel build failed, test skipped\n", label.c_str());
            p = cl::Program(); // return empty/invalid program
          }
          return p;
        };

        // Local-BW kernels use __local pointer arguments.
        // Image kernels use image2d_t / sampler_t.
        // All three cause NVIDIA CUDA-OpenCL to reserve module-level resources
        // that compress the register budget for every other kernel in the same
        // program, triggering CL_OUT_OF_RESOURCES on the v16 kernels.
        // Each gets its own isolated program object.
        cl::Program localProg = buildAuxProg(clGetLocalKernels(), "Local bandwidth");
        cl::Program imgProg;
        if (devInfo.imageSupported)
          imgProg = buildAuxProg(clGetImageKernels(), "Image bandwidth");

        cl::Program int8DpProg;
        if (devInfo.int8DotProductSupported)
          int8DpProg = buildAuxProg(clGetInt8DpKernels(), "INT8 dot-product compute");

        cl_command_queue_properties supportedQueueProps = devices[d].getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
        bool supportsProfilingQueue = (supportedQueueProps & CL_QUEUE_PROFILING_ENABLE) != 0;

        cl_command_queue_properties queueCreateProps = supportsProfilingQueue ? CL_QUEUE_PROFILING_ENABLE : 0;
        cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d], queueCreateProps);

        bool savedUseEventTimer = useEventTimer;
        if (!supportsProfilingQueue)
        {
          if (useEventTimer)
          {
            log->note("  NOTE: Device does not support profiling queue, --use-event-timer disabled\n");
          }
          useEventTimer = false;
        }

        // ---- Phase 1: floating-point compute ---------------------------
        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeSP,
                       "Single-precision compute", "single_precision_compute",
                       "compute_sp", "float", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_float));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeHP,
                       "Half-precision compute", "half_precision_compute",
                       "compute_hp", "half", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_half));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeDP,
                       "Double-precision compute", "double_precision_compute",
                       "compute_dp", "double", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeDPWgsPerCU, sizeof(cl_double));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeMP,
                       "Mixed-precision compute fp16xfp16+fp32", "mixed_precision_compute",
                       "compute_mp", "mp", "gflops",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_float));

        // ---- Phase 2: integer compute ----------------------------------
        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeInt,
                       "Integer compute", "integer_compute",
                       "compute_integer", "int", "gops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeIntFast,
                       "Integer compute Fast 24bit", "integer_compute_fast",
                       "compute_intfast", "int", "gops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeChar,
                       "Integer char (8bit) compute", "integer_compute_char",
                       "compute_char", "char", "gops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_char));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeShort,
                       "Integer short (16bit) compute", "integer_compute_short",
                       "compute_short", "short", "gops",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_short));

        runComputeTest(queue, int8DpProg, devInfo, cfg, Benchmark::ComputeInt8DP,
                       "INT8 dot-product compute", "integer_compute_int8_dp",
                       "compute_int8_dp", "int8_dp", "gops",
                       COMPUTE_INT8_DP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        // ---- Phase 3: bandwidth ----------------------------------------
        runGlobalBandwidthTest(queue, prog, devInfo, cfg);
        runLocalBandwidthTest(queue, localProg, devInfo, cfg);
        runImageBandwidthTest(queue, imgProg, devInfo, cfg);
        runTransferBandwidthTest(queue, prog, devInfo, cfg);

        // ---- Phase 4: latency ------------------------------------------
        if (supportsProfilingQueue)
          runKernelLatency(queue, prog, devInfo, cfg);
        else if (isAllowed(Benchmark::KernelLatency))
        {
          auto test = deviceScope.beginTest({"kernel_launch_latency",
                                             "Kernel launch latency", "us"});
          test.skipAll({"dispatch", "roundtrip"}, ResultStatus::Unsupported,
                       "No profiling queue support");
        }

        useEventTimer = savedUseEventTimer;

        currentDeviceScope = nullptr;
      }
    }
  }
  catch (cl::Error &error)
  {
    std::stringstream ss;
    ss << error.what() << " (" << error.err() << ")";
    log->note(ss.str() + "\n");

    // skip error for no platform
    if (error.err() == CL_INVALID_VALUE || error.err() == CL_PLATFORM_NOT_FOUND_KHR)
    {
      log->note("no platforms found\n");
    }
    else
    {
      return -1;
    }
  }

  return 0;
}

float clPeak::run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel,
                         cl::NDRange &globalSize, cl::NDRange &localSize,
                         unsigned int targetTimeUsLocal, unsigned int forcedIters)
{
  // Time `n` dispatches batched into one submit; returns total time in us.
  // Used for both the calibration probe and the real timed run so the timing
  // methodology matches in both phases.
  auto runBatch = [&](unsigned int n) -> float {
    if (useEventTimer)
    {
      float total = 0;
      for (unsigned int i = 0; i < n; i++)
      {
        cl::Event timeEvent;
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, nullptr, &timeEvent);
        queue.finish();
        total += timeInUS(timeEvent);
      }
      return total;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    for (unsigned int i = 0; i < n; i++)
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.finish();
    auto t2 = std::chrono::high_resolution_clock::now();
    return (float)std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  };

  // Phase 1: untimed warmup (cache + clock ramp). Keep each warmup as its own
  // completed submission so slow kernels do not get batched before calibration.
  for (unsigned int w = 0; w < warmupCount; w++)
  {
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.finish();
  }

  // Phase 2: timed calibration probe. Keep this to one dispatch so warmupCount
  // does not force a multi-dispatch submit on slow kernels.
  unsigned int probeIters = 1;
  float probeUs = runBatch(probeIters);
  double per_iter_us = (double)probeUs / (double)probeIters;

  // Phase 3: real timed run with calibrated iter count.
  unsigned int iters = pickIters(per_iter_us, targetTimeUsLocal, forcedIters);
  float timed = runBatch(iters);
  return (timed / static_cast<float>(iters));
}

BackendInventory clPeak::enumerate()
{
  BackendInventory inv;
  inv.backend = "OpenCL";

  try
  {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    inv.available = !platforms.empty();

    for (size_t p = 0; p < platforms.size(); p++)
    {
      InventoryPlatform plat;
      plat.index = static_cast<int>(p);
      plat.name  = platforms[p].getInfo<CL_PLATFORM_NAME>();
      trimString(plat.name);

      try
      {
        cl_context_properties cps[3] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platforms[p])(),
            0};
        cl::Context ctx(CL_DEVICE_TYPE_ALL, cps);
        std::vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();

        for (size_t d = 0; d < devices.size(); d++)
        {
          device_info_t info = getDeviceInfo(devices[d]);
          InventoryDevice dev;
          dev.index           = static_cast<int>(d);
          dev.name            = info.deviceName;
          dev.typeStr         = (info.clDeviceType & CL_DEVICE_TYPE_CPU) ? "CPU"
                              : (info.clDeviceType & CL_DEVICE_TYPE_GPU) ? "GPU"
                              : (info.clDeviceType & CL_DEVICE_TYPE_ACCELERATOR) ? "Accelerator"
                                                                       : "Other";
          dev.driverVersion   = info.driverVersion;
          dev.numComputeUnits = info.numCUs;
          dev.maxClockMHz     = info.maxClockFreq;
          dev.globalMemBytes  = info.maxGlobalSize;
          dev.maxAllocBytes   = info.maxAllocSize;
          dev.hasFp16         = info.halfSupported;
          dev.hasFp64         = info.doubleSupported;
          plat.devices.push_back(std::move(dev));
        }
      }
      catch (cl::Error &)
      {
        // No usable devices on this platform — leave plat.devices empty.
      }

      inv.platforms.push_back(std::move(plat));
    }
  }
  catch (cl::Error &)
  {
    inv.available = false;
    inv.platforms.clear();
  }

  return inv;
}

void clPeak::printInventory(const BackendInventory &b, std::ostream &os)
{
    os << "\n=== OpenCL backend ===\n";
    for (const auto &plat : b.platforms)
    {
        os << "Platform " << plat.index << ": " << plat.name << "\n";
        for (const auto &d : plat.devices)
        {
            os << "  Device " << d.index << ": " << d.name;
            if (!d.typeStr.empty())
                os << " [" << d.typeStr << "]";
            os << "\n";
            if (!d.driverVersion.empty())
                os << "    Driver    : " << d.driverVersion << "\n";
            if (d.numComputeUnits)
                os << "    CUs       : " << d.numComputeUnits << "\n";
            if (d.maxClockMHz)
                os << "    Clock     : " << d.maxClockMHz << " MHz\n";
            if (d.globalMemBytes)
                os << "    Global mem: " << (d.globalMemBytes / (1024 * 1024)) << " MB\n";
            if (d.maxAllocBytes)
                os << "    Max alloc : " << (d.maxAllocBytes / (1024 * 1024)) << " MB\n";
            os << "    FP16      : " << (d.hasFp16 ? "yes" : "no") << "\n";
            os << "    FP64      : " << (d.hasFp64 ? "yes" : "no") << "\n";
        }
    }
}
