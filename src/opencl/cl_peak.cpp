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

clPeak::clPeak()
{
}

int clPeak::runAll()
{
  auto backendScope = log->beginBackend("OpenCL");
  try
  {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // Devices are numbered consecutively across platforms -- the index
    // --device takes, --list-devices prints and the document records -- so
    // OpenCL selects like every other backend.  The platform is still part
    // of the device's identity in the output; it is just not a selector.
    int deviceIndex = 0;

    for (size_t p = 0; p < platforms.size(); p++)
    {
      if (clpeak::cancelRequested())
        break;

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

      for (size_t d = 0; d < devices.size(); d++, deviceIndex++)
      {
        if (clpeak::cancelRequested())
          break;
        if (!isDeviceSelected(deviceIndex))
          continue;

        device_info_t devInfo = getDeviceInfo(devices[d]);
        benchmark_config_t cfg = benchmark_config_t::forDevice(devInfo.deviceType, devInfo.globalMemCacheSize);
        cfg.targetTimeUs = targetTimeUs;
        if (forceIters)
          cfg.kernelLatencyIters = specifiedIters;

        auto deviceScope = backendScope.beginDevice({
          devInfo.deviceName,
          platformName,
          devInfo.driverVersion,
          {
            {"Compute units", std::to_string(devInfo.numCUs)},
            {"Clock frequency", std::to_string(devInfo.maxClockFreq) + " MHz"},
          },
          static_cast<int>(p),
          deviceIndex
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
        auto buildAuxProg = [&](const std::string &src, const std::string &label,
                                const std::string &options = BUILD_OPTIONS) -> cl::Program
        {
          cl::Program p;
          try
          {
            cl::Program::Sources s(1, src);
            p = cl::Program(ctx, s);
            std::vector<cl::Device> dev = {devices[d]};
            p.build(dev, options.c_str());
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
        if (devInfo.int8DotProductSupported || devInfo.int8DotProductPackedSupported)
        {
          std::string int8BuildOptions = std::string(BUILD_OPTIONS) +
              (devInfo.int8DotProductPackedSupported ? " -DUSE_PACKED_DOT " : "");
          int8DpProg = buildAuxProg(clGetInt8DpKernels(), "INT8 dot-product compute", int8BuildOptions);
        }

        cl_command_queue_properties supportedQueueProps = devices[d].getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
        bool supportsProfilingQueue = (supportedQueueProps & CL_QUEUE_PROFILING_ENABLE) != 0;

        cl_command_queue_properties queueCreateProps = supportsProfilingQueue ? CL_QUEUE_PROFILING_ENABLE : 0;
        cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d], queueCreateProps);

        // ---- Compute (GFLOPS/TFLOPS + GOPS/TOPS) ---------------------------
        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeSP,
                       "Single-precision compute", "single_precision_compute",
                       "compute_sp", "float", "flops",
                       "Peak arithmetic speed of the device's compute units on 32-bit "
                       "fractional numbers -- the ordinary float type.  Nothing "
                       "touches memory, so only the arithmetic units limit the rate.",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_float));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeHP,
                       "Half-precision compute", "half_precision_compute",
                       "compute_hp", "half", "flops",
                       "Peak arithmetic speed on 16-bit fractional numbers -- half "
                       "the size of a normal float, and what graphics and on-device "
                       "AI mostly run on.",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_half));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeDP,
                       "Double-precision compute", "double_precision_compute",
                       "compute_dp", "double", "flops",
                       "Peak arithmetic speed on 64-bit fractional numbers, the "
                       "high-accuracy type scientific computing relies on.  Consumer "
                       "graphics parts deliberately run these many times slower than "
                       "32-bit.",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeDPWgsPerCU, sizeof(cl_double));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeMP,
                       "Mixed-precision compute fp16xfp16+fp32", "mixed_precision_compute",
                       "compute_mp", "mp", "flops",
                       "Peak speed when the device multiplies 16-bit numbers but keeps "
                       "the running total in 32 bits -- the accuracy-preserving "
                       "pattern AI code uses.",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_float));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeInt,
                       "Integer compute", "integer_compute",
                       "compute_integer", "int", "ops",
                       "Peak speed on 32-bit whole numbers -- the arithmetic behind "
                       "indexing, addressing and bit manipulation, which kernels do "
                       "alongside their fractional maths.",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeIntFast,
                       "Integer compute Fast 24bit", "integer_compute_fast",
                       "compute_intfast", "int", "ops",
                       "The same integer maths restricted to 24-bit values, which "
                       "some devices multiply on their faster floating-point hardware "
                       "instead.  Where this beats the plain integer row, the full "
                       "32-bit multiply is the slower path.",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeChar,
                       "Integer char (8bit) compute", "integer_compute_char",
                       "compute_char", "char", "ops",
                       "Peak speed on 8-bit whole numbers, the smallest integer type "
                       "-- worth knowing because image and quantized-AI work is full "
                       "of them.",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_char));

        runComputeTest(queue, prog, devInfo, cfg, Benchmark::ComputeShort,
                       "Integer short (16bit) compute", "integer_compute_short",
                       "compute_short", "short", "ops",
                       "Peak speed on 16-bit whole numbers -- the middle size, "
                       "between the 8-bit and 32-bit rows.",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_short));

        runComputeTest(queue, int8DpProg, devInfo, cfg, Benchmark::ComputeInt8DP,
                       "INT8 dot-product compute", "integer_compute_int8_dp",
                       "compute_int8_dp", "int8_dp", "ops",
                       "Peak speed of the 8-bit dot-product instruction, which "
                       "multiplies four pairs of small whole numbers and sums them in "
                       "one step -- the workhorse of quantized (compressed) neural "
                       "networks.",
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
          auto test = deviceScope.beginTest(
            {"kernel_launch_latency", "Kernel launch latency", "s",
             Category::Unknown,
             "The overhead of asking the device to do anything at all, measured "
             "with a kernel that does no work.  It is what small, frequent jobs "
             "pay before any of their own work begins."});
          test.skipAll({"dispatch", "roundtrip"}, ResultStatus::Unsupported,
                       "No profiling queue support");
        }

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

void clPeak::clampToKernelWG(const cl::Device &dev, cl::Kernel &kernel,
                             cl::NDRange &globalSize, cl::NDRange &localSize)
{
  // Driver picks the local size -- nothing to clamp.
  if (localSize.dimensions() == 0)
    return;

  size_t kernelWG = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(dev);
  if (kernelWG == 0)
    return;

  const size_t *l = static_cast<const size_t *>(localSize);
  if (l[0] <= kernelWG)
    return; // requested local size already fits this kernel

  const size_t *g = static_cast<const size_t *>(globalSize);
  size_t local = kernelWG;
  size_t global = (g[0] / local) * local;
  if (global == 0)
    global = local;

  globalSize = cl::NDRange(global);
  localSize  = cl::NDRange(local);
}

uint64_t clPeak::ndRangeTotal(const cl::NDRange &range)
{
  const size_t *s = static_cast<const size_t *>(range);
  cl_uint dims = range.dimensions();
  uint64_t total = dims ? 1 : 0;
  for (cl_uint i = 0; i < dims; i++)
    total *= (uint64_t)s[i];
  return total;
}

float clPeak::run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel,
                         cl::NDRange &globalSize, cl::NDRange &localSize,
                         unsigned int targetTimeUsLocal, unsigned int forcedIters)
{
  // Keep every launch within the kernel's own work-group limit.
  cl::Device dev = queue.getInfo<CL_QUEUE_DEVICE>();
  clampToKernelWG(dev, kernel, globalSize, localSize);

  // Time `n` dispatches batched into one submit; returns total time in us.
  // Used for both the calibration probe and the real timed run so the timing
  // methodology matches in both phases.
  auto runBatch = [&](unsigned int n) -> float {
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
  inv.id = kBackend;

  try
  {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    inv.available = !platforms.empty();
    if (!inv.available)
      inv.unavailableReason = "no platforms found";

    int deviceIndex = 0;  // consecutive across platforms, as runAll numbers them
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
          dev.index           = deviceIndex++;
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
  catch (cl::Error &error)
  {
    inv.available = false;
    inv.unavailableReason = std::string("no platforms found (") + error.what() +
                            " " + std::to_string(error.err()) + ")";
    inv.platforms.clear();
  }

  return inv;
}
