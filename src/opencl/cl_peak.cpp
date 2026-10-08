#include <opencl/cl_peak.h>
#include <common/inventory.h>
#include <common/options.h>
#include <common/common.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>

// Kernel strings live in cl_kernels.cpp — see clGetTestKernels().
// Benchmark methods live in separate files:
//   cl_kernels.cpp        compute_test.cpp
//   global_bandwidth.cpp  local_bandwidth.cpp  image_bandwidth.cpp
//   transfer_bandwidth.cpp kernel_latency.cpp
//   cl_common.cpp         cl_utils.cpp

clPeak::clPeak()
{
}

namespace
{

// The context's error callback.  A driver reports a failing command here
// -- NVIDIA's "CL_OUT_OF_RESOURCES error executing CL_COMMAND_NDRANGE_KERNEL
// on ..." -- with more than the status code the call returned, and the run
// log is where that reaches a file.  May arrive on a driver thread; the log
// route is built for it.
void CL_CALLBACK contextNotify(const char *errinfo, const void *, size_t, void *)
{
  if (errinfo)
    clpeak::logMessage(clpeak::LogLevel::Error, "opencl", errinfo);
}

// Whether the device takes intel_reqd_sub_group_size(16): the extension, and
// 16 among CL_DEVICE_SUB_GROUP_SIZES_INTEL.  The query is the extension's own,
// so it goes through the C API.
bool offersSubGroup16(const cl::Device &device)
{
  try
  {
    if (device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_intel_required_subgroup_size") ==
        std::string::npos)
      return false;
  }
  catch (cl::Error &)
  {
    return false;
  }
  const cl_device_info kSubGroupSizesIntel = 0x4108;
  size_t bytes = 0;
  if (clGetDeviceInfo(device(), kSubGroupSizesIntel, 0, nullptr, &bytes) != CL_SUCCESS ||
      bytes == 0)
    return false;
  std::vector<size_t> sizes(bytes / sizeof(size_t));
  if (clGetDeviceInfo(device(), kSubGroupSizesIntel, bytes, sizes.data(), nullptr) !=
      CL_SUCCESS)
    return false;
  return std::find(sizes.begin(), sizes.end(), (size_t)16) != sizes.end();
}

} // namespace

int clPeak::runAll()
{
  auto backendScope = log->beginBackend("OpenCL");
  try
  {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // Devices are numbered consecutively across platforms -- the index
    // --devices takes, --list-devices prints and the document records -- so
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
        ctx = cl::Context(CL_DEVICE_TYPE_ALL, cps, contextNotify);
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
          deviceIndex,
          devInfo.deviceType
        });
        currentDeviceScope = &deviceScope;

        // Every program the selected tests run, one per test (clGetTestKernels)
        // and built side by side.  On a cold driver cache the compile is most
        // of a device's setup -- an Arc A380 spent ~45 s on the compute
        // families as one program -- and a compiler works through one program
        // at a time, so separate programs on separate threads put the CPU's
        // cores on it.  Separate programs are also what NVIDIA's OpenCL needs:
        // the local-bandwidth kernels' __local arguments and the image kernels'
        // image2d_t reserve module-level resources that compress the register
        // budget of every other kernel in the same program, triggering
        // CL_OUT_OF_RESOURCES on the v16 kernels.
        struct ProgramBuild
        {
          Benchmark which;
          const char *label;
          std::string options;
          bool quiet;          // a failure is the test's "did not build", not an error
          cl::Program program;
          std::string error;   // what failed, and the compiler's log; empty: built
        };
        std::vector<ProgramBuild> builds;
        auto want = [&](Benchmark which, const char *label, bool wanted,
                        const std::string &options, bool quiet = false)
        {
          if (wanted)
            builds.push_back({which, label, options, quiet, cl::Program(), std::string()});
        };
        // On a device that offers sub-group 16, the float families' affine
        // chain a second time pinned to it (kernels/mad_chain.cl).
        const std::string floatOptions =
            std::string(BUILD_OPTIONS) +
            (offersSubGroup16(devices[d]) ? " -DCLPEAK_ALT_SG16 " : "");
        // The int8 dot builtins hang off OpenCL C 3.0 feature macros, which a
        // compiler left at its default 1.2 never defines.  Its four-accumulator
        // cycle races pinned to sub-group 16 too, like the float families'
        // affine chain (kernels/compute_int8_dp_kernels.cl).
        const std::string int8Options =
            std::string(BUILD_OPTIONS) +
            (devInfo.int8DotProductPackedSupported ? " -DUSE_PACKED_DOT " : "") +
            (devInfo.openclC30 ? " -cl-std=CL3.0 " : "") +
            (offersSubGroup16(devices[d]) ? " -DCLPEAK_ALT_SG16 " : "");
        want(Benchmark::ComputeSP, "Single-precision compute",
             isAllowed(Benchmark::ComputeSP), floatOptions);
        want(Benchmark::ComputeHP, "Half-precision compute",
             devInfo.halfSupported && isAllowed(Benchmark::ComputeHP), floatOptions);
        want(Benchmark::ComputeDP, "Double-precision compute",
             devInfo.doubleSupported && isAllowed(Benchmark::ComputeDP), floatOptions);
        want(Benchmark::ComputeMP, "Mixed-precision compute",
             devInfo.halfSupported && isAllowed(Benchmark::ComputeMP), floatOptions);
        want(Benchmark::ComputeInt, "Integer compute",
             isAllowed(Benchmark::ComputeInt), BUILD_OPTIONS);
        want(Benchmark::ComputeIntFast, "Integer compute Fast 24bit",
             isAllowed(Benchmark::ComputeIntFast), BUILD_OPTIONS);
        want(Benchmark::ComputeChar, "Integer char (8bit) compute",
             isAllowed(Benchmark::ComputeChar), BUILD_OPTIONS);
        want(Benchmark::ComputeShort, "Integer short (16bit) compute",
             isAllowed(Benchmark::ComputeShort), BUILD_OPTIONS);
        want(Benchmark::ComputeInt8DP, "INT8 dot-product compute",
             (devInfo.int8DotProductSupported || devInfo.int8DotProductPackedSupported) &&
                 isAllowed(Benchmark::ComputeInt8DP),
             int8Options, true);
        // The latency test borrows a global-bandwidth kernel.
        want(Benchmark::GlobalBW, "Global memory bandwidth",
             isAllowed(Benchmark::GlobalBW) || isAllowed(Benchmark::KernelLatency),
             BUILD_OPTIONS);
        want(Benchmark::LocalBW, "Local memory bandwidth",
             isAllowed(Benchmark::LocalBW), BUILD_OPTIONS, true);
        want(Benchmark::ImageBW, "Image memory bandwidth",
             devInfo.imageSupported && isAllowed(Benchmark::ImageBW), BUILD_OPTIONS, true);

        {
          std::atomic<size_t> next(0);
          auto work = [&]()
          {
            for (size_t i = next++; i < builds.size(); i = next++)
            {
              ProgramBuild &b = builds[i];
              try
              {
                cl::Program::Sources source(1, clGetTestKernels(b.which));
                b.program = cl::Program(ctx, source);
                std::vector<cl::Device> dev = {devices[d]};
                b.program.build(dev, b.options.c_str());
              }
              catch (cl::Error &error)
              {
                b.error = std::string(error.what()) + " " + std::to_string(error.err());
                try
                {
                  b.error += ":\n" + b.program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[d]);
                }
                catch (cl::Error &)
                {
                }
                b.program = cl::Program();
              }
              catch (std::exception &e)
              {
                b.error = e.what();
                b.program = cl::Program();
              }
            }
          };
          // Up to eight compiles at once: past that the gain is small and a
          // compiler's working set is not.
          size_t threads = std::min<size_t>(builds.size(), 8);
          threads = std::min<size_t>(threads, std::max(1u, std::thread::hardware_concurrency()));
          std::vector<std::thread> pool;
          for (size_t t = 1; t < threads; t++)
            pool.emplace_back(work);
          work();
          for (std::thread &t : pool)
            t.join();
        }

        auto programFor = [&](Benchmark which) -> ProgramBuild *
        {
          for (ProgramBuild &b : builds)
            if (b.which == which)
              return &b;
          return nullptr;
        };
        for (const ProgramBuild &b : builds)
        {
          if (b.error.empty())
            continue;
          // A compute family or the bandwidth kernels failing is the compiler
          // refusing what the device claims to run, and only its log says why
          // -- so that is an error on the run log, not a --verbose extra, and
          // a file from a machine nobody can reach still explains the rows.
          if (b.quiet)
            CLPEAK_VLOG("  %s kernel build failed, test skipped: %s\n", b.label,
                        b.error.c_str());
          else
            CLPEAK_LOG(Error, "OpenCL: %s program build failed on %s (%s)", b.label,
                       devInfo.deviceName.c_str(), b.error.c_str());
        }
        cl::Program none;
        auto prog = [&](Benchmark which) -> cl::Program &
        {
          ProgramBuild *b = programFor(which);
          return b ? b->program : none;
        };
        auto buildError = [&](Benchmark which) -> std::string
        {
          ProgramBuild *b = programFor(which);
          return b && !b->quiet && !b->error.empty()
                     ? "the OpenCL compiler did not build this test's program -- "
                       "its log is on the run log"
                     : std::string();
        };

        cl_command_queue_properties supportedQueueProps = devices[d].getInfo<CL_DEVICE_QUEUE_PROPERTIES>();
        bool supportsProfilingQueue = (supportedQueueProps & CL_QUEUE_PROFILING_ENABLE) != 0;

        cl_command_queue_properties queueCreateProps = supportsProfilingQueue ? CL_QUEUE_PROFILING_ENABLE : 0;
        cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d], queueCreateProps);

        // ---- Compute (GFLOPS/TFLOPS + GOPS/TOPS) ---------------------------
        runComputeTest(queue, prog(Benchmark::ComputeSP),
                       buildError(Benchmark::ComputeSP),
                       devInfo, cfg, Benchmark::ComputeSP,
                       "Single-precision compute", "single_precision_compute",
                       "compute_sp", "float", "flops",
                       "Peak fp32 arithmetic rate of the device's ALUs, with no memory "
                       "traffic.",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_float));

        runComputeTest(queue, prog(Benchmark::ComputeHP),
                       buildError(Benchmark::ComputeHP),
                       devInfo, cfg, Benchmark::ComputeHP,
                       "Half-precision compute", "half_precision_compute",
                       "compute_hp", "half", "flops",
                       "Peak fp16 arithmetic rate, with fp16 inputs and accumulator.",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_half));

        runComputeTest(queue, prog(Benchmark::ComputeDP),
                       buildError(Benchmark::ComputeDP),
                       devInfo, cfg, Benchmark::ComputeDP,
                       "Double-precision compute", "double_precision_compute",
                       "compute_dp", "double", "flops",
                       "Peak fp64 arithmetic rate.",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeDPWgsPerCU, sizeof(cl_double));

        runComputeTest(queue, prog(Benchmark::ComputeMP),
                       buildError(Benchmark::ComputeMP),
                       devInfo, cfg, Benchmark::ComputeMP,
                       "Mixed-precision compute fp16xfp16+fp32", "mixed_precision_compute",
                       "compute_mp", "mp", "flops",
                       "Peak rate of fp16 multiplies accumulated in fp32, "
                       "without the matrix engine.",
                       COMPUTE_FP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_float));

        runComputeTest(queue, prog(Benchmark::ComputeInt),
                       buildError(Benchmark::ComputeInt),
                       devInfo, cfg, Benchmark::ComputeInt,
                       "Integer compute", "integer_compute",
                       "compute_integer", "int", "ops",
                       "Peak 32-bit integer arithmetic rate.",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        runComputeTest(queue, prog(Benchmark::ComputeIntFast),
                       buildError(Benchmark::ComputeIntFast),
                       devInfo, cfg, Benchmark::ComputeIntFast,
                       "Integer compute Fast 24bit", "integer_compute_fast",
                       "compute_intfast", "int", "ops",
                       "Peak rate of mad24, the integer multiply-add on 24-bit "
                       "operands.",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        runComputeTest(queue, prog(Benchmark::ComputeChar),
                       buildError(Benchmark::ComputeChar),
                       devInfo, cfg, Benchmark::ComputeChar,
                       "Integer char (8bit) compute", "integer_compute_char",
                       "compute_char", "char", "ops",
                       "Peak 8-bit integer arithmetic rate.",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_char));

        runComputeTest(queue, prog(Benchmark::ComputeShort),
                       buildError(Benchmark::ComputeShort),
                       devInfo, cfg, Benchmark::ComputeShort,
                       "Integer short (16bit) compute", "integer_compute_short",
                       "compute_short", "short", "ops",
                       "Peak 16-bit integer arithmetic rate.",
                       COMPUTE_INT_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_short));

        runComputeTest(queue, prog(Benchmark::ComputeInt8DP),
                       buildError(Benchmark::ComputeInt8DP),
                       devInfo, cfg, Benchmark::ComputeInt8DP,
                       "INT8 dot-product compute", "integer_compute_int8_dp",
                       "compute_int8_dp", "int8_dp", "ops",
                       "Peak rate of the 4-way int8 dot-product instruction, "
                       "without the matrix engine.",
                       COMPUTE_INT8_DP_WORK_PER_WI, cfg.computeWgsPerCU, sizeof(cl_int));

        // ---- Phase 3: bandwidth ----------------------------------------
        runGlobalBandwidthTest(queue, prog(Benchmark::GlobalBW), devInfo, cfg);
        runLocalBandwidthTest(queue, prog(Benchmark::LocalBW), devInfo, cfg);
        runImageBandwidthTest(queue, prog(Benchmark::ImageBW), devInfo, cfg);
        runTransferBandwidthTest(queue, none, devInfo, cfg);

        // ---- Phase 4: latency ------------------------------------------
        if (supportsProfilingQueue)
          runKernelLatency(queue, prog(Benchmark::GlobalBW), devInfo, cfg);
        else if (isAllowed(Benchmark::KernelLatency))
        {
          auto test = deviceScope.beginTest(
            {"kernel_launch_latency", "Kernel launch latency", "s",
             Category::Unknown,
             "Time to launch a small kernel, the fixed cost every dispatch pays."});
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
