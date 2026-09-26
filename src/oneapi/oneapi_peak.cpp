#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <common/dynlib.h>
#include <common/inventory.h>
#include <common/options.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ostream>
#include <utility>

OneapiPeak::OneapiPeak()
    : initialised(false)
{
}

OneapiPeak::~OneapiPeak() {}

// The OpenCL ICD loader's platforms.  Resolved at run time, not linked: the
// loader is already in the process wherever SYCL's OpenCL adapter is, and the
// oneAPI backend then builds without an OpenCL SDK.
static std::vector<cl_platform_id> openclPlatformIds()
{
  static const auto getPlatformIds = [] {
    void *icd = clpeak::dynOpen({
#if defined(_WIN32)
        "OpenCL.dll",
#else
        "libOpenCL.so.1", "libOpenCL.so",
#endif
    });
    return reinterpret_cast<decltype(&::clGetPlatformIDs)>(
        clpeak::dynSym(icd, "clGetPlatformIDs"));
  }();

  cl_uint count = 0;
  if (!getPlatformIds || getPlatformIds(0, nullptr, &count) != CL_SUCCESS)
    return {};
  std::vector<cl_platform_id> ids(count);
  if (count && getPlatformIds(count, ids.data(), nullptr) != CL_SUCCESS)
    ids.clear();
  return ids;
}

// Every platform SYCL can drive.  get_platforms() is all or nothing: a backend
// whose driver fails to come up throws out of the whole call and takes every
// other backend's platforms with it.  Intel's legacy Windows driver for a UHD
// 630 (31.0.101.2121, .2141) ships a Level Zero driver that does exactly that,
// with UR_RESULT_ERROR_UNINITIALIZED -- sycl-ls itself dies on it -- while its
// OpenCL driver runs SYCL fine, the path Linux takes on the same iGPU.  So on
// a throw, each OpenCL platform goes to SYCL through interop instead:
// make_platform reaches the OpenCL adapter alone and never the one that
// failed.  The adapter rejects the platforms SYCL does not drive (NVIDIA's,
// AMD's), which is not worth more than a --verbose line.
static std::vector<sycl::platform> syclPlatforms()
{
  try
  {
    return sycl::platform::get_platforms();
  }
  catch (const sycl::exception &e)
  {
    CLPEAK_LOG(Warning, "oneAPI: sycl::platform::get_platforms failed: %s; "
               "listing the OpenCL platforms instead", e.what());
  }

  std::vector<sycl::platform> out;
  for (cl_platform_id id : openclPlatformIds())
  {
    try
    {
      out.push_back(sycl::make_platform<sycl::backend::opencl>(id));
    }
    catch (const sycl::exception &e)
    {
      CLPEAK_VLOG("oneAPI: an OpenCL platform SYCL does not drive was skipped: %s\n",
                  e.what());
    }
  }
  return out;
}

// Pick the SYCL devices to benchmark.  clpeak is primarily a GPU tool, so we
// prefer GPUs and (on a typical machine) leave the SYCL CPU device out so it
// doesn't clutter the matrix alongside the iGPU.  But SYCL kernels also run
// fine on the CPU/accelerator runtime, so if there is NO GPU visible (e.g.
// the Level Zero / Intel compute runtime isn't installed) we fall back to
// the CPU and any accelerator rather than reporting nothing.
static std::vector<sycl::device> enumerateDevices()
{
  const std::vector<sycl::platform> platforms = syclPlatforms();
  auto collect = [&](sycl::info::device_type type, std::vector<sycl::device> &out) {
    for (const auto &p : platforms)
    {
      try
      {
        for (const auto &d : p.get_devices(type))
          out.push_back(d);
      }
      catch (const sycl::exception &e)
      {
        CLPEAK_LOG(Error, "oneAPI: sycl::platform::get_devices failed: %s", e.what());
      }
    }
  };

  std::vector<sycl::device> out;
  collect(sycl::info::device_type::gpu, out);
  if (out.empty())
  {
    collect(sycl::info::device_type::cpu, out);
    collect(sycl::info::device_type::accelerator, out);
  }
  return out;
}

static const char *deviceTypeStr(const sycl::device &d)
{
  if (d.is_gpu())         return "GPU";
  if (d.is_cpu())         return "CPU";
  if (d.is_accelerator()) return "Accelerator";
  return "Other";
}

bool OneapiPeak::initRuntime()
{
  if (initialised)
    return true;
  devices = enumerateDevices();
  initialised = true;
  return true;
}

float OneapiPeak::runKernel(OneapiDevice &dev,
                            const KernelSubmitter &submit,
                            unsigned int targetTimeUsLocal,
                            unsigned int forcedIters)
{
  auto runBatch = [&](unsigned int n) -> float {
    try
    {
      auto t0 = std::chrono::high_resolution_clock::now();
      sycl::event last;
      for (unsigned int i = 0; i < n; i++)
        last = submit(dev.stream);
      // in_order queue: waiting on the queue is equivalent to waiting on
      // every prior submission and is the recommended sync point.
      dev.stream.wait_and_throw();
      auto t1 = std::chrono::high_resolution_clock::now();
      auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
      return (float)((double)ns / 1000.0);
    }
    catch (const std::exception &e)
    {
      CLPEAK_VLOG("SYCL submit failed: %s\n", e.what());
      // A failed submission can leave the in-order queue in a permanent error
      // state; recreate it so the next benchmark isn't a false-failure cascade.
      dev.resetQueue();
      return -1.0f;
    }
  };

  try
  {
    for (unsigned int w = 0; w < warmupCount; w++)
      submit(dev.stream);
    dev.stream.wait_and_throw();
  }
  catch (const std::exception &e)
  {
    CLPEAK_VLOG("SYCL warmup failed: %s\n", e.what());
    dev.resetQueue();
    return -1.0f;
  }

  float probeUs = runBatch(1);
  if (probeUs <= 0.0f)
    return -1.0f;

  unsigned int iters = pickIters((double)probeUs, targetTimeUsLocal, forcedIters);
  float totalUs = runBatch(iters);
  return totalUs > 0.0f ? totalUs / static_cast<float>(iters) : -1.0f;
}

int OneapiPeak::runAll()
{
  if (!initRuntime())
  {
    log->note("oneAPI: runtime init failed\n");
    return -1;
  }
  if (devices.empty())
  {
    log->note("oneAPI: no SYCL devices found (no GPU, CPU, or accelerator "
              "visible to the SYCL runtime)\n");
    return 0;
  }

  auto backendScope = log->beginBackend("oneAPI");

  for (int idx = 0; idx < (int)devices.size(); idx++)
  {
    if (clpeak::cancelRequested())
      break;
    if (!isDeviceSelected(idx))
      continue;

    OneapiDevice dev;
    if (!dev.init(idx, devices[idx]))
    {
      log->note("oneAPI: failed to init device " + std::to_string(idx) + "\n");
      continue;
    }

    benchmark_config_t cfg = benchmark_config_t::forDevice(dev.info.deviceType, dev.info.globalMemCacheSize);
    cfg.targetTimeUs = targetTimeUs;
    if (forceIters)
      cfg.kernelLatencyIters = specifiedIters;

    auto deviceScope = backendScope.beginDevice({
      dev.info.deviceName,
      "",
      dev.info.driverVersion,
      {
        {"Vendor",  dev.info.vendor},
        {"Type",    deviceTypeStr(devices[idx])},
        {"Backend", dev.info.backendName},
        {"CUs",     std::to_string(dev.info.numCUs)},
        {"SG",      std::to_string(dev.info.preferredSubGroupSize)},
        {"VRAM",    std::to_string(dev.info.totalGlobalMem / (1024 * 1024)) + " MB"},
      },
      -1,
      idx,
      dev.info.deviceType
    });
    currentDeviceScope = &deviceScope;

    // ---- Compute (GFLOPS/TFLOPS + GOPS/TOPS) ---------------------------
    if (isAllowed(Benchmark::ComputeSP))     runComputeSP(dev, cfg);
    if (isAllowed(Benchmark::ComputeHP))     runComputeHP(dev, cfg);
    if (isAllowed(Benchmark::ComputeDP))     runComputeDP(dev, cfg);
    if (isAllowed(Benchmark::ComputeMP))     runComputeMP(dev, cfg);
    if (isAllowed(Benchmark::ComputeBF16))   runComputeBF16(dev, cfg);
    if (isAllowed(Benchmark::ComputeInt))         runComputeInt32(dev, cfg);
    if (isAllowed(Benchmark::MatrixCompute)) runJointMatrix(dev, cfg);
    if (isAllowed(Benchmark::Gemm))          runOnemkl(dev, cfg);


    if (isAllowed(Benchmark::GlobalBW))     runGlobalBandwidth(dev, cfg);
    if (isAllowed(Benchmark::LocalBW))      runLocalBandwidth(dev, cfg);
    if (isAllowed(Benchmark::ImageBW))      runImageBandwidth(dev, cfg);
    if (isAllowed(Benchmark::TransferBW))   runTransferBandwidth(dev, cfg);

    if (isAllowed(Benchmark::KernelLatency)) runKernelLatency(dev, cfg);

    currentDeviceScope = nullptr;
  }

  return 0;
}

BackendInventory OneapiPeak::enumerate()
{
  BackendInventory inv;
  inv.id = kBackend;

  auto devs = enumerateDevices();
  if (devs.empty())
  {
    inv.unavailableReason = "no SYCL devices found";
    return inv;
  }
  inv.available = true;

  InventoryPlatform plat;
  plat.index = 0;
  plat.name = "oneAPI/SYCL";

  for (int i = 0; i < (int)devs.size(); i++)
  {
    InventoryDevice d;
    d.index = i;
    try { d.name = devs[i].get_info<sycl::info::device::name>(); }
    catch (...) { d.name = "<unknown>"; }
    d.typeStr = deviceTypeStr(devs[i]);
    plat.devices.push_back(std::move(d));
  }

  inv.platforms.push_back(std::move(plat));
  return inv;
}

#endif // ENABLE_ONEAPI
