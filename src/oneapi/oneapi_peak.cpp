#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <common/inventory.h>
#include <common/options.h>
#include <chrono>
#include <cstdio>
#include <ostream>
#include <utility>

OneapiPeak::OneapiPeak()
    : deviceIndex(-1),
      initialised(false)
{
}

OneapiPeak::~OneapiPeak() {}

void OneapiPeak::applyOptions(const CliOptions &opts)
{
  Peak::applyOptions(opts);
  deviceIndex = opts.oneapiDeviceIndex;
}

// Pull every SYCL-visible GPU device.  Filter CPU/accelerator out — clpeak
// is a GPU benchmark tool, and on a typical install the SYCL CPU device
// would otherwise show up alongside the iGPU and confuse the matrix.
static std::vector<sycl::device> enumerateGpus()
{
  std::vector<sycl::device> out;
  try
  {
    for (const auto &p : sycl::platform::get_platforms())
    {
      for (const auto &d : p.get_devices(sycl::info::device_type::gpu))
        out.push_back(d);
    }
  }
  catch (const sycl::exception &e)
  {
    fprintf(stderr, "sycl::platform::get_platforms failed: %s\n", e.what());
  }
  return out;
}

bool OneapiPeak::initRuntime()
{
  if (initialised)
    return true;
  devices = enumerateGpus();
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
    catch (const sycl::exception &e)
    {
      fprintf(stderr, "SYCL submit failed: %s\n", e.what());
      return -1.0f;
    }
  };

  try
  {
    for (unsigned int w = 0; w < warmupCount; w++)
      submit(dev.stream);
    dev.stream.wait_and_throw();
  }
  catch (const sycl::exception &e)
  {
    fprintf(stderr, "SYCL warmup failed: %s\n", e.what());
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
    log->note("oneAPI: no SYCL GPU devices found\n");
    return -1;
  }

  auto backendScope = log->beginBackend("oneAPI");

  for (int idx = 0; idx < (int)devices.size(); idx++)
  {
    if (deviceIndex >= 0 && idx != deviceIndex)
      continue;

    OneapiDevice dev;
    if (!dev.init(idx, devices[idx]))
    {
      log->note("oneAPI: failed to init device " + std::to_string(idx) + "\n");
      continue;
    }

    benchmark_config_t cfg = benchmark_config_t::forDevice(dev.info.deviceType);
    cfg.targetTimeUs = targetTimeUs;
    if (forceIters)
      cfg.kernelLatencyIters = specifiedIters;

    auto deviceScope = backendScope.beginDevice({
      dev.info.deviceName,
      "",
      dev.info.driverVersion,
      {
        {"Vendor",  dev.info.vendor},
        {"Backend", dev.info.backendName},
        {"CUs",     std::to_string(dev.info.numCUs)},
        {"SG",      std::to_string(dev.info.preferredSubGroupSize)},
        {"VRAM",    std::to_string(dev.info.totalGlobalMem / (1024 * 1024)) + " MB"},
      },
      -1,
      idx
    });
    currentDeviceScope = &deviceScope;

    if (isAllowed(Benchmark::ComputeSP))     runComputeSP(dev, cfg);
    if (isAllowed(Benchmark::ComputeHP))     runComputeHP(dev, cfg);
    if (isAllowed(Benchmark::ComputeDP))     runComputeDP(dev, cfg);
    if (isAllowed(Benchmark::ComputeMP))     runComputeMP(dev, cfg);
    if (isAllowed(Benchmark::ComputeBF16))   runComputeBF16(dev, cfg);
    if (isAllowedAs(Benchmark::JointMatrix, Category::FpCompute))
      runJointMatrix(dev, cfg, Category::FpCompute);
    if (isAllowedAs(Benchmark::Onemkl, Category::FpCompute))
      runOnemkl(dev, cfg);

    if (isAllowed(Benchmark::ComputeInt))         runComputeInt32(dev, cfg);
    if (isAllowed(Benchmark::ComputeInt8DP))      runComputeInt8DP(dev, cfg);
    if (isAllowed(Benchmark::ComputeInt4Packed))  runComputeInt4Packed(dev, cfg);
    if (isAllowedAs(Benchmark::JointMatrix, Category::IntCompute))
      runJointMatrix(dev, cfg, Category::IntCompute);
    if (isAllowed(Benchmark::AtomicThroughput))   runAtomicThroughput(dev, cfg);

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
  inv.backend = "oneAPI";

  auto devs = enumerateGpus();
  if (devs.empty())
    return inv;
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
    d.typeStr = "GPU";
    plat.devices.push_back(std::move(d));
  }

  inv.platforms.push_back(std::move(plat));
  return inv;
}

void OneapiPeak::printInventory(const BackendInventory &b, std::ostream &os)
{
  os << "\n=== oneAPI backend ===\n";
  if (!b.available)
  {
    os << "oneAPI: no SYCL GPU devices found\n";
    return;
  }
  for (const auto &plat : b.platforms)
    for (const auto &d : plat.devices)
    {
      os << "  oneAPI Device " << d.index << ": " << d.name;
      if (!d.typeStr.empty())
        os << " [" << d.typeStr << "]";
      os << "\n";
    }
}

#endif // ENABLE_ONEAPI
