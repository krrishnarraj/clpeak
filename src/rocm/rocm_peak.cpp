#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>
#include <common/inventory.h>
#include <common/options.h>
#include <chrono>
#include <cstdio>
#include <ostream>
#include <utility>

static const char *hipErrStr(hipError_t r)
{
  const char *s = hipGetErrorString(r);
  return s ? s : "unknown HIP error";
}

RocmPeak::RocmPeak()
    : deviceIndex(-1),
      initialised(false)
{
}

RocmPeak::~RocmPeak() {}

void RocmPeak::applyOptions(const CliOptions &opts)
{
  Peak::applyOptions(opts);
  deviceIndex = opts.rocmDeviceIndex;
}

bool RocmPeak::initRuntime()
{
  if (initialised)
    return true;

  int n = 0;
  hipError_t r = hipGetDeviceCount(&n);
  if (r != hipSuccess)
  {
    fprintf(stderr, "hipGetDeviceCount failed: %s\n", hipErrStr(r));
    return false;
  }
  for (int i = 0; i < n; i++)
    devIndices.push_back(i);
  initialised = true;
  return true;
}

float RocmPeak::runKernel(RocmDevice &dev, hipFunction_t fn,
                          uint32_t gridX, uint32_t blockX,
                          void **args,
                          unsigned int targetTimeUsLocal, unsigned int forcedIters)
{
  hipEvent_t start = nullptr, stop = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&stop) != hipSuccess)
    return -1.0f;

  auto runBatch = [&](unsigned int n) -> float {
    hipError_t status = hipSuccess;
    (void)hipEventRecord(start, dev.stream);
    for (unsigned int i = 0; i < n; i++)
    {
      status = hipModuleLaunchKernel(fn, gridX, 1, 1, blockX, 1, 1,
                                     0, dev.stream, args, nullptr);
      if (status != hipSuccess)
        break;
    }
    if (status != hipSuccess)
    {
      fprintf(stderr, "hipModuleLaunchKernel failed: %s\n", hipErrStr(status));
      return -1.0f;
    }
    (void)hipEventRecord(stop, dev.stream);
    status = hipEventSynchronize(stop);
    if (status != hipSuccess)
    {
      fprintf(stderr, "hipEventSynchronize failed: %s\n", hipErrStr(status));
      return -1.0f;
    }
    float ms = 0.0f;
    (void)hipEventElapsedTime(&ms, start, stop);
    return ms * 1000.0f;
  };

  for (unsigned int w = 0; w < warmupCount; w++)
  {
    hipError_t lr = hipModuleLaunchKernel(fn, gridX, 1, 1, blockX, 1, 1,
                                          0, dev.stream, args, nullptr);
    hipError_t sr = hipStreamSynchronize(dev.stream);
    if (lr != hipSuccess || sr != hipSuccess)
    {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(stop);
      return -1.0f;
    }
  }

  float probeUs = runBatch(1);
  if (probeUs <= 0.0f)
  {
    (void)hipEventDestroy(start);
    (void)hipEventDestroy(stop);
    return -1.0f;
  }

  unsigned int iters = pickIters((double)probeUs, targetTimeUsLocal, forcedIters);
  float totalUs = runBatch(iters);

  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);

  return totalUs > 0.0f ? totalUs / static_cast<float>(iters) : -1.0f;
}

int RocmPeak::runAll()
{
  if (!initRuntime())
  {
    log->note("ROCm: runtime init failed\n");
    return -1;
  }
  if (devIndices.empty())
  {
    log->note("ROCm: no devices found\n");
    return -1;
  }

  auto backendScope = log->beginBackend("ROCm");

  for (int idx : devIndices)
  {
    if (deviceIndex >= 0 && idx != deviceIndex)
      continue;

    RocmDevice dev;
    if (!dev.init(idx))
    {
      log->note("ROCm: failed to init device " + std::to_string(idx) + "\n");
      continue;
    }

    benchmark_config_t cfg = benchmark_config_t::forDevice(DeviceType::Gpu);
    cfg.targetTimeUs = targetTimeUs;
    if (forceIters)
      cfg.kernelLatencyIters = specifiedIters;

    auto deviceScope = backendScope.beginDevice({
      dev.info.deviceName,
      "",
      dev.info.driverVersion,
      {
        {"Arch",  dev.info.archName},
        {"HIP",   dev.info.runtimeVersion},
        {"CUs",   std::to_string(dev.info.numCUs)},
        {"Wave",  std::to_string(dev.info.warpSize)},
        {"VRAM",  std::to_string(dev.info.totalGlobalMem / (1024 * 1024)) + " MB"},
      },
      -1,
      idx
    });
    currentDeviceScope = &deviceScope;

    if (isAllowed(Benchmark::ComputeSP))
      runComputeSP(dev, cfg);
    if (isAllowed(Benchmark::ComputeHP))
      runComputeHP(dev, cfg);
    if (isAllowed(Benchmark::ComputeDP))
      runComputeDP(dev, cfg);
    if (isAllowed(Benchmark::ComputeMP))
      runComputeMP(dev, cfg);
    if (isAllowed(Benchmark::ComputeBF16))
      runComputeBF16(dev, cfg);
    if (isAllowedAs(Benchmark::Rocwmma, Category::FpCompute))
      runRocwmma(dev, cfg, Category::FpCompute);
    if (isAllowedAs(Benchmark::Mfma, Category::FpCompute))
      runMfma(dev, cfg, Category::FpCompute);
    if (isAllowedAs(Benchmark::Mfma, Category::FpCompute))
      runSparseMfma(dev, cfg, Category::FpCompute);
    if (isAllowedAs(Benchmark::Rocblas, Category::FpCompute))
      runRocblas(dev, cfg, Category::FpCompute);
    if (isAllowedAs(Benchmark::Rocblas, Category::FpCompute))
      runHipblasLt(dev, cfg);

    if (isAllowed(Benchmark::ComputeInt))
      runComputeInt32(dev, cfg);
    if (isAllowed(Benchmark::ComputeInt8DP))
      runComputeInt8DP(dev, cfg);

    if (isAllowedAs(Benchmark::Rocwmma, Category::IntCompute))
      runRocwmma(dev, cfg, Category::IntCompute);
    if (isAllowedAs(Benchmark::Mfma, Category::IntCompute))
      runMfma(dev, cfg, Category::IntCompute);
    if (isAllowedAs(Benchmark::Mfma, Category::IntCompute))
      runSparseMfma(dev, cfg, Category::IntCompute);
    if (isAllowedAs(Benchmark::Rocblas, Category::IntCompute))
      runRocblas(dev, cfg, Category::IntCompute);
    if (isAllowed(Benchmark::AtomicThroughput))
      runAtomicThroughput(dev, cfg);

    if (isAllowed(Benchmark::GlobalBW))
      runGlobalBandwidth(dev, cfg);
    if (isAllowed(Benchmark::LocalBW))
      runLocalBandwidth(dev, cfg);
    if (isAllowed(Benchmark::ImageBW))
      runImageBandwidth(dev, cfg);
    if (isAllowed(Benchmark::TransferBW))
      runTransferBandwidth(dev, cfg);

    if (isAllowed(Benchmark::KernelLatency))
      runKernelLatency(dev, cfg);

    currentDeviceScope = nullptr;
  }

  return 0;
}

BackendInventory RocmPeak::enumerate()
{
  BackendInventory inv;
  inv.backend = "ROCm";

  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n == 0)
    return inv;
  inv.available = true;

  InventoryPlatform plat;
  plat.index = 0;
  plat.name = "ROCm/HIP";

  for (int i = 0; i < n; i++)
  {
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, i) != hipSuccess)
      continue;

    InventoryDevice dev;
    dev.index = i;
    dev.name = props.name;
    dev.typeStr = "GPU";
    plat.devices.push_back(std::move(dev));
  }

  inv.platforms.push_back(std::move(plat));
  return inv;
}

void RocmPeak::printInventory(const BackendInventory &b, std::ostream &os)
{
  os << "\n=== ROCm backend ===\n";
  if (!b.available)
  {
    os << "ROCm: runtime init failed or no devices found\n";
    return;
  }
  for (const auto &plat : b.platforms)
    for (const auto &d : plat.devices)
    {
      os << "  ROCm Device " << d.index << ": " << d.name;
      if (!d.typeStr.empty())
        os << " [" << d.typeStr << "]";
      os << "\n";
    }
}

#endif // ENABLE_ROCM
