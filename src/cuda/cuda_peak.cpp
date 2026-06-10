#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/options.h>
#include <common/inventory.h>
#include <common/common.h>
#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Small helpers (also in cuda_device.cpp for CU_CHECK)
// ---------------------------------------------------------------------------

static const char *cuErrStr(CUresult r)
{
  const char *s = nullptr;
  cuGetErrorString(r, &s);
  return s ? s : "unknown CUDA error";
}

// ---------------------------------------------------------------------------
// CudaPeak
// ---------------------------------------------------------------------------

CudaPeak::CudaPeak()
    : initialised(false)
{
}

CudaPeak::~CudaPeak() {}

void CudaPeak::applyOptions(const CliOptions &opts)
{
    Peak::applyOptions(opts);
    deviceIndices = opts.cudaDeviceIndices;
}

bool CudaPeak::initDriver()
{
  if (initialised)
    return true;
  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS)
  {
    fprintf(stderr, "cuInit failed: %s\n", cuErrStr(r));
    return false;
  }
  int n = 0;
  r = cuDeviceGetCount(&n);
  if (r != CUDA_SUCCESS)
    return false;
  for (int i = 0; i < n; i++)
    devIndices.push_back(i);
  initialised = true;
  return true;
}

float CudaPeak::runKernel(CudaDevice &dev, CUfunction fn,
                          uint32_t gridX, uint32_t blockX,
                          void **args,
                          unsigned int targetTimeUsLocal, unsigned int forcedIters)
{
  CUevent start, stop;
  cuEventCreate(&start, CU_EVENT_DEFAULT);
  cuEventCreate(&stop, CU_EVENT_DEFAULT);

  // Time `n` launches batched as one stream submit; returns total us.
  auto runBatch = [&](unsigned int n) -> float {
    cuEventRecord(start, dev.stream);
    for (unsigned int i = 0; i < n; i++)
      cuLaunchKernel(fn, gridX, 1, 1, blockX, 1, 1, 0, dev.stream, args, nullptr);
    cuEventRecord(stop, dev.stream);
    cuEventSynchronize(stop);
    float ms = 0;
    cuEventElapsedTime(&ms, start, stop);
    return ms * 1000.0f;
  };

  // Phase 1: untimed warmup. Keep each warmup as its own completed launch so
  // slow kernels do not get batched before calibration.
  for (unsigned int w = 0; w < warmupCount; w++)
  {
    cuLaunchKernel(fn, gridX, 1, 1, blockX, 1, 1, 0, dev.stream, args, nullptr);
    cuStreamSynchronize(dev.stream);
  }

  // Phase 2: timed calibration probe. Keep this to one launch so warmupCount
  // does not force a multi-launch batch on slow kernels.
  unsigned int probeIters = 1;
  float probeUs = runBatch(probeIters);
  double per_iter_us = (double)probeUs / (double)probeIters;

  // Phase 3: real timed run with calibrated iter count.
  unsigned int iters = pickIters(per_iter_us, targetTimeUsLocal, forcedIters);
  float totalUs = runBatch(iters);

  cuEventDestroy(start);
  cuEventDestroy(stop);

  return totalUs / static_cast<float>(iters); // microseconds / iter
}

int CudaPeak::runAll()
{
  if (!initDriver())
  {
    log->note("CUDA: driver init failed\n");
    return -1;
  }
  if (devIndices.empty())
  {
    log->note("CUDA: no devices found\n");
    return 0;
  }

  auto backendScope = log->beginBackend("CUDA");

  for (int idx : devIndices)
  {
    if (!deviceIndices.empty() &&
        std::find(deviceIndices.begin(), deviceIndices.end(), idx) == deviceIndices.end())
      continue;

    CudaDevice dev;
    if (!dev.init(idx))
    {
      log->note("CUDA: failed to init device " + std::to_string(idx) + "\n");
      continue;
    }

    benchmark_config_t cfg = benchmark_config_t::forDevice(DeviceType::Gpu);
    cfg.targetTimeUs = targetTimeUs;
    if (forceIters)
      cfg.kernelLatencyIters = specifiedIters;

    auto deviceScope = backendScope.beginDevice({
      dev.info.deviceName,
      "",   // platform defaults to "CUDA"
      dev.info.driverVersion,
      {
        {"Arch",  dev.info.archName},
        {"NVRTC", dev.info.runtimeVersion},
        {"SMs",   std::to_string(dev.info.numSMs)},
        {"VRAM",  std::to_string(dev.info.totalGlobalMem / (1024 * 1024)) + " MB"},
      },
      -1,
      idx
    });
    currentDeviceScope = &deviceScope;

    // ---- Phase 1: floating-point compute (GFLOPS / TFLOPS) -------------
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
    if (isAllowedAs(Benchmark::Wmma, Category::FpCompute))
      runWmma(dev, cfg, Category::FpCompute);
    if (isAllowedAs(Benchmark::Cublas, Category::FpCompute))
      runCublas(dev, cfg, Category::FpCompute);

    // ---- Phase 2: integer compute (GOPS / TOPS) ------------------------
    if (isAllowed(Benchmark::ComputeInt))
      runComputeInt32(dev, cfg);
    if (isAllowed(Benchmark::ComputeInt8DP))
      runComputeInt8DP(dev, cfg);

    if (isAllowedAs(Benchmark::Wmma, Category::IntCompute))
      runWmma(dev, cfg, Category::IntCompute);
    if (isAllowedAs(Benchmark::Cublas, Category::IntCompute))
      runCublas(dev, cfg, Category::IntCompute);
    // ---- Phase 3: bandwidth (GBPS) -------------------------------------
    if (isAllowed(Benchmark::GlobalBW))
      runGlobalBandwidth(dev, cfg);
    if (isAllowed(Benchmark::LocalBW))
      runLocalBandwidth(dev, cfg);
    if (isAllowed(Benchmark::ImageBW))
      runImageBandwidth(dev, cfg);
    if (isAllowed(Benchmark::TransferBW))
      runTransferBandwidth(dev, cfg);

    // ---- Phase 4: latency (us) -----------------------------------------
    if (isAllowed(Benchmark::KernelLatency))
      runKernelLatency(dev, cfg);

    currentDeviceScope = nullptr;
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Benchmark methods live in separate files:
//   cuda_device.cpp       compute_kernel.cpp
//   compute_float.cpp     compute_int.cpp       wmma.cpp
//   cuda_blas.cpp         global_bandwidth.cpp  local_bandwidth.cpp
//   image_bandwidth.cpp   transfer_bandwidth.cpp
//   kernel_latency.cpp
// ---------------------------------------------------------------------------

// Free-function enumeration used by --list-devices and the Android JNI surface.
// Uses the static driver API directly — no CudaPeak instance required.
BackendInventory CudaPeak::enumerate()
{
  BackendInventory inv;
  inv.backend = "CUDA";

  if (cuInit(0) != CUDA_SUCCESS)
    return inv;
  int n = 0;
  if (cuDeviceGetCount(&n) != CUDA_SUCCESS || n == 0)
    return inv;
  inv.available = true;

  InventoryPlatform plat;
  plat.index = 0;
  plat.name = "CUDA";

  for (int i = 0; i < n; i++)
  {
    CUdevice d;
    if (cuDeviceGet(&d, i) != CUDA_SUCCESS)
      continue;
    char name[256] = {0};
    cuDeviceGetName(name, sizeof(name), d);
    int maj = 0, min = 0;
    cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d);
    cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d);

    InventoryDevice dev;
    dev.index = i;
    dev.name = name;
    dev.typeStr = "sm_" + std::to_string(maj) + std::to_string(min);
    plat.devices.push_back(std::move(dev));
  }

  inv.platforms.push_back(std::move(plat));
  return inv;
}

void CudaPeak::printInventory(const BackendInventory &b, std::ostream &os)
{
    os << "\n=== CUDA backend ===\n";
    if (!b.available)
    {
        os << "CUDA: driver init failed or no devices found\n";
        return;
    }
    for (const auto &plat : b.platforms)
        for (const auto &d : plat.devices)
        {
            os << "  CUDA Device " << d.index << ": " << d.name;
            if (!d.typeStr.empty())
                os << " [" << d.typeStr << "]";
            os << "\n";
        }
}

#endif // ENABLE_CUDA
