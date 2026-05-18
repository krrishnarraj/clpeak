#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/options.h>
#include <common/inventory.h>
#include <common/common.h>
#include <nvrtc.h>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <vector>
#include <string>

#ifndef CLPEAK_CUDA_INCLUDE_DIR
#define CLPEAK_CUDA_INCLUDE_DIR "/usr/local/cuda/include"
#endif

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

static const char *cuErrStr(CUresult r)
{
  const char *s = nullptr;
  cuGetErrorString(r, &s);
  return s ? s : "unknown CUDA error";
}

#define CU_CHECK(call)                                                                \
  do                                                                                  \
  {                                                                                   \
    CUresult _r = (call);                                                             \
    if (_r != CUDA_SUCCESS)                                                           \
    {                                                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cuErrStr(_r)); \
      return false;                                                                   \
    }                                                                                 \
  } while (0)

// ---------------------------------------------------------------------------
// CudaDevice
// ---------------------------------------------------------------------------

CudaDevice::CudaDevice() : device(0), context(nullptr), stream(nullptr) {}

CudaDevice::~CudaDevice() { cleanup(); }

bool CudaDevice::init(int devIndex)
{
  CU_CHECK(cuDeviceGet(&device, devIndex));

  char nameBuf[256] = {0};
  CU_CHECK(cuDeviceGetName(nameBuf, sizeof(nameBuf), device));
  info.deviceName = nameBuf;

  CU_CHECK(cuDeviceGetAttribute(&info.major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CU_CHECK(cuDeviceGetAttribute(&info.minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  CU_CHECK(cuDeviceGetAttribute(&info.numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
  CU_CHECK(cuDeviceGetAttribute(&info.maxThreadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
  CU_CHECK(cuDeviceGetAttribute(&info.clockRateKHz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device));

  size_t mem = 0;
  CU_CHECK(cuDeviceTotalMem(&mem, device));
  info.totalGlobalMem = mem;

  int driverVer = 0;
  cuDriverGetVersion(&driverVer);
  {
    std::stringstream ss;
    ss << (driverVer / 1000) << "." << (driverVer % 100) / 10;
    info.driverVersion = ss.str();
  }
  {
    int major = 0, minor = 0;
    nvrtcVersion(&major, &minor);
    std::stringstream ss;
    ss << major << "." << minor;
    info.runtimeVersion = ss.str();
  }
  {
    std::stringstream ss;
    ss << "sm_" << info.major << info.minor;
    info.archName = ss.str();
  }

  // Capability bits.  Compute-capability cutoffs follow CUDA's documented
  // first-architecture-supporting-X numbers.
  info.fp16Supported = (info.major > 5) || (info.major == 5 && info.minor >= 3);
  info.bf16Supported = (info.major >= 8);
  info.dp4aSupported = (info.major > 6) || (info.major == 6 && info.minor >= 1);
  info.wmmaSupported = (info.major >= 7);
  info.wmmaInt8Supported = (info.major > 7) || (info.major == 7 && info.minor >= 2);
  info.fp8MmaSupported = (info.major >= 9) || (info.major == 8 && info.minor >= 9);
  info.fp4MmaSupported = (info.major >= 12);
  info.tf32GemmSupported = (info.major >= 8);
  info.int8GemmSupported = (info.major > 7) || (info.major == 7 && info.minor >= 5);
  info.int4GemmSupported = (info.major >= 9);
  info.dpTensorSupported = (info.major >= 8);
  // s4 mma.sync was added on Turing (sm_75), kept on Ampere/Ada, removed on
  // Hopper+ (sm_90+).  Allow 7.5..8.9 inclusive.
  {
    int cc = info.major * 10 + info.minor;
    info.int4MmaSupported = (cc >= 75) && (cc <= 89);
  }
  info.bmmaSupported = (info.major > 7) || (info.major == 7 && info.minor >= 5);
  info.int8MmaSparseSupported = (info.major >= 8);

  info.deviceType = DeviceType::Gpu;   // CUDA devices are always GPUs

  // CUDA 13 promoted cuCtxCreate to a 4-arg signature taking a
  // CUctxCreateParams*; CUDA 12 and earlier expose only the 3-arg form.
  // Branch on CUDA_VERSION so the same source builds against both toolkits.
#if CUDA_VERSION >= 13000
  CU_CHECK(cuCtxCreate(&context, nullptr, 0, device));
#else
  CU_CHECK(cuCtxCreate(&context, 0, device));
#endif
  CU_CHECK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));

  return true;
}

void CudaDevice::cleanup()
{
  for (auto &kv : moduleCache)
    cuModuleUnload(kv.second);
  moduleCache.clear();

  if (stream)
  {
    cuStreamDestroy(stream);
    stream = nullptr;
  }
  if (context)
  {
    cuCtxDestroy(context);
    context = nullptr;
  }
}

bool CudaDevice::getKernel(const char *src, const char *srcName,
                           const char *kernelName, CUfunction &fn,
                           const std::vector<const char *> &extraOpts)
{
  // Cache by source-pointer identity: every embedded .cu blob is a distinct
  // extern in cuda_kernels_generated.cpp, so pointer equality is sufficient.
  auto it = moduleCache.find(src);
  CUmodule mod = nullptr;
  if (it != moduleCache.end())
  {
    mod = it->second;
  }
  else
  {
    nvrtcProgram prog;
    nvrtcResult nr = nvrtcCreateProgram(&prog, src, srcName, 0, nullptr, nullptr);
    if (nr != NVRTC_SUCCESS)
    {
      fprintf(stderr, "nvrtcCreateProgram failed: %s\n", nvrtcGetErrorString(nr));
      return false;
    }

    // Build the option list.  --gpu-architecture=sm_<cc> targets cubin
    // directly; the driver still JITs from PTX if needed but we get the
    // straightest path on the matching arch.  -I<CUDA_INCLUDE_DIR> resolves
    // <mma.h>, <cuda_fp16.h>, <cuda_bf16.h>, <cuda_fp8.h> at runtime.
    std::stringstream archOpt;
    archOpt << "--gpu-architecture=sm_" << info.major << info.minor;
    std::string archStr = archOpt.str();

    std::string incOpt = std::string("-I") + CLPEAK_CUDA_INCLUDE_DIR;

    std::vector<const char *> opts;
    bool hasArchOverride = false;
    for (auto *e : extraOpts)
    {
      if (e && (std::strncmp(e, "--gpu-architecture=", 19) == 0 ||
                std::strncmp(e, "-arch=", 6) == 0))
      {
        hasArchOverride = true;
        break;
      }
    }
    if (!hasArchOverride)
      opts.push_back(archStr.c_str());
    opts.push_back(incOpt.c_str());
    for (auto *e : extraOpts)
      opts.push_back(e);

    nr = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    if (nr != NVRTC_SUCCESS)
    {
      size_t logSize = 0;
      nvrtcGetProgramLogSize(prog, &logSize);
      std::string log(logSize, '\0');
      if (logSize > 1)
        nvrtcGetProgramLog(prog, &log[0]);
      fprintf(stderr, "NVRTC compile of %s failed:\n%s\n", srcName, log.c_str());
      nvrtcDestroyProgram(&prog);
      return false;
    }

    // Prefer cubin for the matching architecture; fall back to PTX if
    // cubin retrieval is unsupported in this NVRTC build.
    size_t cubinSize = 0;
    nr = nvrtcGetCUBINSize(prog, &cubinSize);
    if (nr == NVRTC_SUCCESS && cubinSize > 0)
    {
      std::vector<char> cubin(cubinSize);
      nvrtcGetCUBIN(prog, cubin.data());
      CUresult lr = cuModuleLoadData(&mod, cubin.data());
      if (lr != CUDA_SUCCESS)
      {
        fprintf(stderr, "cuModuleLoadData(cubin %s) failed: %s\n", srcName, cuErrStr(lr));
        nvrtcDestroyProgram(&prog);
        return false;
      }
    }
    else
    {
      size_t ptxSize = 0;
      nvrtcGetPTXSize(prog, &ptxSize);
      std::vector<char> ptx(ptxSize);
      nvrtcGetPTX(prog, ptx.data());
      CUresult lr = cuModuleLoadData(&mod, ptx.data());
      if (lr != CUDA_SUCCESS)
      {
        fprintf(stderr, "cuModuleLoadData(ptx %s) failed: %s\n", srcName, cuErrStr(lr));
        nvrtcDestroyProgram(&prog);
        return false;
      }
    }
    nvrtcDestroyProgram(&prog);
    moduleCache[src] = mod;
  }

  CUresult r = cuModuleGetFunction(&fn, mod, kernelName);
  if (r != CUDA_SUCCESS)
  {
    fprintf(stderr, "cuModuleGetFunction(%s in %s) failed: %s\n",
            kernelName, srcName, cuErrStr(r));
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// CudaPeak
// ---------------------------------------------------------------------------

CudaPeak::CudaPeak()
    : deviceIndex(-1),
      initialised(false)
{
}

CudaPeak::~CudaPeak() {}

void CudaPeak::applyOptions(const CliOptions &opts)
{
    Peak::applyOptions(opts);
    deviceIndex = opts.cudaDeviceIndex;
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
    return -1;
  }

  auto backendScope = log->beginBackend("CUDA");

  for (int idx : devIndices)
  {
    if (deviceIndex >= 0 && idx != deviceIndex)
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
    if (isAllowed(Benchmark::ComputeInt4Packed))
      runComputeInt4Packed(dev, cfg);
    if (isAllowedAs(Benchmark::Wmma, Category::IntCompute))
      runWmma(dev, cfg, Category::IntCompute);
    if (isAllowedAs(Benchmark::Cublas, Category::IntCompute))
      runCublas(dev, cfg, Category::IntCompute);
    if (isAllowed(Benchmark::AtomicThroughput))
      runAtomicThroughput(dev, cfg);

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
// Shared compute-peak driver.  Mirrors vkPeak::runComputeKernel in spirit:
// allocate a single device-local output buffer, dispatch each variant of
// the same kernel against it with NVRTC-compiled kernels.
// ---------------------------------------------------------------------------

int CudaPeak::runComputeKernel(CudaDevice &dev, benchmark_config_t &cfg,
                               const cuda_compute_desc_t &d)
{
  auto test = currentDeviceScope->beginTest({d.resultTag, d.title, d.unit});

  if (d.skip)
  {
    if (d.variants && d.numVariants > 0)
    {
      for (uint32_t i = 0; i < d.numVariants; i++)
        test.skip(d.variants[i].label, ResultStatus::Unsupported,
                  d.skipMsg ? d.skipMsg : "Skipped");
    }
    else
    {
      test.skip(d.metricLabel, ResultStatus::Unsupported,
                d.skipMsg ? d.skipMsg : "Skipped");
    }
    return 0;
  }

  struct Variant
  {
    const char *label;
    const char *kernelName;
    const char *src;
    const char *srcName;
  };
  std::vector<Variant> variants;
  if (d.variants && d.numVariants > 0)
    for (uint32_t i = 0; i < d.numVariants; i++)
      variants.push_back({d.variants[i].label, d.variants[i].kernelName,
                          d.variants[i].src, d.variants[i].srcName});
  else
    variants.push_back({d.metricLabel, d.kernelName, d.src, d.srcName});

  // Scale to numSMs so high-SM parts (H100, B200, …) don't get under-saturated;
  // floor at 32M preserves behavior on small dev cards.  Clamp by VRAM below.
  const uint32_t blockSize = d.blockSize ? d.blockSize : 256;
  const uint32_t outPerBlock = d.outElemsPerBlock ? d.outElemsPerBlock : blockSize;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numSMs);
  uint64_t bytesPerBlock = (uint64_t)outPerBlock * d.elemSize;
  uint64_t maxBlocks = dev.info.totalGlobalMem / 4 / bytesPerBlock; // cap at 1/4 VRAM
  uint64_t wantBlocks = globalThreads / blockSize;
  uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
  uint32_t numBlocks = (uint32_t)pickBlocks;
  uint64_t bufferBytes = (uint64_t)numBlocks * bytesPerBlock;

  CUdeviceptr outputBuf = 0;
  if (cuMemAlloc(&outputBuf, bufferBytes) != CUDA_SUCCESS)
  {
    for (const auto &v : variants)
      test.skip(v.label, ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  std::vector<const char *> nvrtcOpts;
  for (uint32_t i = 0; i < d.numExtraNvrtcOpts; i++)
    nvrtcOpts.push_back(d.extraNvrtcOpts[i]);

  for (const auto &v : variants)
  {
    CUfunction fn;
    if (!dev.getKernel(v.src, v.srcName, v.kernelName, fn, nvrtcOpts))
    {
      test.skip(v.label, ResultStatus::Error, "compile/load failed");
      continue;
    }

    void *args[2];
    args[0] = &outputBuf;
    args[1] = const_cast<void *>(d.scalarArg);

    float us = runKernel(dev, fn, numBlocks, blockSize, args,
                         cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    uint64_t totalThreads = (uint64_t)numBlocks * blockSize;
    double divider = d.unitDivider > 0.0 ? d.unitDivider : 1e9;
    float value = (float)((double)totalThreads * (double)d.workPerWI * 1e6 / us / divider);

    test.emit(v.label, value);
  }

  cuMemFree(outputBuf);
  return 0;
}

// ---------------------------------------------------------------------------
// Benchmark methods live in separate category files:
//   compute_float.cpp    compute_int.cpp    wmma.cpp
//   global_bandwidth.cpp local_bandwidth.cpp image_bandwidth.cpp
//   transfer_bandwidth.cpp kernel_latency.cpp atomic_throughput.cpp
//   cuda_blas.cpp (cuBLAS GEMM)
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
