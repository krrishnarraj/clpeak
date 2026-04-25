#ifdef ENABLE_CUDA

#include <cuda_peak.h>
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

#define CU_CHECK(call) do { CUresult _r = (call); \
  if (_r != CUDA_SUCCESS) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cuErrStr(_r)); \
    return false; } } while (0)

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
    std::stringstream ss; ss << major << "." << minor;
    info.runtimeVersion = ss.str();
  }
  {
    std::stringstream ss; ss << "sm_" << info.major << info.minor;
    info.archName = ss.str();
  }

  // Capability bits.  Compute-capability cutoffs follow CUDA's documented
  // first-architecture-supporting-X numbers.
  info.fp16Supported    = (info.major > 5) || (info.major == 5 && info.minor >= 3);
  info.bf16Supported    = (info.major >= 8);
  info.dp4aSupported    = (info.major > 6) || (info.major == 6 && info.minor >= 1);
  info.wmmaSupported     = (info.major >= 7);
  info.wmmaInt8Supported = (info.major >  7) || (info.major == 7 && info.minor >= 2);
  info.fp8MmaSupported   = (info.major >= 9) || (info.major == 8 && info.minor >= 9);

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

  if (stream)  { cuStreamDestroy(stream);  stream = nullptr; }
  if (context) { cuCtxDestroy(context);    context = nullptr; }
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
    opts.push_back(archStr.c_str());
    opts.push_back(incOpt.c_str());
    for (auto *e : extraOpts) opts.push_back(e);

    nr = nvrtcCompileProgram(prog, (int)opts.size(), opts.data());
    if (nr != NVRTC_SUCCESS)
    {
      size_t logSize = 0;
      nvrtcGetProgramLogSize(prog, &logSize);
      std::string log(logSize, '\0');
      if (logSize > 1) nvrtcGetProgramLog(prog, &log[0]);
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
  : warmupCount(2), specifiedIters(0), forceIters(false), listDevices(false),
    initialised(false) {}

CudaPeak::~CudaPeak() {}

bool CudaPeak::initDriver()
{
  if (initialised) return true;
  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS)
  {
    fprintf(stderr, "cuInit failed: %s\n", cuErrStr(r));
    return false;
  }
  int n = 0;
  r = cuDeviceGetCount(&n);
  if (r != CUDA_SUCCESS) return false;
  for (int i = 0; i < n; i++) devIndices.push_back(i);
  initialised = true;
  return true;
}

float CudaPeak::runKernel(CudaDevice &dev, CUfunction fn,
                          uint32_t gridX, uint32_t blockX,
                          void **args, unsigned int iters)
{
  CUevent start, stop;
  cuEventCreate(&start, CU_EVENT_DEFAULT);
  cuEventCreate(&stop,  CU_EVENT_DEFAULT);

  for (unsigned int w = 0; w < warmupCount; w++)
  {
    cuLaunchKernel(fn, gridX, 1, 1, blockX, 1, 1, 0, dev.stream, args, nullptr);
  }
  cuStreamSynchronize(dev.stream);

  // Time iters launches as one batch.  Per-iter event pairs would multiply
  // the host-side event overhead by `iters`; one pair around the full batch
  // gives a cleaner mean and matches how clpeak's OpenCL path totals iters.
  cuEventRecord(start, dev.stream);
  for (unsigned int i = 0; i < iters; i++)
    cuLaunchKernel(fn, gridX, 1, 1, blockX, 1, 1, 0, dev.stream, args, nullptr);
  cuEventRecord(stop, dev.stream);
  cuEventSynchronize(stop);

  float ms = 0;
  cuEventElapsedTime(&ms, start, stop);

  cuEventDestroy(start);
  cuEventDestroy(stop);

  return (ms * 1000.0f) / static_cast<float>(iters); // microseconds / iter
}

int CudaPeak::runAll()
{
  log->print(NEWLINE "=== CUDA backend ===" NEWLINE);
  if (!initDriver())
  {
    log->print("CUDA: driver init failed" NEWLINE);
    return -1;
  }
  if (devIndices.empty())
  {
    log->print("CUDA: no devices found" NEWLINE);
    return -1;
  }

  if (listDevices)
  {
    for (int idx : devIndices)
    {
      CUdevice d; cuDeviceGet(&d, idx);
      char name[256] = {0};
      cuDeviceGetName(name, sizeof(name), d);
      int maj = 0, min = 0;
      cuDeviceGetAttribute(&maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, d);
      cuDeviceGetAttribute(&min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, d);
      std::stringstream ss;
      ss << "  CUDA Device " << idx << ": " << name << " [sm_" << maj << min << "]" << NEWLINE;
      log->print(ss.str());
    }
    return 0;
  }

  // Mirror the OpenCL/Vulkan logger context stack (clpeak > platform > device).
  log->xmlOpenTag("clpeak");
  log->xmlAppendAttribs("os", OS_NAME);
  log->xmlOpenTag("platform");
  log->xmlAppendAttribs("name", "CUDA");
  log->xmlAppendAttribs("backend", "CUDA");

  for (int idx : devIndices)
  {
    CudaDevice dev;
    if (!dev.init(idx))
    {
      log->print(NEWLINE "CUDA: failed to init device " + std::to_string(idx) + NEWLINE);
      continue;
    }

    benchmark_config_t cfg = benchmark_config_t::forDevice(CL_DEVICE_TYPE_GPU);
    if (forceIters)
    {
      cfg.computeIters    = specifiedIters;
      cfg.globalBWIters   = specifiedIters;
      cfg.transferBWIters = specifiedIters;
      cfg.kernelLatencyIters = specifiedIters;
    }

    log->print(NEWLINE "CUDA Device: " + dev.info.deviceName + NEWLINE);
    log->print(TAB "Arch          : " + dev.info.archName + NEWLINE);
    log->print(TAB "Driver        : " + dev.info.driverVersion + NEWLINE);
    log->print(TAB "NVRTC         : " + dev.info.runtimeVersion + NEWLINE);
    log->print(TAB "SMs           : ");
    log->print((unsigned int)dev.info.numSMs);
    log->print(NEWLINE);
    log->print(TAB "VRAM          : ");
    log->print((unsigned int)(dev.info.totalGlobalMem / (1024 * 1024)));
    log->print(" MB" NEWLINE);

    log->xmlOpenTag("device");
    log->xmlAppendAttribs("name", dev.info.deviceName);
    log->xmlAppendAttribs("api", "cuda");
    log->xmlAppendAttribs("driver_version", dev.info.driverVersion);
    log->xmlAppendAttribs("arch", dev.info.archName);

    runComputeSP(dev, cfg);
    runComputeHP(dev, cfg);
    runComputeDP(dev, cfg);
    runComputeMP(dev, cfg);
    runComputeBF16(dev, cfg);
    runComputeInt8DP(dev, cfg);
    runComputeInt4Packed(dev, cfg);
    runWmma(dev, cfg);
    runGlobalBandwidth(dev, cfg);
    runTransferBandwidth(dev, cfg);
    runKernelLatency(dev, cfg);

    log->print(NEWLINE);
    log->xmlCloseTag(); // device
  }

  log->xmlCloseTag(); // platform
  log->xmlCloseTag(); // clpeak
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
  log->print(NEWLINE TAB);
  log->print(d.title);
  log->print(NEWLINE);
  log->xmlOpenTag(d.xmlTag);
  log->xmlAppendAttribs("unit", d.unit);
  if (d.extraAttribKey && d.extraAttribVal)
    log->xmlAppendAttribs(d.extraAttribKey, d.extraAttribVal);

  if (d.skip)
  {
    log->print(TAB TAB);
    log->print(d.skipMsg ? d.skipMsg : "Skipped");
    log->print(NEWLINE);
    log->xmlCloseTag();
    return 0;
  }

  struct Variant { const char *label; const char *kernelName; const char *src; const char *srcName; };
  std::vector<Variant> variants;
  if (d.variants && d.numVariants > 0)
    for (uint32_t i = 0; i < d.numVariants; i++)
      variants.push_back({d.variants[i].label, d.variants[i].kernelName,
                          d.variants[i].src,   d.variants[i].srcName});
  else
    variants.push_back({d.metricLabel, d.kernelName, d.src, d.srcName});

  // Size to saturate: target ~32M output elements like the Vulkan path.
  // numSMs * 2048 threads is the canonical "lots of warps in flight"
  // upper bound; clamp by total global memory just to be safe.
  const uint32_t blockSize = d.blockSize ? d.blockSize : 256;
  const uint32_t outPerBlock = d.outElemsPerBlock ? d.outElemsPerBlock : blockSize;
  uint64_t globalThreads = 32ULL * 1024 * 1024;
  uint64_t bytesPerBlock = (uint64_t)outPerBlock * d.elemSize;
  uint64_t maxBlocks = dev.info.totalGlobalMem / 4 / bytesPerBlock; // cap at 1/4 VRAM
  uint64_t wantBlocks = globalThreads / blockSize;
  uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
  uint32_t numBlocks = (uint32_t)pickBlocks;
  uint64_t bufferBytes = (uint64_t)numBlocks * bytesPerBlock;

  CUdeviceptr outputBuf = 0;
  if (cuMemAlloc(&outputBuf, bufferBytes) != CUDA_SUCCESS)
  {
    log->print(TAB TAB "Failed to allocate output buffer" NEWLINE);
    log->xmlCloseTag();
    return -1;
  }

  std::vector<const char *> nvrtcOpts;
  for (uint32_t i = 0; i < d.numExtraNvrtcOpts; i++)
    nvrtcOpts.push_back(d.extraNvrtcOpts[i]);

  for (const auto &v : variants)
  {
    log->print(TAB TAB);
    log->print(v.label);
    log->print(" : ");

    CUfunction fn;
    if (!dev.getKernel(v.src, v.srcName, v.kernelName, fn, nvrtcOpts))
    {
      log->print("compile/load failed" NEWLINE);
      continue;
    }

    // Two args: output pointer + scalar A.  cuLaunchKernel takes an array
    // of pointers to argument values; we always pass both slots since every
    // compute kernel here is declared as (T* out, scalar A).
    void *args[2];
    args[0] = &outputBuf;
    args[1] = const_cast<void *>(d.scalarArg);

    float us = runKernel(dev, fn, numBlocks, blockSize, args, cfg.computeIters);
    uint64_t totalThreads = (uint64_t)numBlocks * blockSize;
    float value = ((float)totalThreads * (float)d.workPerWI) / us / 1e3f;

    log->print(value);
    log->print(NEWLINE);
    log->xmlRecord(v.label, value);
  }

  cuMemFree(outputBuf);
  log->xmlCloseTag();
  return 0;
}

// ---------------------------------------------------------------------------
// Concrete compute benchmarks.
// ---------------------------------------------------------------------------

int CudaPeak::runComputeSP(CudaDevice &dev, benchmark_config_t &cfg)
{
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title       = "Single-precision compute (GFLOPS)";
  d.xmlTag      = "single_precision_compute";
  d.unit        = "gflops";
  d.metricLabel = "float";
  d.kernelName  = "compute_sp";
  d.src         = cuda_kernels::compute_sp_src;
  d.srcName     = cuda_kernels::compute_sp_name;
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);
  d.scalarArg   = &A;
  d.scalarSize  = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeHP(CudaDevice &dev, benchmark_config_t &cfg)
{
  static const cuda_compute_variant_t variants[] = {
    { "half",  "compute_hp",  cuda_kernels::compute_hp_src, cuda_kernels::compute_hp_name },
    { "half2", "compute_hp2", cuda_kernels::compute_hp_src, cuda_kernels::compute_hp_name },
  };
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title       = "Half-precision compute (GFLOPS)";
  d.xmlTag      = "half_precision_compute";
  d.unit        = "gflops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);    // 32-bit slot per thread; we store the reduced fp32 result
  d.scalarArg   = &A;
  d.scalarSize  = sizeof(A);
  d.skip        = !dev.info.fp16Supported;
  d.skipMsg     = "fp16 not supported on this compute capability! Skipped";
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeDP(CudaDevice &dev, benchmark_config_t &cfg)
{
  double A = 1.3;
  cuda_compute_desc_t d = {};
  d.title       = "Double-precision compute (GFLOPS)";
  d.xmlTag      = "double_precision_compute";
  d.unit        = "gflops";
  d.metricLabel = "double";
  d.kernelName  = "compute_dp";
  d.src         = cuda_kernels::compute_dp_src;
  d.srcName     = cuda_kernels::compute_dp_name;
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(double);
  d.scalarArg   = &A;
  d.scalarSize  = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeMP(CudaDevice &dev, benchmark_config_t &cfg)
{
  // Single variant: NVIDIA shader-core fp16xfp16+fp32 issues at FP32 rate.
  // The packed (HFMA2) path is fp16xfp16+fp16 -- that's compute_hp2, not MP.
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title       = "Mixed-precision compute fp16xfp16+fp32 (GFLOPS)";
  d.xmlTag      = "mixed_precision_compute";
  d.unit        = "gflops";
  d.metricLabel = "mp";
  d.kernelName  = "compute_mp";
  d.src         = cuda_kernels::compute_mp_src;
  d.srcName     = cuda_kernels::compute_mp_name;
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);
  d.scalarArg   = &A;
  d.scalarSize  = sizeof(A);
  d.skip        = !dev.info.fp16Supported;
  d.skipMsg     = "fp16 not supported on this compute capability! Skipped";
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeBF16(CudaDevice &dev, benchmark_config_t &cfg)
{
  // Single variant: shader-core bf16xbf16+fp32 issues at FP32 rate on
  // Ampere+.  Packed BF16 is reachable through tensor cores (wmma), not
  // an SFU-style packed shader instruction, so a bf16_2 variant wouldn't
  // be a different code path.
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title       = "BF16 compute bf16xbf16+fp32 (GFLOPS)";
  d.xmlTag      = "bfloat16_compute";
  d.unit        = "gflops";
  d.metricLabel = "bf16";
  d.kernelName  = "compute_bf16";
  d.src         = cuda_kernels::compute_bf16_src;
  d.srcName     = cuda_kernels::compute_bf16_name;
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);
  d.scalarArg   = &A;
  d.scalarSize  = sizeof(A);
  d.skip        = !dev.info.bf16Supported;
  d.skipMsg     = "bf16 requires sm_80 or newer (Ampere+)! Skipped";
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeInt8DP(CudaDevice &dev, benchmark_config_t &cfg)
{
  static const cuda_compute_variant_t variants[] = {
    { "int8_dp",  "compute_int8_dp",  cuda_kernels::compute_int8_dp_src, cuda_kernels::compute_int8_dp_name },
    { "int8_dp2", "compute_int8_dp2", cuda_kernels::compute_int8_dp_src, cuda_kernels::compute_int8_dp_name },
    { "int8_dp4", "compute_int8_dp4", cuda_kernels::compute_int8_dp_src, cuda_kernels::compute_int8_dp_name },
  };
  int A = 4;
  cuda_compute_desc_t d = {};
  d.title       = "INT8 dot-product compute (__dp4a) (GIOPS)";
  d.xmlTag      = "integer_compute_int8_dp";
  d.unit        = "giops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_INT8_DP_WORK_PER_WI;
  d.elemSize    = sizeof(int);
  d.scalarArg   = &A;
  d.scalarSize  = sizeof(A);
  d.skip        = !dev.info.dp4aSupported;
  d.skipMsg     = "__dp4a requires sm_61 or newer (Pascal+)! Skipped";
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeInt4Packed(CudaDevice &dev, benchmark_config_t &cfg)
{
  int A = 3;
  cuda_compute_desc_t d = {};
  d.title           = "Packed INT4 compute (emulated) (GIOPS)";
  d.xmlTag          = "int4_packed_compute";
  d.unit            = "giops";
  d.metricLabel     = "int4_packed";
  d.kernelName      = "compute_int4_packed";
  d.src             = cuda_kernels::compute_int4_packed_src;
  d.srcName         = cuda_kernels::compute_int4_packed_name;
  d.workPerWI       = COMPUTE_INT4_PACKED_WORK_PER_WI;
  d.elemSize        = sizeof(int);
  d.scalarArg       = &A;
  d.scalarSize      = sizeof(A);
  d.extraAttribKey  = "emulated";
  d.extraAttribVal  = "true";
  return runComputeKernel(dev, cfg, d);
}

// ---------------------------------------------------------------------------
// WMMA + FP8 mma.sync umbrella -- mirrors vkPeak::runCoopMatrix.
// ---------------------------------------------------------------------------

int CudaPeak::runWmma(CudaDevice &dev, benchmark_config_t &cfg)
{
  if (!dev.info.wmmaSupported)
  {
    log->print(NEWLINE TAB "WMMA tensor-core compute (TFLOPS/TOPS)" NEWLINE);
    log->xmlOpenTag("wmma");
    log->print(TAB TAB "WMMA requires sm_70 or newer (Volta+)! Skipped" NEWLINE);
    log->xmlCloseTag();
    return 0;
  }

  // Shared geometry: one warp (32 threads) per block, m16n16k16 tile per
  // wmma fragment, 256 outer iters → COOPMAT_WORK_PER_WI per thread.
  const uint32_t warp = 32;
  const uint32_t outElems = 16 * 16; // M*N

  // FP16 WMMA
  {
    float A = 1.3f;
    cuda_compute_desc_t d = {};
    d.title          = "WMMA fp16xfp16+fp32 16x16x16 (GFLOPS)";
    d.xmlTag         = "wmma_fp16";
    d.unit           = "gflops";
    d.metricLabel    = "wmma_fp16";
    d.kernelName     = "wmma_fp16";
    d.src            = cuda_kernels::wmma_fp16_src;
    d.srcName        = cuda_kernels::wmma_fp16_name;
    d.workPerWI      = COOPMAT_WORK_PER_WI * 4; // 4 parallel chains per kernel
    d.elemSize       = sizeof(float);
    d.blockSize      = warp;
    d.outElemsPerBlock = outElems;
    d.scalarArg      = &A;
    d.scalarSize     = sizeof(A);
    d.extraAttribKey = "tile";
    d.extraAttribVal = "16x16x16";
    runComputeKernel(dev, cfg, d);
  }
  // BF16 WMMA
  {
    float A = 1.3f;
    cuda_compute_desc_t d = {};
    d.title          = "WMMA bf16xbf16+fp32 16x16x16 (GFLOPS)";
    d.xmlTag         = "wmma_bf16";
    d.unit           = "gflops";
    d.metricLabel    = "wmma_bf16";
    d.kernelName     = "wmma_bf16";
    d.src            = cuda_kernels::wmma_bf16_src;
    d.srcName        = cuda_kernels::wmma_bf16_name;
    d.workPerWI      = COOPMAT_WORK_PER_WI * 4; // 4 parallel chains per kernel
    d.elemSize       = sizeof(float);
    d.blockSize      = warp;
    d.outElemsPerBlock = outElems;
    d.scalarArg      = &A;
    d.scalarSize     = sizeof(A);
    d.skip           = !dev.info.bf16Supported;
    d.skipMsg        = "bf16 WMMA requires sm_80 or newer (Ampere+)! Skipped";
    d.extraAttribKey = "tile";
    d.extraAttribVal = "16x16x16";
    runComputeKernel(dev, cfg, d);
  }
  // INT8 WMMA
  {
    int A = 3;
    cuda_compute_desc_t d = {};
    d.title          = "WMMA int8xint8+int32 16x16x16 (GIOPS)";
    d.xmlTag         = "wmma_int8";
    d.unit           = "giops";
    d.metricLabel    = "wmma_int8";
    d.kernelName     = "wmma_int8";
    d.src            = cuda_kernels::wmma_int8_src;
    d.srcName        = cuda_kernels::wmma_int8_name;
    d.workPerWI      = COOPMAT_WORK_PER_WI * 4; // 4 parallel chains per kernel
    d.elemSize       = sizeof(int);
    d.blockSize      = warp;
    d.outElemsPerBlock = outElems;
    d.scalarArg      = &A;
    d.scalarSize     = sizeof(A);
    d.skip           = !dev.info.wmmaInt8Supported;
    d.skipMsg        = "INT8 WMMA requires sm_72 or newer (Turing+)! Skipped";
    d.extraAttribKey = "tile";
    d.extraAttribVal = "16x16x16";
    runComputeKernel(dev, cfg, d);
  }
  // FP8 mma.sync E4M3 (PTX) - sm_89+
  {
    float A = 1.3f;
    cuda_compute_desc_t d = {};
    d.title          = "FP8(E4M3) mma.sync m16n8k32+fp32 (GFLOPS)";
    d.xmlTag         = "wmma_fp8_e4m3";
    d.unit           = "gflops";
    d.metricLabel    = "fp8_e4m3";
    d.kernelName     = "wmma_fp8_e4m3";
    d.src            = cuda_kernels::wmma_fp8_e4m3_src;
    d.srcName        = cuda_kernels::wmma_fp8_e4m3_name;
    d.workPerWI      = COOPMAT_WORK_PER_WI * 4; // 4 parallel chains per kernel
    d.elemSize       = sizeof(float);
    d.blockSize      = warp;
    d.outElemsPerBlock = 16 * 8; // m16n8 output tile
    d.scalarArg      = &A;
    d.scalarSize     = sizeof(A);
    d.skip           = !dev.info.fp8MmaSupported;
    d.skipMsg        = "FP8 mma.sync requires sm_89 or newer (Ada/Hopper+)! Skipped";
    d.extraAttribKey = "tile";
    d.extraAttribVal = "m16n8k32";
    runComputeKernel(dev, cfg, d);
  }
  // FP8 mma.sync E5M2 (PTX) - sm_89+
  {
    float A = 1.3f;
    cuda_compute_desc_t d = {};
    d.title          = "FP8(E5M2) mma.sync m16n8k32+fp32 (GFLOPS)";
    d.xmlTag         = "wmma_fp8_e5m2";
    d.unit           = "gflops";
    d.metricLabel    = "fp8_e5m2";
    d.kernelName     = "wmma_fp8_e5m2";
    d.src            = cuda_kernels::wmma_fp8_e5m2_src;
    d.srcName        = cuda_kernels::wmma_fp8_e5m2_name;
    d.workPerWI      = COOPMAT_WORK_PER_WI * 4; // 4 parallel chains per kernel
    d.elemSize       = sizeof(float);
    d.blockSize      = warp;
    d.outElemsPerBlock = 16 * 8;
    d.scalarArg      = &A;
    d.scalarSize     = sizeof(A);
    d.skip           = !dev.info.fp8MmaSupported;
    d.skipMsg        = "FP8 mma.sync requires sm_89 or newer (Ada/Hopper+)! Skipped";
    d.extraAttribKey = "tile";
    d.extraAttribVal = "m16n8k32";
    runComputeKernel(dev, cfg, d);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Global bandwidth (CUDA)
// ---------------------------------------------------------------------------

int CudaPeak::runGlobalBandwidth(CudaDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = cfg.globalBWIters;
  const uint32_t blockSize = 256;

  uint64_t maxItems = dev.info.totalGlobalMem / sizeof(float) / 4; // input+output, plus margin
  uint64_t numItems = (maxItems / (blockSize * FETCH_PER_WI)) * (blockSize * FETCH_PER_WI);
  if (numItems > cfg.globalBWMaxSize / sizeof(float))
    numItems = (cfg.globalBWMaxSize / sizeof(float) / (blockSize * FETCH_PER_WI)) * (blockSize * FETCH_PER_WI);

  uint32_t numBlocks = (uint32_t)(numItems / FETCH_PER_WI / blockSize);
  if (numBlocks == 0) numBlocks = 1;

  log->print(NEWLINE TAB "Global memory bandwidth (GBPS)" NEWLINE);
  log->xmlOpenTag("global_memory_bandwidth");
  log->xmlAppendAttribs("unit", "gbps");

  CUdeviceptr inBuf = 0, outBuf = 0;
  if (cuMemAlloc(&inBuf,  numItems * sizeof(float)) != CUDA_SUCCESS ||
      cuMemAlloc(&outBuf, numItems * sizeof(float)) != CUDA_SUCCESS)
  {
    log->print(TAB TAB "Failed to allocate buffers" NEWLINE);
    if (inBuf) cuMemFree(inBuf);
    log->xmlCloseTag();
    return -1;
  }
  // Touch input so we measure DRAM not zero-page.
  cuMemsetD32(inBuf, 0x3f800000u, numItems);

  CUfunction fn;
  if (!dev.getKernel(cuda_kernels::global_bandwidth_src,
                     cuda_kernels::global_bandwidth_name,
                     "global_bandwidth", fn))
  {
    cuMemFree(inBuf); cuMemFree(outBuf);
    log->print(TAB TAB "Compile failed" NEWLINE);
    log->xmlCloseTag();
    return -1;
  }

  void *args[2] = { &inBuf, &outBuf };
  log->print(TAB TAB "float   : ");
  float us = runKernel(dev, fn, numBlocks, blockSize, args, iters);
  float gbps = ((float)numItems * sizeof(float)) / us / 1e3f;
  log->print(gbps);
  log->print(NEWLINE);
  log->xmlRecord("float", gbps);

  cuMemFree(inBuf); cuMemFree(outBuf);
  log->xmlCloseTag();
  return 0;
}

// ---------------------------------------------------------------------------
// Host<->device transfer bandwidth (CUDA)
// ---------------------------------------------------------------------------

int CudaPeak::runTransferBandwidth(CudaDevice &dev, benchmark_config_t &cfg)
{
  const uint64_t bytes = cfg.transferBWMaxSize ? cfg.transferBWMaxSize : (1ull << 27);
  unsigned int iters = cfg.transferBWIters ? cfg.transferBWIters : 10;

  log->print(NEWLINE TAB "Transfer bandwidth (GBPS)" NEWLINE);
  log->xmlOpenTag("transfer_bandwidth");
  log->xmlAppendAttribs("unit", "gbps");

  CUdeviceptr dBuf = 0;
  if (cuMemAlloc(&dBuf, bytes) != CUDA_SUCCESS)
  {
    log->print(TAB TAB "Failed to allocate device buffer" NEWLINE);
    log->xmlCloseTag();
    return -1;
  }
  // Pinned host buffer -- the transfer-BW number we want is the pinned
  // path; pageable host allocations cap well below PCIe peak.
  void *hPinned = nullptr;
  if (cuMemAllocHost(&hPinned, bytes) != CUDA_SUCCESS)
  {
    cuMemFree(dBuf);
    log->print(TAB TAB "Failed to allocate pinned host buffer" NEWLINE);
    log->xmlCloseTag();
    return -1;
  }

  auto timeXfer = [&](bool h2d) -> float {
    CUevent s, e;
    cuEventCreate(&s, CU_EVENT_DEFAULT);
    cuEventCreate(&e, CU_EVENT_DEFAULT);
    // Warmup
    for (unsigned w = 0; w < warmupCount; w++)
    {
      if (h2d) cuMemcpyHtoDAsync(dBuf, hPinned, bytes, dev.stream);
      else     cuMemcpyDtoHAsync(hPinned, dBuf, bytes, dev.stream);
    }
    cuStreamSynchronize(dev.stream);
    cuEventRecord(s, dev.stream);
    for (unsigned i = 0; i < iters; i++)
    {
      if (h2d) cuMemcpyHtoDAsync(dBuf, hPinned, bytes, dev.stream);
      else     cuMemcpyDtoHAsync(hPinned, dBuf, bytes, dev.stream);
    }
    cuEventRecord(e, dev.stream);
    cuEventSynchronize(e);
    float ms = 0;
    cuEventElapsedTime(&ms, s, e);
    cuEventDestroy(s); cuEventDestroy(e);
    return (ms / iters) * 1000.0f; // microseconds per transfer
  };

  float usH2D = timeXfer(true);
  float gbpsH2D = (float)bytes / usH2D / 1e3f;
  log->print(TAB TAB "Host->Dev (pinned) : ");
  log->print(gbpsH2D);
  log->print(NEWLINE);
  log->xmlRecord("h2d_pinned", gbpsH2D);

  float usD2H = timeXfer(false);
  float gbpsD2H = (float)bytes / usD2H / 1e3f;
  log->print(TAB TAB "Dev->Host (pinned) : ");
  log->print(gbpsD2H);
  log->print(NEWLINE);
  log->xmlRecord("d2h_pinned", gbpsD2H);

  cuMemFreeHost(hPinned);
  cuMemFree(dBuf);
  log->xmlCloseTag();
  return 0;
}

// ---------------------------------------------------------------------------
// Kernel launch latency (CUDA)
// ---------------------------------------------------------------------------

int CudaPeak::runKernelLatency(CudaDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = cfg.kernelLatencyIters ? cfg.kernelLatencyIters : 1000;

  log->print(NEWLINE TAB "Kernel launch latency (us)" NEWLINE);
  log->xmlOpenTag("kernel_launch_latency");
  log->xmlAppendAttribs("unit", "us");

  CUfunction fn;
  if (!dev.getKernel(cuda_kernels::kernel_latency_src,
                     cuda_kernels::kernel_latency_name,
                     "kernel_latency_noop", fn))
  {
    log->print(TAB TAB "Compile failed" NEWLINE);
    log->xmlCloseTag();
    return -1;
  }

  void *args[1] = { nullptr }; // no args
  // Single-block, single-thread no-op so the only thing being timed is the
  // launch+complete round-trip, not any work.
  float us = runKernel(dev, fn, 1, 1, args, iters);
  log->print(TAB TAB "noop : ");
  log->print(us);
  log->print(NEWLINE);
  log->xmlRecord("noop", us);

  log->xmlCloseTag();
  return 0;
}

#endif // ENABLE_CUDA
