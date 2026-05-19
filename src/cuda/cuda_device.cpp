#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
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

#endif // ENABLE_CUDA
