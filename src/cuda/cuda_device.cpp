#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <cstring>
#include <cstdio>
#include <sstream>
#include <vector>
#include <string>

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
      CLPEAK_VLOG("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cuErrStr(_r));      \
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
    // The kernels are precompiled fatbins; report the CUDA toolkit the build
    // was compiled against (CUDA_VERSION from <cuda.h>) rather than an NVRTC
    // version, since NVRTC is no longer used at runtime.
    std::stringstream ss;
    ss << (CUDA_VERSION / 1000) << "." << (CUDA_VERSION % 1000) / 10;
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
  info.fp4MmaSparseSupported = (info.major >= 12);
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

bool CudaDevice::getKernel(const cuda_kernels::Blob &blob,
                           const char *kernelName, CUfunction &fn)
{
  // Cache by blob-data pointer: every embedded fatbin is a distinct array in
  // cuda_kernels_generated.cpp, so pointer equality is sufficient.
  auto it = moduleCache.find(blob.data);
  CUmodule mod = nullptr;
  if (it != moduleCache.end())
  {
    mod = it->second;
  }
  else
  {
    // The blob is a precompiled multi-arch fatbin.  cuModuleLoadData selects
    // the cubin matching this device's compute capability, or JITs the
    // embedded PTX via the driver -- no NVRTC, no toolkit headers needed.
    CUresult lr = cuModuleLoadData(&mod, blob.data);
    if (lr != CUDA_SUCCESS)
    {
      CLPEAK_VLOG("cuModuleLoadData(%s) failed: %s\n", blob.name, cuErrStr(lr));
      return false;
    }
    moduleCache[blob.data] = mod;
  }

  CUresult r = cuModuleGetFunction(&fn, mod, kernelName);
  if (r != CUDA_SUCCESS)
  {
    CLPEAK_VLOG("cuModuleGetFunction(%s in %s) failed: %s\n",
                kernelName, blob.name, cuErrStr(r));
    return false;
  }
  return true;
}

#endif // ENABLE_CUDA
