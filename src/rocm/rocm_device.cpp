#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <cstdio>
#include <sstream>
#include <string>

static const char *hipErrStr(hipError_t r)
{
  const char *s = hipGetErrorString(r);
  return s ? s : "unknown HIP error";
}

static std::string formatHipVersionLocal(int v)
{
  if (v <= 0)
    return "";
  if (v >= 10000000)
  {
    std::stringstream ss;
    ss << (v / 10000000) << "." << ((v / 100000) % 100) << "." << (v % 100000);
    return ss.str();
  }
  return std::to_string(v);
}

#define HIP_CHECK(call)                                                               \
  do                                                                                  \
  {                                                                                   \
    hipError_t _r = (call);                                                           \
    if (_r != hipSuccess)                                                             \
    {                                                                                 \
      CLPEAK_VLOG("HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipErrStr(_r));      \
      return false;                                                                   \
    }                                                                                 \
  } while (0)

RocmDevice::RocmDevice() : deviceIndex(-1), stream(nullptr) {}

RocmDevice::~RocmDevice() { cleanup(); }

bool RocmDevice::init(int devIndex)
{
  deviceIndex = devIndex;
  HIP_CHECK(hipSetDevice(devIndex));

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, devIndex));

  info.deviceName = props.name;
  info.archName = props.gcnArchName[0] ? props.gcnArchName : "gfx";
  info.numCUs = props.multiProcessorCount;
  info.maxThreadsPerBlock = props.maxThreadsPerBlock;
  info.totalGlobalMem = props.totalGlobalMem;
  info.clockRateKHz = props.clockRate;
  info.warpSize = props.warpSize;
  info.deviceType = DeviceType::Gpu;

  int driverVer = 0;
  if (hipDriverGetVersion(&driverVer) == hipSuccess)
    info.driverVersion = formatHipVersionLocal(driverVer);
  int runtimeVer = 0;
  if (hipRuntimeGetVersion(&runtimeVer) == hipSuccess)
    info.runtimeVersion = formatHipVersionLocal(runtimeVer);

  // HIP's FP16/BF16 language types exist across current AMD ROCm targets.
  // If a specific ASIC lowers a path through emulation, the benchmark still
  // reports the effective HIP-native rate.
  info.fp16Supported = true;
  info.bf16Supported = true;
  // gcnArchName carries feature flags (e.g. "gfx942:sramecc+:xnack-").
  // Match against the base ISA only, otherwise the suffix makes every
  // exact comparison miss and rocWMMA looks "unsupported" on supported GPUs.
  const std::string archBase = info.archName.substr(0, info.archName.find(':'));
  info.rocwmmaSupported =
      archBase == "gfx908" || archBase == "gfx90a" ||
      archBase == "gfx940" || archBase == "gfx941" ||
      archBase == "gfx942" || archBase == "gfx950" ||
      archBase == "gfx1100" || archBase == "gfx1101" ||
      archBase == "gfx1102" || archBase == "gfx1200" ||
      archBase == "gfx1201";

  HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
  return true;
}

void RocmDevice::cleanup()
{
  for (auto &kv : moduleCache)
    (void)hipModuleUnload(kv.second);
  moduleCache.clear();

  if (stream)
  {
    (void)hipStreamDestroy(stream);
    stream = nullptr;
  }
}

bool RocmDevice::getKernel(const rocm_kernels::Blob &blob,
                           const char *kernelName, hipFunction_t &fn)
{
  // Cache by blob-data pointer: every embedded code object is a distinct array
  // in rocm_kernels_generated, so pointer equality is sufficient.
  auto it = moduleCache.find(blob.data);
  hipModule_t mod = nullptr;
  if (it != moduleCache.end())
  {
    mod = it->second;
  }
  else
  {
    if (blob.len == 0 || blob.data == nullptr)
    {
      CLPEAK_VLOG("kernel %s was not built for any supported gfx arch\n", blob.name);
      return false;
    }
    // The blob is a precompiled code-object bundle; the HIP runtime selects the
    // slice matching this device's gfx arch -- no HIPRTC, no ROCm headers.
    hipError_t hr = hipModuleLoadData(&mod, blob.data);
    if (hr != hipSuccess)
    {
      CLPEAK_VLOG("hipModuleLoadData(%s) failed: %s\n", blob.name, hipErrStr(hr));
      return false;
    }
    moduleCache[blob.data] = mod;
  }

  hipError_t r = hipModuleGetFunction(&fn, mod, kernelName);
  if (r != hipSuccess)
  {
    CLPEAK_VLOG("hipModuleGetFunction(%s in %s) failed: %s\n",
                kernelName, blob.name, hipErrStr(r));
    return false;
  }
  return true;
}

#endif // ENABLE_ROCM
