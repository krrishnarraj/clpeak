#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#ifndef CLPEAK_ROCM_INCLUDE_DIR
#define CLPEAK_ROCM_INCLUDE_DIR "/opt/rocm/include"
#endif

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
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipErrStr(_r)); \
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
  info.rocwmmaSupported =
      info.archName == "gfx908" || info.archName == "gfx90a" ||
      info.archName == "gfx940" || info.archName == "gfx941" ||
      info.archName == "gfx942" || info.archName == "gfx950" ||
      info.archName == "gfx1100" || info.archName == "gfx1101" ||
      info.archName == "gfx1102" || info.archName == "gfx1200" ||
      info.archName == "gfx1201";

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

bool RocmDevice::getKernel(const char *src, const char *srcName,
                           const char *kernelName, hipFunction_t &fn,
                           const std::vector<const char *> &extraOpts)
{
  auto it = moduleCache.find(src);
  hipModule_t mod = nullptr;
  if (it != moduleCache.end())
  {
    mod = it->second;
  }
  else
  {
    hiprtcProgram prog;
    hiprtcResult rr = hiprtcCreateProgram(&prog, src, srcName, 0, nullptr, nullptr);
    if (rr != HIPRTC_SUCCESS)
    {
      fprintf(stderr, "hiprtcCreateProgram failed: %s\n", hiprtcGetErrorString(rr));
      return false;
    }

    std::vector<std::string> ownedOpts;
    if (!info.archName.empty())
      ownedOpts.push_back(std::string("--gpu-architecture=") + info.archName);
    ownedOpts.push_back(std::string("-I") + CLPEAK_ROCM_INCLUDE_DIR);
#ifdef CLPEAK_ROCWMMA_INCLUDE_DIR
    ownedOpts.push_back(std::string("-I") + CLPEAK_ROCWMMA_INCLUDE_DIR);
#endif
    ownedOpts.push_back("-D__HIP_PLATFORM_AMD__=1");
    ownedOpts.push_back("-O3");

    std::vector<const char *> opts;
    for (const auto &o : ownedOpts)
      opts.push_back(o.c_str());
    for (auto *e : extraOpts)
      opts.push_back(e);

    rr = hiprtcCompileProgram(prog, (int)opts.size(), opts.data());
    if (rr != HIPRTC_SUCCESS)
    {
      size_t logSize = 0;
      hiprtcGetProgramLogSize(prog, &logSize);
      std::string log(logSize, '\0');
      if (logSize > 1)
        hiprtcGetProgramLog(prog, &log[0]);
      fprintf(stderr, "HIPRTC compile of %s failed:\n%s\n", srcName, log.c_str());
      hiprtcDestroyProgram(&prog);
      return false;
    }

    size_t codeSize = 0;
    rr = hiprtcGetCodeSize(prog, &codeSize);
    if (rr != HIPRTC_SUCCESS || codeSize == 0)
    {
      fprintf(stderr, "hiprtcGetCodeSize(%s) failed: %s\n",
              srcName, hiprtcGetErrorString(rr));
      hiprtcDestroyProgram(&prog);
      return false;
    }

    std::vector<char> code(codeSize);
    rr = hiprtcGetCode(prog, code.data());
    hiprtcDestroyProgram(&prog);
    if (rr != HIPRTC_SUCCESS)
    {
      fprintf(stderr, "hiprtcGetCode(%s) failed: %s\n", srcName, hiprtcGetErrorString(rr));
      return false;
    }

    hipError_t hr = hipModuleLoadData(&mod, code.data());
    if (hr != hipSuccess)
    {
      fprintf(stderr, "hipModuleLoadData(%s) failed: %s\n", srcName, hipErrStr(hr));
      return false;
    }
    moduleCache[src] = mod;
  }

  hipError_t r = hipModuleGetFunction(&fn, mod, kernelName);
  if (r != hipSuccess)
  {
    fprintf(stderr, "hipModuleGetFunction(%s in %s) failed: %s\n",
            kernelName, srcName, hipErrStr(r));
    return false;
  }
  return true;
}

#endif // ENABLE_ROCM
