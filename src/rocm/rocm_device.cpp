#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

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

// For init() only: also records what failed in initError, which is all a run
// without --verbose says about a device that would not initialize.
#define HIP_CHECK(call, what)                                                         \
  do                                                                                  \
  {                                                                                   \
    hipError_t _r = (call);                                                           \
    if (_r != hipSuccess)                                                             \
    {                                                                                 \
      CLPEAK_VLOG("HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipErrStr(_r));      \
      initError = std::string(what) + ": " + hipErrStr(_r);                           \
      return false;                                                                   \
    }                                                                                 \
  } while (0)

RocmDevice::RocmDevice() : deviceIndex(-1), stream(nullptr) {}

RocmDevice::~RocmDevice() { cleanup(); }

bool RocmDevice::init(int devIndex)
{
  deviceIndex = devIndex;
  HIP_CHECK(hipSetDevice(devIndex), "hipSetDevice");

  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, devIndex), "hipGetDeviceProperties");

  info.deviceName = props.name;
  info.archName = props.gcnArchName[0] ? props.gcnArchName : "gfx";
  info.numCUs = props.multiProcessorCount;
  info.maxThreadsPerBlock = props.maxThreadsPerBlock;
  info.totalGlobalMem = props.totalGlobalMem;
  info.clockRateKHz = props.clockRate;
  info.l2CacheSize = props.l2CacheSize > 0 ? (uint64_t)props.l2CacheSize : 0;
  info.integrated = props.integrated != 0;
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
  // gcnArchName carries feature flags (e.g. "gfx942:sramecc+:xnack-"); a
  // bundle names its slices by processor alone, so compare against that.
  archBase = info.archName.substr(0, info.archName.find(':'));

  // compute_sp is built for every arch in the build (CLPEAK_ROCM_ALL in
  // src/rocm/CMakeLists.txt), so its bundle is the build's coverage.
  archCovered = sliceFor(rocm_kernels::compute_sp) != Slice::Absent;

  // HIP answers hipErrorOutOfMemory for any failure to set a stream up, not
  // only memory: the first stream also builds HIP's own blit kernels, and a
  // failed build reads the same.  On a Radeon 890M that was LLVM's verifier
  // failing inside a GUI process whose GL driver had loaded another LLVM
  // (src/ffi/engine.cpp) -- visible only in the HIP runtime's own log, so the
  // note points there.
  const hipError_t sr = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
  if (sr != hipSuccess)
  {
    CLPEAK_VLOG("HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipErrStr(sr));
    initError = std::string("hipStreamCreateWithFlags: ") + hipErrStr(sr) +
                " (HIP reports any failure to set up a stream this way; "
                "AMD_LOG_LEVEL=1 prints the cause)";
    return false;
  }
  return true;
}

// The gfx processors an embedded code-object bundle has a slice for, read from
// its clang offload-bundle header -- the table the HIP runtime itself consults
// to pick a slice: the magic, an entry count, then per entry its offset, size,
// triple length and triple, integers little-endian 64-bit.  A device entry's
// triple reads "hipv4-amdgcn-amd-amdhsa--gfx1150", with ":xnack+"-style
// suffixes when built for a feature; the host entry has no amdgcn triple.
// False when the bytes are not an uncompressed bundle: a compressed one
// ("CCOB") keeps its table behind its codec.
static bool bundleArchs(const rocm_kernels::Blob &blob, std::vector<std::string> &archs)
{
  static const char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  static const char kDevice[] = "amdgcn-amd-amdhsa-";
  const size_t magicLen = sizeof(kMagic) - 1;
  const unsigned char *p = blob.data;
  const size_t len = blob.len;

  archs.clear();
  if (!p || len < magicLen + 8 || std::memcmp(p, kMagic, magicLen) != 0)
    return false;

  auto u64 = [&](size_t at, uint64_t &v) {
    if (at > len || len - at < 8)
      return false;
    v = 0;
    for (int i = 7; i >= 0; i--)
      v = (v << 8) | p[at + i];
    return true;
  };

  uint64_t entries = 0;
  if (!u64(magicLen, entries))
    return false;
  size_t at = magicLen + 8;
  for (uint64_t e = 0; e < entries; e++)
  {
    uint64_t tripleLen = 0;
    if (!u64(at + 16, tripleLen))   // past the entry's offset and size
      return false;
    at += 24;
    if (tripleLen > len - at)
      return false;
    const std::string triple(reinterpret_cast<const char *>(p + at), (size_t)tripleLen);
    at += (size_t)tripleLen;

    // "<kind>-amdgcn-amd-amdhsa-<env>-<target id>": the id follows the dash
    // that closes the (empty) environment, and may hold dashes of its own
    // (gfx11-generic).
    const size_t t = triple.find(kDevice);
    if (t == std::string::npos)
      continue;
    const size_t dash = triple.find('-', t + sizeof(kDevice) - 1);
    if (dash == std::string::npos)
      continue;
    const std::string id = triple.substr(dash + 1);
    archs.push_back(id.substr(0, id.find(':')));
  }
  return true;
}

RocmDevice::Slice RocmDevice::sliceFor(const rocm_kernels::Blob &blob) const
{
  // A stub: the toolkit could target none of the kernel's arch group.
  if (blob.len == 0 || blob.data == nullptr)
    return Slice::Absent;

  std::vector<std::string> archs;
  if (!bundleArchs(blob, archs))
    return Slice::Unknown;

  bool generic = false;
  for (const auto &a : archs)
  {
    if (a == archBase)
      return Slice::Present;
    // A generic slice (gfx11-generic) runs on members of its family where the
    // runtime supports it, which only the load can say.
    generic = generic || (a.size() > 8 && a.compare(a.size() - 8, 8, "-generic") == 0);
  }
  return generic ? Slice::Unknown : Slice::Absent;
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

RocmKernel RocmDevice::getKernel(const rocm_kernels::Blob &blob,
                                 const char *kernelName, const char *notBuilt)
{
  RocmKernel k;

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
    if (sliceFor(blob) == Slice::Absent)
    {
      CLPEAK_VLOG("%s has no %s code object\n", blob.name, archBase.c_str());
      if (!archCovered)
      {
        k.reason = "this build has no " + archBase + " code";
      }
      else if (notBuilt)
      {
        k.status = ResultStatus::Unsupported;
        k.reason = notBuilt;
      }
      else
      {
        k.reason = std::string(blob.name) + " was not built for " + archBase;
      }
      return k;
    }
    // The blob is a precompiled code-object bundle; the HIP runtime selects the
    // slice matching this device's gfx arch -- no HIPRTC, no ROCm headers.
    hipError_t hr = hipModuleLoadData(&mod, blob.data);
    if (hr != hipSuccess)
    {
      CLPEAK_VLOG("hipModuleLoadData(%s) failed: %s\n", blob.name, hipErrStr(hr));
      k.reason = std::string("code object failed to load: ") + hipErrStr(hr);
      return k;
    }
    moduleCache[blob.data] = mod;
  }

  hipError_t r = hipModuleGetFunction(&k.fn, mod, kernelName);
  if (r != hipSuccess)
  {
    CLPEAK_VLOG("hipModuleGetFunction(%s in %s) failed: %s\n",
                kernelName, blob.name, hipErrStr(r));
    k.fn = nullptr;
    k.reason = std::string("kernel missing from its code object: ") + hipErrStr(r);
    return k;
  }
  k.status = ResultStatus::Ok;
  return k;
}

#endif // ENABLE_ROCM
