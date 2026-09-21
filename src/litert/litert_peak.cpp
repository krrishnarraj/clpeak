#ifdef ENABLE_LITERT

#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_runtime.h"
#include "litert_session.h"

#include <common/console_mute.h>
#include <common/dynlib.h>
#include <common/options.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <ostream>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

// ---------------------------------------------------------------------------
// Devices
// ---------------------------------------------------------------------------
// LiteRT does not enumerate hardware.  It has three accelerator slots -- CPU,
// GPU, NPU -- and each is present when its library loads and its first model
// compiles with nothing handed back to the CPU.  So a device here is an
// accelerator that has proved itself on one tiny graph, named as precisely
// as the runtime lets us: the GPU by the accelerator library that answered
// (Metal, OpenCL, WebGPU), the NPU by the vendor dispatch library beside the
// runtime.  Accelerators first, CPU last, like every other backend.

namespace
{

// Which vendor dispatch libraries sit in the NPU directory.  LiteRT loads
// libLiteRtDispatch_<Vendor>; the file name is the only place the vendor's
// identity is written down before a model compiles.
struct NpuVendor
{
  const char *fileTag;   // in libLiteRtDispatch_<fileTag>
  const char *vendor;
  const char *display;
};
const NpuVendor kNpuVendors[] = {
    {"Qualcomm", "Qualcomm", "Qualcomm Hexagon NPU via LiteRT"},
    {"MediaTek", "MediaTek", "MediaTek NeuroPilot APU via LiteRT"},
    {"GoogleTensor", "Google", "Google Tensor TPU via LiteRT"},
    {"Samsung", "Samsung", "Samsung Exynos AI LiteCore NPU via LiteRT"},
    {"IntelOpenvino", "Intel", "Intel NPU via LiteRT (OpenVINO)"},
};

// Which vendors' dispatch libraries sit in `dir`, in the table's order.  A
// directory listing first; where there is no directory to list -- an
// Android app's libraries stay inside the APK ("base.apk!/lib/arm64-v8a",
// which the linker opens but no directory iterator can) -- each vendor's
// library is tried by name, the way LiteRT itself will open it.  More than
// one is normal: tools/fetch_litert_npu.sh stages every vendor's shim, and
// only the one for the silicon underneath will bring up a device.
std::vector<const NpuVendor *> findNpuVendors(const std::string &dir)
{
  std::vector<const NpuVendor *> found;
  if (dir.empty())
    return found;
  std::error_code ec;
  std::filesystem::directory_iterator it(dir, ec), end;
  if (!ec)
  {
    std::vector<std::string> names;
    for (; !ec && it != end; it.increment(ec))
      names.push_back(it->path().filename().string());
    for (const auto &v : kNpuVendors)
      for (const std::string &name : names)
        if ((name.rfind("libLiteRtDispatch_", 0) == 0 || name.rfind("LiteRtDispatch", 0) == 0) &&
            name.find(v.fileTag) != std::string::npos)
        {
          found.push_back(&v);
          break;
        }
    return found;
  }
  for (const auto &v : kNpuVendors)
  {
    const std::string name = std::string("libLiteRtDispatch_") + v.fileTag + ".so";
    const std::string full = dir + "/" + name;
    if (clpeak::dynOpen({full.c_str(), name.c_str()}))
      found.push_back(&v);
  }
  return found;
}

// Does a file name carry a vendor's tag, however the vendor spelled it?
// The shims are not consistent among themselves: Google's dispatch library
// is libLiteRtDispatch_GoogleTensor.so and its compiler plugin
// libLiteRtCompilerPlugin_google_tensor.so.  Case and underscores are
// dropped from both sides before comparing.
bool nameCarriesTag(const std::string &name, const std::string &tag)
{
  auto squash = [](const std::string &in) {
    std::string out;
    for (char c : in)
      if (c != '_')
        out += (char)std::tolower((unsigned char)c);
    return out;
  };
  return squash(name).find(squash(tag)) != std::string::npos;
}

// Which vendor's NPU this device carries, from the system's own answer.
// `ro.soc.manufacturer` (Android 12+, Build.SOC_MANUFACTURER) says "Google",
// "QTI", "Mediatek", "Samsung"; the shims are named after the same vendors.
// Null when the property is absent or names nobody in the table -- which is
// not a guess: the caller then leaves LiteRT to its own choice.
const NpuVendor *vendorOfThisSoc(std::string *socModel)
{
#if defined(__ANDROID__)
  char buf[PROP_VALUE_MAX] = {0};
  if (socModel && __system_property_get("ro.soc.model", buf) > 0)
    *socModel = buf;
  buf[0] = 0;
  if (__system_property_get("ro.soc.manufacturer", buf) <= 0)
    return nullptr;
  std::string maker;
  for (const char *c = buf; *c; c++)
    maker += (char)std::tolower((unsigned char)*c);
  struct { const char *needle; const char *vendor; } const kMakers[] = {
      {"google", "Google"},   {"qti", "Qualcomm"},     {"qualcomm", "Qualcomm"},
      {"mediatek", "MediaTek"}, {"samsung", "Samsung"}, {"intel", "Intel"},
  };
  for (const auto &m : kMakers)
    if (maker.find(m.needle) != std::string::npos)
      for (const auto &v : kNpuVendors)
        if (std::string(v.vendor) == m.vendor)
          return &v;
  return nullptr;
#else
  (void)socModel;
  return nullptr;
#endif
}

// One vendor's shims in a directory of their own (Android).  LiteRT loads
// the first libLiteRtDispatch_* it lists in the dispatch directory, so an
// app that packages every vendor's shims -- the one APK that serves a
// Pixel and a Snapdragon alike -- has to hand it a directory holding one
// vendor's, and the app's lib dir is not that.  With a stage directory
// given (litertSetNpuStageDir: the app's support directory) and more than
// one vendor beside the runtime, the SoC's vendor gets `<stage>/<tag>/`
// holding links to its packaged files.  Links, not copies: the lib dir
// moves with every install, so the links are remade at every launch, and
// nothing is duplicated.  With one vendor or none, nothing changes.
//
// The Qualcomm runtime's Hexagon-side libraries (libQnnHtpV*Skel.so) stay
// in the lib dir, and the DSP's loader finds them through
// ADSP_LIBRARY_PATH: LiteRT points that at the dispatch directory unless it
// is already set, in which case it prepends -- so it is set here first, to
// the lib dir, and the ONNX QNN plugin honours an existing value the same
// way.
void litertStageNpuVendor(const LitertRuntime &rt, std::string *socModel)
{
  litertSetNpuResolvedDir("");
  if (!litertNpuDirOverride().empty())
    return;
  const std::string stage = litertNpuStageDir();
  const std::vector<const NpuVendor *> present = findNpuVendors(rt.libraryDir);
  const NpuVendor *pick = vendorOfThisSoc(socModel);
  if (present.size() <= 1 || stage.empty())
    return;
  if (!pick)
  {
    CLPEAK_VLOG("litert: %zu vendors' NPU shims beside the runtime and no SoC vendor "
                "to choose by (ro.soc.manufacturer); LiteRT loads the first it lists\n",
                present.size());
    return;
  }
  if (std::find(present.begin(), present.end(), pick) == present.end())
  {
    CLPEAK_VLOG("litert: this SoC is %s's and no %s shim is beside the runtime\n",
                pick->vendor, pick->vendor);
    return;
  }

  std::error_code ec;
  const std::filesystem::path dir = std::filesystem::path(stage) / pick->fileTag;
  std::filesystem::remove_all(dir, ec);
  std::filesystem::create_directories(dir, ec);
  if (ec)
  {
    CLPEAK_VLOG("litert: cannot stage %s's shims in %s: %s\n", pick->vendor,
                dir.string().c_str(), ec.message().c_str());
    return;
  }
  int linked = 0;
  std::filesystem::directory_iterator it(rt.libraryDir, ec), end;
  for (; !ec && it != end; it.increment(ec))
  {
    const std::string name = it->path().filename().string();
    const bool shim = name.rfind("libLiteRtDispatch_", 0) == 0 ||
                      name.rfind("libLiteRtCompilerPlugin_", 0) == 0;
    if (!shim || !nameCarriesTag(name, pick->fileTag))
      continue;
    std::error_code lec;
    std::filesystem::create_symlink(it->path(), dir / name, lec);
    if (lec)
      CLPEAK_VLOG("litert: cannot link %s into %s: %s\n", name.c_str(),
                  dir.string().c_str(), lec.message().c_str());
    else
      linked++;
  }
  if (!linked)
    return;
#if !defined(_WIN32)
  if (std::string(pick->vendor) == "Qualcomm")
    setenv("ADSP_LIBRARY_PATH", rt.libraryDir.c_str(), /*overwrite=*/0);
#endif
  CLPEAK_VLOG("litert: staged %d of %s's shims in %s for this SoC\n", linked,
              pick->vendor, dir.string().c_str());
  litertSetNpuResolvedDir(dir.string());
}

// The GPU accelerator library that registered, from what LiteRT logged
// while the environment came up ("Dynamically loaded GPU accelerator
// (libLiteRtMetalAccelerator.dylib) registered.").
std::string gpuBackendFrom(const std::string &log)
{
  if (log.find("MetalAccelerator") != std::string::npos)
    return "Metal";
  if (log.find("OpenClAccelerator") != std::string::npos || log.find("ClGlAccelerator") != std::string::npos)
    return "OpenCL";
  if (log.find("WebGpuAccelerator") != std::string::npos)
    return "WebGPU";
  if (log.find("VulkanAccelerator") != std::string::npos)
    return "Vulkan";
  return std::string();
}

// The usable-device list, probed once per runtime and NPU directory.  The
// runtime is identified by its library handle, not by the address of the
// record litertRuntime() returns: that record is the loader's one static,
// so its address is the same whichever library is loaded into it, while a
// handle is unique per mapped file and never unmapped (common/dynlib.h).
// Both can change between GUI runs (Settings: library, NPU directory), and
// each change means a fresh probe -- the accelerators a runtime brings are
// that runtime's.
std::mutex g_devMutex;
bool g_devProbed = false;
const void *g_devRuntime = nullptr;   // LitertRuntime::lib
std::string g_devNpuDir;
std::vector<litert_device_info_t> g_devs;
std::vector<std::pair<litert_device_info_t, std::string>> g_skipped;

// One tiny fp32 graph on the accelerator.  Its creation log is what names
// the GPU backend, so it is returned too.
bool probeAccel(const LitertRuntime &rt, litert_device_info_t &dev, std::string &reason,
                std::string &log)
{
  const LitertPlan plan = litertPlanFor(LitertFormat::Fp32, dev.accel);
  std::string err;
  auto s = LitertSession::create(rt, dev, litertTrivialModel(plan, 64), litertConfigFor(plan), err);
  if (!s)
  {
    reason = err;
    return false;
  }
  log = litertEnvironmentLog(dev.accel) + s->creationLog;
  if (!s->onDevice())
  {
    reason = "the accelerator loaded but took no operation of a one-node graph (" +
             s->offDevice() + ")";
    return false;
  }
  std::vector<float> x(64, 0.5f);
  if (!s->writeInput(0, x.data(), x.size() * 4, err) || !s->run(err))
  {
    reason = "a one-node graph did not run: " + err;
    return false;
  }
  return true;
}

} // namespace

std::vector<litert_device_info_t> litertUsableDevices(
    const LitertRuntime &rt, std::vector<std::pair<litert_device_info_t, std::string>> *skipped)
{
  std::lock_guard<std::mutex> lock(g_devMutex);
  if (!g_devProbed || g_devRuntime != rt.lib || g_devNpuDir != litertNpuDir())
  {
    g_devProbed = true;
    g_devRuntime = rt.lib;
    g_devNpuDir = litertNpuDir();
    g_devs.clear();
    g_skipped.clear();

    // NPU: only when a vendor dispatch library is there to be loaded; LiteRT
    // otherwise logs a request for DispatchLibraryDir and takes the graph
    // on the CPU, which the fallback guard would catch one step later.
    // LiteRT itself picks the library: it lists the directory and loads the
    // first libLiteRtDispatch_* it finds (litert_dispatch.cc, with a warning
    // when there are several), once per process.  So the device is whatever
    // it loaded -- its log names the path -- and a directory holding one
    // vendor's shim is the way to choose, which litertStageNpuVendor
    // arranges on a phone carrying several.
    {
      litert_device_info_t dev;
      dev.accel = LitertAccel::Npu;
      dev.typeStr = "NPU";
      dev.deviceType = DeviceType::Accelerator;
      std::string socModel;
      litertStageNpuVendor(rt, &socModel);
      g_devNpuDir = litertNpuDir();
      dev.detail = socModel;
      const std::vector<const NpuVendor *> vendors = findNpuVendors(litertNpuDir());
      if (vendors.empty())
      {
        dev.displayName = "NPU via LiteRT";
        g_skipped.emplace_back(dev, "no libLiteRtDispatch_<vendor> library in " +
                                        (litertNpuDir().empty() ? std::string("the runtime's directory")
                                                                : litertNpuDir()) +
                                        "; --litert-npu-dir names where the vendor runtime is");
      }
      else
      {
        dev.vendor = vendors.front()->vendor;
        dev.displayName = vendors.front()->display;
        std::string reason, log;
        const bool ok = probeAccel(rt, dev, reason, log);
        // Which shim LiteRT actually loaded, from its own log line.
        const std::string envLog = litertEnvironmentLog(LitertAccel::Npu);
        for (const auto &v : kNpuVendors)
          if (envLog.find(std::string("libLiteRtDispatch_") + v.fileTag) != std::string::npos ||
              envLog.find(std::string("LiteRtDispatch_") + v.fileTag) != std::string::npos)
          {
            dev.vendor = v.vendor;
            dev.displayName = v.display;
            break;
          }
        if (ok)
          g_devs.push_back(dev);
        else
        {
          if (vendors.size() > 1)
            reason += " (several vendors' dispatch libraries are staged in " + litertNpuDir() +
                      " and LiteRT loads the first it lists; stage only the one for this device)";
          g_skipped.emplace_back(dev, reason);
        }
      }
    }

    // GPU: the accelerator library beside the runtime, whichever API it is
    // built on.  The environment comes up first, on its own: which library
    // registered is in its log, and the OpenCL one dereferences a null
    // (strlen inside libLiteRtClGlAccelerator, LiteRT 2.2.0) when it
    // compiles a model on a machine with no OpenCL library at all -- an
    // emulator, a box without a driver -- rather than declining.  So when
    // that is the accelerator, an OpenCL library has to be loadable before
    // a model is sent to it.
    {
      litert_device_info_t dev;
      dev.accel = LitertAccel::Gpu;
      dev.typeStr = "GPU";
      dev.deviceType = DeviceType::Gpu;
      dev.displayName = "GPU via LiteRT";
      std::string reason, log;
      bool ok = litertPrepareEnvironment(rt, dev.accel, reason);
      if (ok)
      {
        dev.vendor = gpuBackendFrom(litertEnvironmentLog(dev.accel));
        if (dev.vendor == "OpenCL")
        {
          void *cl = clpeak::dynOpen({
#if defined(__ANDROID__)
              "libOpenCL.so", "libOpenCL-pixel.so", "libOpenCL-car.so",
#elif defined(_WIN32)
              "OpenCL.dll",
#else
              "libOpenCL.so.1", "libOpenCL.so",
#endif
          });
          if (!cl)
          {
            ok = false;
            reason = "LiteRT's GPU accelerator here is the OpenCL one and no OpenCL library can be "
                     "loaded; the accelerator crashes rather than declining in that case (LiteRT "
                     "2.2.0), so no model is sent to it";
          }
          // The handle is kept: the accelerator dlopens the same library.
        }
      }
      if (ok && probeAccel(rt, dev, reason, log))
      {
        if (dev.vendor.empty())
          dev.vendor = gpuBackendFrom(log);
        if (!dev.vendor.empty())
          dev.displayName += " (" + dev.vendor + ")";
        g_devs.push_back(dev);
      }
      else
        g_skipped.emplace_back(dev, reason);
    }

    // CPU: XNNPACK, built into the runtime.
    {
      litert_device_info_t dev;
      dev.accel = LitertAccel::Cpu;
      dev.typeStr = "CPU";
      dev.deviceType = DeviceType::Cpu;
      dev.displayName = "CPU via LiteRT (XNNPACK)";
      std::string reason, log;
      if (probeAccel(rt, dev, reason, log))
        g_devs.push_back(dev);
      else
        g_skipped.emplace_back(dev, reason);
    }
  }
  if (skipped)
    *skipped = g_skipped;
  return g_devs;
}

LitertRuntimeStatus litertRuntimeStatus()
{
  LitertRuntimeStatus st;
  if (const LitertRuntime *rt = litertRuntime())
  {
    st.available = true;
    st.version = "ABI " + rt->abiVersion;
    st.path = rt->path;
  }
  else
  {
    st.error = litertLoadDiagnostic();
    if (st.error.empty())
      st.error = "LiteRT library not found";
  }
  return st;
}

// ---------------------------------------------------------------------------
// LitertPeak
// ---------------------------------------------------------------------------

LitertPeak::LitertPeak() = default;
LitertPeak::~LitertPeak() = default;

int LitertPeak::runAll()
{
  const LitertRuntime *rt = litertRuntime();
  if (!rt)
  {
    const std::string why = litertLoadDiagnostic();
    log->note("LiteRT: " + (why.empty() ? std::string("LiteRT library (libLiteRt) not found") : why) + "\n");
    return 0;   // absent runtime is not an error, like a missing GPU driver
  }

  std::vector<std::pair<litert_device_info_t, std::string>> skipped;
  auto devs = litertUsableDevices(*rt, &skipped);
  // Verbose only: an accelerator this machine has no library or silicon for
  // is the normal case, not a warning.
  for (const auto &sk : skipped)
    CLPEAK_VLOG("LiteRT: skipping %s (%s)\n", sk.first.displayName.c_str(), sk.second.c_str());
  if (devs.empty())
  {
    log->note("LiteRT: no accelerator could run a model\n");
    return 0;
  }

  auto backendScope = log->beginBackend("LiteRT");

  // The fills round to half in hardware where there is hardware for it;
  // once per run, under --verbose, say so if it ever disagrees with the
  // routine it stands in for (it should not: both are IEEE round-to-nearest).
  if (clpeak::verboseEnabled() && !litertHalfConversionsAgree())
    CLPEAK_VLOG("LiteRT: the hardware fp16 conversion disagrees with the reference routine\n");

  for (int idx = 0; idx < (int)devs.size(); idx++)
  {
    if (clpeak::cancelRequested())
      break;
    if (!isDeviceSelected(idx))
      continue;

    const litert_device_info_t &dev = devs[idx];

    benchmark_config_t cfg = benchmark_config_t::forDevice(dev.deviceType);
    cfg.targetTimeUs = targetTimeUs;

    std::vector<DeviceProp> details = {
        {"Accelerator", litertAccelName(dev.accel)},
        {"Type", dev.typeStr},
        {"LiteRT", "ABI " + rt->abiVersion},
    };
    if (!dev.vendor.empty())
      details.push_back({dev.accel == LitertAccel::Gpu ? "GPU backend" : "NPU vendor", dev.vendor});
    if (!dev.detail.empty())
      details.push_back({"Device", dev.detail});

    auto deviceScope = backendScope.beginDevice({
        dev.displayName,
        "",   // platform defaults to the backend name
        rt->abiVersion,
        details,
        -1,
        idx,
    });
    currentDeviceScope = &deviceScope;

    // LiteRT's accelerator delegates narrate on stderr from their own worker
    // threads, asynchronously -- weight-upload and kernel-compile threads
    // print between and after the calls that spawned them, so no per-call mute
    // can bracket it, and the desktop wheels export no sink logger to divert
    // it either.  One stderr-only scope over the device's whole benchmark
    // silences that noise (the result table is on stdout and stays visible)
    // and watches for the phrases a lost accelerator prints.  A WebGPU device
    // that exhausts Vulkan's file descriptors is the reason this matters: it
    // is lost, keeps returning success from invokes that compute nothing, and
    // would otherwise publish an impossible rate under a clean-looking table.
    static const std::vector<std::string> kLossMarkers = {
        "is lost", "Failed to invoke", "failed to invoke",
        "OUT_OF_HOST_MEMORY", "Ran out of file descriptors",
    };
    bool deviceLost = false;
    {
      // Mute every device's scope, not just the accelerators': a lost GPU's
      // Dawn worker threads keep logging asynchronously after its own tests
      // end, into the next device's (the CPU's) scope, so leaving the CPU
      // unmuted lets that spill through.  Only an accelerator's own scope
      // watches for the loss markers, though -- a marker that leaks into the
      // CPU's scope is the GPU's dying breath, not a CPU fault.
      clpeak::ScopedConsoleMute devMute(
          clpeak::ScopedConsoleMute::Capture::Verbose,
          dev.accel == LitertAccel::Cpu ? std::vector<std::string>{} : kLossMarkers,
          /*stderrOnly=*/true);

      // Run one benchmark, then let a device that has just announced its loss
      // stop the rest: every later test on it would only compile models it
      // cannot run and reprint the same failure.
      auto phase = [&](Benchmark b, int (LitertPeak::*fn)(const LitertRuntime &,
                                                          const litert_device_info_t &,
                                                          benchmark_config_t &)) {
        if (deviceLost || clpeak::cancelRequested() || !isAllowed(b))
          return;
        (this->*fn)(*rt, dev, cfg);
        if (devMute.sawWatched())
          deviceLost = true;
      };

      // ---- Compute (FLOPS + OPS) -------------------------------------------
      phase(Benchmark::Gemm, &LitertPeak::runGemm);
      phase(Benchmark::Conv, &LitertPeak::runConv);
      // ---- What the speed rows cost in accuracy ----------------------------
      phase(Benchmark::NumericError, &LitertPeak::runNumericError);
      // ---- AI composite (whole transformer block) --------------------------
      phase(Benchmark::TransformerBlock, &LitertPeak::runBlock);
      // ---- Bandwidth -------------------------------------------------------
      phase(Benchmark::Activation, &LitertPeak::runActivation);
      phase(Benchmark::TensorBW, &LitertPeak::runTensorBandwidth);
      phase(Benchmark::TransferBW, &LitertPeak::runTransferBandwidth);
      // ---- Latency ---------------------------------------------------------
      phase(Benchmark::KernelLatency, &LitertPeak::runDispatchLatency);
    }

    // The accelerator's environment goes with its device.  Nothing needs it
    // once the device's tests are done -- a later run rebuilds it -- and
    // while it lives it holds the accelerator: the GPU's memory, the Dawn or
    // OpenCL device, and on Linux the file descriptors those keep.  A WebGPU
    // device that died by exhausting the process's descriptors ("Ran out of
    // file descriptors", VK_ERROR_OUT_OF_HOST_MEMORY) had, when its
    // environment lived on, taken every ONNX GPU provider that ran after it
    // in the same process down with it.  Tearing the environment down is
    // the one thing that can hand any of that back, so it happens here for
    // a lost device and a healthy one alike, and the descriptor count says
    // how much came back.
    unsigned long fdsBefore = 0, fdsAfter = 0, fdLimit = 0;
    const bool fdsKnown = clpeak::openFileDescriptors(fdsBefore, fdLimit);
    {
      clpeak::ScopedConsoleMute mute;
      litertResetEnvironment(*rt, dev.accel);
    }
    if (fdsKnown)
    {
      (void)clpeak::openFileDescriptors(fdsAfter, fdLimit);
      CLPEAK_VLOG("litert[%s]: environment released; %lu open file descriptors, %lu before"
                  "%s\n",
                  dev.displayName.c_str(), fdsAfter, fdsBefore,
                  fdLimit ? (" (limit " + std::to_string(fdLimit) + ")").c_str() : "");
    }

    // Emitted after the mute closes, so it reaches the console: a device lost
    // mid-run leaves any rates already printed for it untrustworthy, and this
    // says so rather than letting a clean table imply they are sound.
    if (deviceLost)
    {
      std::string fds;
      if (fdsKnown)
        fds = "; the process holds " + std::to_string(fdsAfter) + " open file descriptors" +
              (fdLimit ? " of a limit of " + std::to_string(fdLimit) : std::string()) +
              " after releasing its environment (" + std::to_string(fdsBefore) +
              " before), which is what every later backend in this process has left";
      CLPEAK_LOG(Warning,
                 "LiteRT %s: the accelerator reported it was lost during the run "
                 "(a Vulkan/WebGPU file-descriptor or host-memory exhaustion on "
                 "this platform); any rates shown for it are unreliable and its "
                 "remaining tests were skipped%s",
                 dev.displayName.c_str(), fds.c_str());
    }

    currentDeviceScope = nullptr;
  }

  return 0;
}

BackendInventory LitertPeak::enumerate()
{
  BackendInventory inv;
  inv.id = kBackend;

  const LitertRuntime *rt = litertRuntime();
  if (!rt)
  {
    const std::string why = litertLoadDiagnostic();
    inv.unavailableReason = why.empty() ? "LiteRT library (libLiteRt) not found" : why;
    return inv;
  }
  inv.info = "LiteRT ABI " + rt->abiVersion;
  inv.available = true;

  std::vector<std::pair<litert_device_info_t, std::string>> skipped;
  auto devs = litertUsableDevices(*rt, &skipped);
  for (const auto &sk : skipped)
    inv.notes.push_back(sk.first.displayName + ": " + sk.second);

  InventoryPlatform plat;
  plat.index = 0;
  plat.name = "LiteRT";
  for (int i = 0; i < (int)devs.size(); i++)
  {
    InventoryDevice d;
    d.index = i;
    d.name = devs[i].displayName;
    d.typeStr = devs[i].typeStr;
    plat.devices.push_back(std::move(d));
  }
  inv.platforms.push_back(std::move(plat));
  return inv;
}

#endif // ENABLE_LITERT
