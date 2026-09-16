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
#include <filesystem>
#include <mutex>
#include <ostream>

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

const NpuVendor *findNpuVendor(const std::string &dir)
{
  if (dir.empty())
    return nullptr;
  std::error_code ec;
  std::filesystem::directory_iterator it(dir, ec), end;
  for (; !ec && it != end; it.increment(ec))
  {
    const std::string name = it->path().filename().string();
    if (name.rfind("libLiteRtDispatch_", 0) != 0 && name.rfind("LiteRtDispatch", 0) != 0)
      continue;
    for (const auto &v : kNpuVendors)
      if (name.find(v.fileTag) != std::string::npos)
        return &v;
  }
  return nullptr;
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
    {
      litert_device_info_t dev;
      dev.accel = LitertAccel::Npu;
      dev.typeStr = "NPU";
      dev.deviceType = DeviceType::Accelerator;
      const NpuVendor *v = findNpuVendor(litertNpuDir());
      if (!v)
      {
        dev.displayName = "NPU via LiteRT";
        g_skipped.emplace_back(dev, "no libLiteRtDispatch_<vendor> library in " +
                                        (litertNpuDir().empty() ? std::string("the runtime's directory")
                                                                : litertNpuDir()) +
                                        "; --litert-npu-dir names where the vendor runtime is");
      }
      else
      {
        dev.vendor = v->vendor;
        dev.displayName = v->display;
        std::string reason, log;
        if (probeAccel(rt, dev, reason, log))
          g_devs.push_back(dev);
        else
          g_skipped.emplace_back(dev, reason);
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
        {"LiteRT", "ABI " + rt->abiVersion + (rt->path.empty() ? "" : " (" + rt->path + ")")},
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

    // Emitted after the mute closes, so it reaches the console: a device lost
    // mid-run leaves any rates already printed for it untrustworthy, and this
    // says so rather than letting a clean table imply they are sound.
    if (deviceLost)
      CLPEAK_LOG(Warning,
                 "LiteRT %s: the accelerator reported it was lost during the run "
                 "(a Vulkan/WebGPU file-descriptor or host-memory exhaustion on "
                 "this platform); any rates shown for it are unreliable and its "
                 "remaining tests were skipped",
                 dev.displayName.c_str());

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
  inv.info = "LiteRT ABI " + rt->abiVersion + (rt->path.empty() ? "" : ", " + rt->path);
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
