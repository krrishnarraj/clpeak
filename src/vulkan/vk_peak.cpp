#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/options.h>
#include <common/inventory.h>
#include <common/common.h>
#include <algorithm>
#include <tuple>
#include <cstring>
#include <sstream>
#include <utility>
#include <chrono>
#include <cfloat>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

#ifdef VK_HAS_ANY_COOPMAT
// Short name for a cooperative-matrix component type, for the --verbose dump
// of what the driver advertised.  Only the types the coopmat shaders can use
// are named; anything else prints its raw enum value.
static const char *coopmatComponentName(VkComponentTypeKHR t)
{
  switch (t)
  {
  case VK_COMPONENT_TYPE_FLOAT16_KHR:     return "fp16";
  case VK_COMPONENT_TYPE_FLOAT32_KHR:     return "fp32";
  case VK_COMPONENT_TYPE_FLOAT64_KHR:     return "fp64";
  case VK_COMPONENT_TYPE_BFLOAT16_KHR:    return "bf16";
  case VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT: return "fp8e4m3";
  case VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT: return "fp8e5m2";
  case VK_COMPONENT_TYPE_SINT8_KHR:       return "sint8";
  case VK_COMPONENT_TYPE_SINT16_KHR:      return "sint16";
  case VK_COMPONENT_TYPE_SINT32_KHR:      return "sint32";
  case VK_COMPONENT_TYPE_UINT8_KHR:       return "uint8";
  case VK_COMPONENT_TYPE_UINT16_KHR:      return "uint16";
  case VK_COMPONENT_TYPE_UINT32_KHR:      return "uint32";
  default:                                return "other";
  }
}

// One line per dtype for the --verbose dump: which tile pickTile settled on,
// and the subgroup width it will actually run at.  "advertised at N" means the
// device named that width for this tile; "width unknown" means only the
// width-agnostic KHR query was available and the default is a guess.
static void logPickedTile(const char *label, const coopmat_tile_t &t,
                          const vk_device_info_t &info)
{
  if (!t.supported)
  {
    CLPEAK_VLOG("  picked %-8s (none)\n", label);
    return;
  }
  CLPEAK_VLOG("  picked %-8s %ux%ux%u, running subgroup %u (%s)\n", label,
              t.M, t.N, t.K,
              coopmatSubgroupWidth(info, t.subgroupSize),
              t.subgroupSize ? "advertised at this width" : "width unknown");
}
#endif

// ---------------------------------------------------------------------------
// vkPeak
// ---------------------------------------------------------------------------

vkPeak::vkPeak()
  : instance(VK_NULL_HANDLE)
{
}

vkPeak::~vkPeak()
{
  cleanup();
}

void vkPeak::applyOptions(const CliOptions &opts)
{
    Peak::applyOptions(opts);
    deviceIndices = opts.vkDeviceIndices;
}

bool vkPeak::initInstance()
{
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "clpeak-vulkan";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "clpeak";
  // Only request 1.1 when we compiled in code that actually needs 1.1-core
  // symbols (e.g. vkGetPhysicalDeviceFeatures2 for INT8 DP feature query).
  // Otherwise request 1.0 so we work on older drivers / Android API levels
  // where libvulkan only exposes 1.0.
  // 1.1 gets us vkGetPhysicalDeviceFeatures2; cooperative matrix brings its
  // own extension + VK_KHR_vulkan_memory_model, so 1.1 is sufficient as the
  // instance version (MoltenVK 1.2 headers reject some older loaders).
#if defined(VK_HAS_COMPUTE_INT8_DP_V1) || defined(VK_HAS_COMPUTE_MP_V1) || defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_ANY_COOPMAT)
  appInfo.apiVersion = VK_API_VERSION_1_1;
#else
  appInfo.apiVersion = VK_API_VERSION_1_0;
#endif

  VkInstanceCreateInfo instCI = {};
  instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instCI.pApplicationInfo = &appInfo;

#if defined(__APPLE__) || defined(__MACOSX)
  // MoltenVK portability.  Newer Apple SDK loaders require applications to
  // opt in before non-conformant portability drivers are enumerated.
  std::vector<const char *> extensions;
  auto hasInstanceExt = [](const char *name) {
    uint32_t extCount = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr) != VK_SUCCESS)
      return false;
    std::vector<VkExtensionProperties> props(extCount);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &extCount, props.data()) != VK_SUCCESS)
      return false;
    for (const auto &prop : props)
      if (strcmp(prop.extensionName, name) == 0)
        return true;
    return false;
  };
  if (hasInstanceExt(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
  {
    instCI.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
  }
  if (hasInstanceExt(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
  instCI.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  instCI.ppEnabledExtensionNames = extensions.empty() ? nullptr : extensions.data();
#endif

  m_instanceResult = vkCreateInstance(&instCI, nullptr, &instance);
  if (m_instanceResult != VK_SUCCESS)
    return false;

  uint32_t devCount = 0;
  vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
  if (devCount > 0)
  {
    physicalDevices.resize(devCount);
    vkEnumeratePhysicalDevices(instance, &devCount, physicalDevices.data());
  }

  return true;
}

void vkPeak::cleanup()
{
  if (instance != VK_NULL_HANDLE)
  {
    vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
  }
}

float vkPeak::runKernel(VulkanDevice &dev, VkPipeline pipeline,
                        VkPipelineLayout pipeLayout,
                        VkDescriptorSet descriptorSet,
                        uint32_t groupCountX,
                        unsigned int targetTimeUsLocal, unsigned int forcedIters,
                        bool serialize,
                        const void *pushData, uint32_t pushSize)
{
  // Allocate command buffer
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = dev.commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer cmdBuf;
  vkAllocateCommandBuffers(dev.device, &allocInfo, &cmdBuf);

  // Record `n` dispatches into cmdBuf, submit once, return total wall-time
  // in us (or -1 on submit failure).
  auto runBatch = [&](unsigned int n) -> float {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descriptorSet, 0, nullptr);
    if (pushData && pushSize > 0)
      vkCmdPushConstants(cmdBuf, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    // Vulkan lets consecutive vkCmdDispatch calls with no declared dependency
    // overlap, and NVIDIA does overlap them.  For the bandwidth tests that puts
    // two passes over the same buffer or image in flight at once, walking it in
    // the same order, so the trailing pass reads from cache rather than memory
    // and the "every element is read exactly once" premise the byte count rests
    // on no longer holds.  CUDA and OpenCL cannot hit this -- one stream and an
    // in-order queue both serialise -- which is a large part of why Vulkan is
    // the odd one out on image bandwidth.  Barrier between dispatches where the
    // caller asks for it; see the runKernel comment in vk_peak.h.
    VkMemoryBarrier memBarrier = {};
    memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    for (unsigned int i = 0; i < n; i++)
    {
      if (serialize && i > 0)
        vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                             1, &memBarrier, 0, nullptr, 0, nullptr);
      vkCmdDispatch(cmdBuf, groupCountX, 1, 1);
    }
    vkEndCommandBuffer(cmdBuf);

    auto start = std::chrono::high_resolution_clock::now();
    VkResult submitRes = dev.submitAndWait(cmdBuf);
    auto end = std::chrono::high_resolution_clock::now();
    vkResetCommandBuffer(cmdBuf, 0);
    if (submitRes != VK_SUCCESS) return -1.0f;
    return (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  };

  // Phase 1: untimed warmup (cache + clock ramp).  One dispatch per submit
  // so the first dispatch's compile/load cost amortizes naturally.
  for (unsigned int w = 0; w < warmupCount; w++)
  {
    if (runBatch(1) < 0.0f)
    {
      vkFreeCommandBuffers(dev.device, dev.commandPool, 1, &cmdBuf);
      return -1.0f;
    }
  }

  // Phase 2: timed calibration probe. Keep this to one dispatch so warmupCount
  // does not force a multi-dispatch submit on slow kernels.
  unsigned int probeIters = 1;
  float probeUs = runBatch(probeIters);
  if (probeUs < 0.0f)
  {
    vkFreeCommandBuffers(dev.device, dev.commandPool, 1, &cmdBuf);
    return -1.0f;
  }
  if (probeUs <= 0.0f) probeUs = 1.0f; // floor for very fast paths
  double per_iter_us = (double)probeUs / (double)probeIters;

  // Phase 3: real timed run with calibrated iter count.
  unsigned int iters = pickIters(per_iter_us, targetTimeUsLocal, forcedIters);
  float timed = runBatch(iters);

  vkFreeCommandBuffers(dev.device, dev.commandPool, 1, &cmdBuf);

  // Driver/device failure (e.g. Adreno+Turnip losing the device on a
  // shader that uses an advertised-but-unsupported feature): vkQueueWaitIdle
  // returns instantly with VK_ERROR_DEVICE_LOST, leaving timed ~= 0 which
  // would otherwise propagate as "inf" GFLOPS/GBPS into the report.
  if (timed < 0.0f)
    return -1.0f;
  if (timed <= 0.0f)
    return -1.0f;
  return timed / static_cast<float>(iters);
}

int vkPeak::runAll()
{
  auto backendScope = log->beginBackend("Vulkan");
  if (!initInstance())
  {
    if (m_instanceResult == VK_ERROR_INCOMPATIBLE_DRIVER)
    {
      log->note("Vulkan: no compatible driver found\n");
      return 0;
    }
    log->note("Vulkan: failed to create instance (VkResult " +
              std::to_string(m_instanceResult) + ")\n");
    return -1;
  }
  if (physicalDevices.empty())
  {
    log->note("Vulkan: no devices found\n");
    return 0;
  }

  for (size_t d = 0; d < physicalDevices.size(); d++)
  {
    if (clpeak::cancelRequested())
      break;
    if (!deviceIndices.empty() &&
        std::find(deviceIndices.begin(), deviceIndices.end(), static_cast<int>(d)) == deviceIndices.end())
      continue;

    VulkanDevice dev;
    if (!dev.init(instance, physicalDevices[d]))
    {
      log->note("Vulkan: failed to init device " + std::to_string(d) + "\n");
      continue;
    }

#ifdef VK_HAS_ANY_COOPMAT
    // Enumerate cooperative-matrix properties to decide which tile to run for
    // each dtype, and at which subgroup width.  Done here (not in
    // VulkanDevice::init) because the entry points are instance-level extension
    // functions that must be resolved via vkGetInstanceProcAddr against the
    // real VkInstance.
    //
    // Two queries, and the difference matters.  The KHR one is width-agnostic:
    // it returns the union of tiles over every subgroup width the device
    // supports, so a tile it lists is not necessarily valid at the width we
    // pin.  VK_EXT_cooperative_matrix_maintenance1 takes a subgroup size and
    // answers for that width alone, so asking it per width tells us which tile
    // goes with which width.  Prefer it; fall back to KHR when it is absent,
    // where the best we can do is the device's native width.
    if (dev.info.cooperativeMatrixSupported)
    {
      // One advertised (dtype, shape, width) combination.  width == 0 means it
      // came from the width-agnostic KHR query.
      struct CoopCand {
        uint32_t M, N, K, width;
        VkComponentTypeKHR a, b, c, d;
      };
      std::vector<CoopCand> cands;

#ifdef VK_EXT_cooperative_matrix_maintenance1
      if (dev.info.coopmatMaintenance1Supported)
      {
        auto pfn2 = (PFN_vkGetPhysicalDeviceCooperativeMatrixProperties2EXT)
            vkGetInstanceProcAddr(instance,
                "vkGetPhysicalDeviceCooperativeMatrixProperties2EXT");
        const uint32_t lo = dev.info.minSubgroupSize ? dev.info.minSubgroupSize
                                                     : dev.info.subgroupSize;
        const uint32_t hi = dev.info.maxSubgroupSize ? dev.info.maxSubgroupSize
                                                     : dev.info.subgroupSize;
        for (uint32_t sg = lo; pfn2 && sg && sg <= hi; sg *= 2)
        {
          VkPhysicalDeviceCooperativeMatrixInfo2EXT info2 = {};
          info2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_INFO_2_EXT;
          info2.scope = VK_SCOPE_SUBGROUP_KHR;
          info2.subgroupSize = sg;
          uint32_t n2 = 0;
          if (pfn2(physicalDevices[d], &info2, &n2, nullptr) != VK_SUCCESS)
          {
            CLPEAK_VLOG("  at subgroup %u: query failed\n", sg);
            continue;
          }
          std::vector<VkCooperativeMatrixProperties2EXT> p2(n2);
          for (auto &q : p2) q.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_2_EXT;
          if (n2 && pfn2(physicalDevices[d], &info2, &n2, p2.data()) != VK_SUCCESS)
            continue;
          CLPEAK_VLOG("  at subgroup %u: %u tile(s)\n", sg, n2);
          for (auto &q : p2)
          {
            CLPEAK_VLOG("    %ux%ux%u  A=%s B=%s C=%s D=%s\n",
                        q.MGranularity, q.NGranularity, q.KGranularity,
                        coopmatComponentName(q.AType), coopmatComponentName(q.BType),
                        coopmatComponentName(q.CType), coopmatComponentName(q.ResultType));
            cands.push_back({q.MGranularity, q.NGranularity, q.KGranularity, sg,
                             q.AType, q.BType, q.CType, q.ResultType});
          }
        }
      }
#endif

      if (cands.empty())
      {
        CLPEAK_VLOG("  no per-subgroup-width tile list (%s); falling back to the "
                    "width-agnostic KHR query\n",
                    dev.info.coopmatMaintenance1Supported
                        ? "query returned nothing"
                        : "VK_EXT_cooperative_matrix_maintenance1 not available");
        auto pfn = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
        uint32_t propCount = 0;
        if (pfn && pfn(physicalDevices[d], &propCount, nullptr) == VK_SUCCESS && propCount > 0)
        {
          std::vector<VkCooperativeMatrixPropertiesKHR> props(propCount);
          for (auto &p : props) p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
          if (pfn(physicalDevices[d], &propCount, props.data()) == VK_SUCCESS)
          {
            CLPEAK_VLOG("Vulkan: device %zu advertises %u cooperative-matrix "
                        "properties\n", d, propCount);
            for (auto &p : props)
            {
              CLPEAK_VLOG("  %ux%ux%u  A=%s B=%s C=%s D=%s  scope=%s%s\n",
                          p.MSize, p.NSize, p.KSize,
                          coopmatComponentName(p.AType), coopmatComponentName(p.BType),
                          coopmatComponentName(p.CType), coopmatComponentName(p.ResultType),
                          p.scope == VK_SCOPE_SUBGROUP_KHR  ? "subgroup" :
                          p.scope == VK_SCOPE_WORKGROUP_KHR ? "workgroup" : "other",
                          p.saturatingAccumulation ? "  saturating (unusable)" : "");
              if (p.scope != VK_SCOPE_SUBGROUP_KHR) continue;
              // The shaders emit OpCooperativeMatrixMulAddKHR *without* the
              // SaturatingAccumulationKHR operand, and that operand is required
              // if and only if the matched property has saturatingAccumulation
              // set (VUID-RuntimeSpirv-OpCooperativeMatrixMulAddKHR-10060).  A
              // saturating property describes a shape these shaders may not run.
              if (p.saturatingAccumulation) continue;
              cands.push_back({p.MSize, p.NSize, p.KSize, 0,
                               p.AType, p.BType, p.CType, p.ResultType});
            }
          }
        }
      }

      // Pick one tile per input/accumulator dtype.  We don't assume a shape --
      // among the advertised tiles prefer a square 16x16 face (fewer, larger
      // MMAs per output element), then the largest volume (M*N*K) to minimize
      // loop overhead.  Whatever K the hardware wants (16 for fp16/bf16, 32 for
      // 8-bit types, ...) flows through to the shader as a specialization
      // constant, and so does the width.
      //
      // Among widths for the same shape, prefer the device's native subgroup
      // size, then the widest available: native is what every device measured
      // so far wants, and falling to a narrower one is what lets a tile run at
      // all when the driver does not offer it at the native width.
      auto pickTile = [&](VkComponentTypeKHR ab, VkComponentTypeKHR acc) {
        coopmat_tile_t best;
        auto rank = [&](const coopmat_tile_t &t) {
          bool square16 = (t.M == 16 && t.N == 16);
          bool native   = (t.subgroupSize == dev.info.subgroupSize);
          return std::make_tuple(square16 ? 0 : 1,
                                 -(int64_t)t.M * (int64_t)t.N * (int64_t)t.K,
                                 native ? 0 : 1,
                                 -(int64_t)t.subgroupSize);
        };
        for (auto &p : cands)
        {
          if (p.a != ab || p.b != ab) continue;
          if (p.c != acc || p.d != acc) continue;
          coopmat_tile_t cand{true, p.M, p.N, p.K, p.width};
          if (!best.supported || rank(cand) < rank(best)) best = cand;
        }
        return best;
      };

      dev.info.coopmatFP32    = pickTile(VK_COMPONENT_TYPE_FLOAT32_KHR,     VK_COMPONENT_TYPE_FLOAT32_KHR);
      dev.info.coopmatFP16    = pickTile(VK_COMPONENT_TYPE_FLOAT16_KHR,     VK_COMPONENT_TYPE_FLOAT32_KHR);
      dev.info.coopmatFP16F16 = pickTile(VK_COMPONENT_TYPE_FLOAT16_KHR,     VK_COMPONENT_TYPE_FLOAT16_KHR);
      dev.info.coopmatBF16    = pickTile(VK_COMPONENT_TYPE_BFLOAT16_KHR,    VK_COMPONENT_TYPE_FLOAT32_KHR);
      dev.info.coopmatFP8E4M3 = pickTile(VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT, VK_COMPONENT_TYPE_FLOAT32_KHR);
      dev.info.coopmatFP8E5M2 = pickTile(VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT, VK_COMPONENT_TYPE_FLOAT32_KHR);
      dev.info.coopmatINT8    = pickTile(VK_COMPONENT_TYPE_SINT8_KHR,       VK_COMPONENT_TYPE_SINT32_KHR);

      logPickedTile("fp32",    dev.info.coopmatFP32,    dev.info);
      logPickedTile("fp16",    dev.info.coopmatFP16,    dev.info);
      logPickedTile("fp16+f16", dev.info.coopmatFP16F16, dev.info);
      logPickedTile("bf16",    dev.info.coopmatBF16,    dev.info);
      logPickedTile("fp8e4m3", dev.info.coopmatFP8E4M3, dev.info);
      logPickedTile("fp8e5m2", dev.info.coopmatFP8E5M2, dev.info);
      logPickedTile("int8",    dev.info.coopmatINT8,    dev.info);

      CLPEAK_VLOG("  subgroup: device reports %u, range %u..%u, size control %s\n",
                  dev.info.subgroupSize, dev.info.minSubgroupSize,
                  dev.info.maxSubgroupSize,
                  dev.info.subgroupSizeControl ? "yes" : "no");
    }
#endif

    // Vulkan has no cache-size query at all, so the device-local heap is the
    // only handle on the cache the working set has to outgrow -- and only on a
    // discrete part, where that heap is really the board's own memory.  See
    // benchmark_config_t::forDevice.
    benchmark_config_t cfg = benchmark_config_t::forDevice(
            (dev.info.vkDeviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) ? DeviceType::Cpu : DeviceType::Gpu,
            /*lastLevelCacheBytes=*/0,
            (dev.info.vkDeviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? dev.info.heapSize : 0);
    cfg.targetTimeUs = targetTimeUs;
    if (forceIters)
      cfg.kernelLatencyIters = specifiedIters;

    std::vector<logger::Prop> deviceProps;
    deviceProps.push_back({"API version", dev.info.apiVersion});
    deviceProps.push_back({"VRAM", std::to_string(dev.info.heapSize / (1024*1024)) + " MB"});
    if (dev.info.numCUs > 0)
      deviceProps.push_back({"Compute units", std::to_string(dev.info.numCUs)});

    auto deviceScope = backendScope.beginDevice({
      dev.info.deviceName,
      "",   // platform defaults to "Vulkan"
      dev.info.driverVersion,
      deviceProps,
      -1,
      static_cast<int>(d)
    });
    currentDeviceScope = &deviceScope;

    // ---- Phase 1: floating-point compute (GFLOPS / TFLOPS) ---------
    if (isAllowed(Benchmark::ComputeSP))       runComputeSP(dev, cfg);
#ifdef VK_HAS_COMPUTE_HP_V1
    if (isAllowed(Benchmark::ComputeHP))       runComputeHP(dev, cfg);
#endif
#ifdef VK_HAS_COMPUTE_DP_V1
    if (isAllowed(Benchmark::ComputeDP))       runComputeDP(dev, cfg);
#endif
#ifdef VK_HAS_COMPUTE_MP_V1
    if (isAllowed(Benchmark::ComputeMP))       runComputeMP(dev, cfg);
#endif
#ifdef VK_HAS_COMPUTE_BF16_V1
    if (isAllowed(Benchmark::ComputeBF16))     runComputeBF16(dev, cfg);
#endif
#ifdef VK_HAS_ANY_COOPMAT
    if (isAllowedAs(Benchmark::CoopMatrix, Category::FpCompute))
        runCoopMatrix(dev, cfg, /*intPart=*/false);
#endif
    // ---- Phase 2: integer compute (GOPS / TOPS) --------------------
#ifdef VK_HAS_COMPUTE_INT32_V1
    if (isAllowed(Benchmark::ComputeInt))        runComputeInt32(dev, cfg);
#endif
#ifdef VK_HAS_COMPUTE_INT8_DP_V1
    if (isAllowed(Benchmark::ComputeInt8DP))     runComputeInt8DP(dev, cfg);
#endif
#ifdef VK_HAS_ANY_COOPMAT
    if (isAllowedAs(Benchmark::CoopMatrix, Category::IntCompute))
        runCoopMatrix(dev, cfg, /*intPart=*/true);
#endif
    // ---- Phase 3: bandwidth (GBPS) ---------------------------------
    if (isAllowed(Benchmark::GlobalBW))        runGlobalBandwidth(dev, cfg);
    if (isAllowed(Benchmark::LocalBW))         runLocalBandwidth(dev, cfg);
    if (isAllowed(Benchmark::ImageBW))         runImageBandwidth(dev, cfg);
    if (isAllowed(Benchmark::TransferBW))      runTransferBandwidth(dev, cfg);

    // ---- Phase 4: latency (us) -------------------------------------
    if (isAllowed(Benchmark::KernelLatency))   runKernelLatency(dev, cfg);

    currentDeviceScope = nullptr;
    // deviceScope auto-closes here
  }

  // backendScope auto-closes here
  return 0;
}

// ---------------------------------------------------------------------------
// Benchmark methods live in separate files:
//   vulkan_device.cpp     compute_kernel.cpp
//   compute_float.cpp     compute_int.cpp       coopmat.cpp
//   global_bandwidth.cpp  local_bandwidth.cpp   image_bandwidth.cpp
//   transfer_bandwidth.cpp kernel_latency.cpp
// ---------------------------------------------------------------------------

// Free-function enumeration used by --list-devices (desktop) and the Android
// JNI surface. Spins up a throwaway VkInstance just long enough to read
// physical-device properties, then tears it down.
BackendInventory vkPeak::enumerate()
{
  BackendInventory inv;
  inv.backend = "Vulkan";

  vkPeak vk;
  if (!vk.initInstance())
    return inv;  // available stays false

  inv.available = !vk.physicalDevices.empty();

  InventoryPlatform plat;
  plat.index = 0;
  plat.name  = "Vulkan";

  for (size_t i = 0; i < vk.physicalDevices.size(); i++)
  {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(vk.physicalDevices[i], &props);

    InventoryDevice dev;
    dev.index   = static_cast<int>(i);
    dev.name    = props.deviceName;
    dev.typeStr = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   ? "Discrete GPU"
                : (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? "Integrated GPU"
                : (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)            ? "CPU"
                                                                               : "Other";
    {
      std::ostringstream api;
      api << VK_VERSION_MAJOR(props.apiVersion) << "."
          << VK_VERSION_MINOR(props.apiVersion) << "."
          << VK_VERSION_PATCH(props.apiVersion);
      dev.apiVersion = api.str();
    }
    plat.devices.push_back(std::move(dev));
  }

  inv.platforms.push_back(std::move(plat));
  return inv;
}

void vkPeak::printInventory(const BackendInventory &b, std::ostream &os)
{
    os << "\n=== Vulkan backend ===\n";
    if (!b.available)
    {
        os << "Vulkan: failed to create instance or no devices found\n";
        return;
    }
    for (const auto &plat : b.platforms)
        for (const auto &d : plat.devices)
        {
            os << "  Vulkan Device " << d.index << ": " << d.name;
            if (!d.typeStr.empty())
                os << " [" << d.typeStr << "]";
            os << "\n";
            if (!d.apiVersion.empty())
                os << "    API       : " << d.apiVersion << "\n";
        }
}

#endif // ENABLE_VULKAN
