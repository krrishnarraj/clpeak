#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

// ---------------------------------------------------------------------------
// Kernel launch latency (Vulkan) -- two metrics:
//
//   dispatch  : one-way host-submit -> GPU-kernel-start.  Requires
//               VK_EXT_calibrated_timestamps to map host time into the GPU
//               clock domain.  Skipped (with note) when the extension is
//               unavailable.
//   roundtrip : std::chrono around vkQueueSubmit + vkQueueWaitIdle.  Always
//               reported, directly comparable to the same metric on
//               OpenCL/CUDA/Metal.
// ---------------------------------------------------------------------------

int vkPeak::runKernelLatency(VulkanDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = forceIters ? specifiedIters
                                  : (cfg.kernelLatencyIters ? cfg.kernelLatencyIters : 1000);

  logger::TestSpec testSpec;
  testSpec.tag = "kernel_launch_latency";
  testSpec.display = "Kernel launch latency";
  testSpec.unit = "us";
  auto test = currentDeviceScope->beginTest(testSpec);

  // Pipeline layout with no descriptor sets and no push constants.
  VkPipelineLayoutCreateInfo plCI = {};
  plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
  if (vkCreatePipelineLayout(dev.device, &plCI, nullptr, &pipeLayout) != VK_SUCCESS)
  {
    CLPEAK_VLOG("pipeline layout creation failed\n");
    return -1;
  }

  // Shader module from the embedded no-op SPIR-V.
  VkShaderModuleCreateInfo smCI = {};
  smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smCI.codeSize = vk_shaders::kernel_latency_size;
  smCI.pCode    = vk_shaders::kernel_latency;
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  if (vkCreateShaderModule(dev.device, &smCI, nullptr, &shaderModule) != VK_SUCCESS)
  {
    vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
    CLPEAK_VLOG("shader module creation failed\n");
    return -1;
  }

  VkComputePipelineCreateInfo pCI = {};
  pCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pCI.layout = pipeLayout;
  pCI.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  pCI.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  pCI.stage.module = shaderModule;
  pCI.stage.pName  = "main";
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (vkCreateComputePipelines(dev.device, VK_NULL_HANDLE, 1, &pCI, nullptr, &pipeline) != VK_SUCCESS)
  {
    vkDestroyShaderModule(dev.device, shaderModule, nullptr);
    vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
    CLPEAK_VLOG("compute pipeline creation failed\n");
    return -1;
  }

  // GPU timestamp query pool used by the dispatch-latency path: a single
  // TOP_OF_PIPE timestamp tells us when the GPU started executing.
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(dev.physicalDevice, &props);
  float timestampPeriodNs = props.limits.timestampPeriod;
  bool haveTimestamps = (timestampPeriodNs > 0.0f) && dev.info.calibratedTimestampsSupported;

  // Resolve VK_EXT_calibrated_timestamps function pointers.  These let us
  // capture a (host_t, gpu_t) pair "at the same instant" so we can convert
  // GPU timestamps into the host clock domain and measure exact one-way
  // dispatch latency, the same way OpenCL profiling reports QUEUED -> START.
  PFN_vkGetCalibratedTimestampsEXT pfnGetCalTs = nullptr;
  VkTimeDomainEXT hostDomain = VK_TIME_DOMAIN_DEVICE_EXT; // sentinel: unset
  if (haveTimestamps)
  {
    pfnGetCalTs = (PFN_vkGetCalibratedTimestampsEXT)
        vkGetDeviceProcAddr(dev.device, "vkGetCalibratedTimestampsEXT");
    auto pfnGetDomains = (PFN_vkGetPhysicalDeviceCalibrateableTimeDomainsEXT)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT");
    if (pfnGetCalTs && pfnGetDomains)
    {
      uint32_t n = 0;
      pfnGetDomains(dev.physicalDevice, &n, nullptr);
      std::vector<VkTimeDomainEXT> domains(n);
      pfnGetDomains(dev.physicalDevice, &n, domains.data());
      // Prefer a host monotonic clock so we can read it cheaply with
      // std::chrono::steady_clock (CLOCK_MONOTONIC on POSIX, QPC on Windows).
      VkTimeDomainEXT preferred[] = {
#if defined(_WIN32)
        VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT,
#else
        VK_TIME_DOMAIN_CLOCK_MONOTONIC_EXT,
        VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT,
#endif
      };
      for (auto p : preferred)
      {
        for (auto d : domains) if (d == p) { hostDomain = p; break; }
        if (hostDomain != VK_TIME_DOMAIN_DEVICE_EXT) break;
      }
    }
    if (hostDomain == VK_TIME_DOMAIN_DEVICE_EXT)
      haveTimestamps = false; // can't pin host<->device clocks; skip dispatch
  }

  VkQueryPool queryPool = VK_NULL_HANDLE;
  if (haveTimestamps)
  {
    VkQueryPoolCreateInfo qpCI = {};
    qpCI.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpCI.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpCI.queryCount = 1;
    if (vkCreateQueryPool(dev.device, &qpCI, nullptr, &queryPool) != VK_SUCCESS)
      haveTimestamps = false;
  }

  // Pre-record a 1x1x1 dispatch.  Reusing one command buffer across iters
  // keeps record overhead out of the timing loop.
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = dev.commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
  if (vkAllocateCommandBuffers(dev.device, &allocInfo, &cmdBuf) != VK_SUCCESS)
  {
    if (queryPool != VK_NULL_HANDLE)
      vkDestroyQueryPool(dev.device, queryPool, nullptr);
    vkDestroyPipeline(dev.device, pipeline, nullptr);
    vkDestroyShaderModule(dev.device, shaderModule, nullptr);
    vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
    CLPEAK_VLOG("command buffer allocation failed\n");
    return -1;
  }

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  vkBeginCommandBuffer(cmdBuf, &beginInfo);
  if (haveTimestamps)
  {
    vkCmdResetQueryPool(cmdBuf, queryPool, 0, 1);
    vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
  }
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdDispatch(cmdBuf, 1, 1, 1);
  vkEndCommandBuffer(cmdBuf);

  // Get a calibrated (host, gpu) pair.  Stored in driver-domain units:
  //   ns since CLOCK_MONOTONIC epoch (POSIX) or QPC ticks (Windows).
  // We re-read the matching host timestamp in the SAME domain inside the
  // hot loop, so the units always match across the subtraction.
  auto getCalibrated = [&](uint64_t &hostOut, uint64_t &gpuOut) -> bool {
    VkCalibratedTimestampInfoEXT infos[2] = {};
    infos[0].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
    infos[0].timeDomain = hostDomain;
    infos[1].sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT;
    infos[1].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
    uint64_t ts[2] = {0, 0};
    uint64_t maxDev = 0;
    if (pfnGetCalTs(dev.device, 2, infos, ts, &maxDev) != VK_SUCCESS)
      return false;
    hostOut = ts[0];
    gpuOut  = ts[1];
    return true;
  };

  // Read the host clock in the same domain as `hostDomain`.  POSIX:
  // CLOCK_MONOTONIC nanoseconds; Windows: QPC ticks.  Matches what the
  // driver returns for VK_TIME_DOMAIN_CLOCK_MONOTONIC{,_RAW}_EXT and
  // VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT respectively, so the
  // raw uint64 can be subtracted directly from `hostCalib`.
  auto hostNowInDriverDomain = [&]() -> uint64_t {
#if defined(_WIN32)
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (uint64_t)c.QuadPart;
#else
    struct timespec ts;
    clock_gettime((hostDomain == VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT)
                    ? CLOCK_MONOTONIC_RAW : CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif
  };

  // Conversion factor from host driver-domain "ticks" to nanoseconds.
  // POSIX: ticks ARE nanoseconds, factor = 1.  Windows: ticks are QPC,
  // factor = 1e9 / QPC_frequency.
  double hostTickNs = 1.0;
#if defined(_WIN32)
  {
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    hostTickNs = 1e9 / (double)freq.QuadPart;
  }
#endif

  // Calibrate ONCE before the loop.  vkGetCalibratedTimestampsEXT itself
  // costs a few microseconds; doing it per-iter pollutes the dispatch
  // measurement by exactly that cost.  Clock drift over our short
  // measurement window is negligible (<<1us across thousands of iters
  // on every modern OS clock).
  uint64_t hostCalib = 0, gpuCalib = 0;
  if (haveTimestamps && !getCalibrated(hostCalib, gpuCalib))
    haveTimestamps = false;

  bool submitFailed = false;

  // Warmup
  for (unsigned int w = 0; w < warmupCount; w++)
  {
    if (dev.submitAndWait(cmdBuf) != VK_SUCCESS) { submitFailed = true; break; }
  }

  double totalDispatchUs  = 0;
  double totalRoundtripUs = 0;
  unsigned int dispatchSamples = 0;
  if (!submitFailed)
  {
    for (unsigned int i = 0; i < iters; i++)
    {
      uint64_t hostSubmit = haveTimestamps ? hostNowInDriverDomain() : 0;
      auto chronoStart = std::chrono::high_resolution_clock::now();
      VkResult sr = dev.submitAndWait(cmdBuf);
      auto chronoEnd = std::chrono::high_resolution_clock::now();
      if (sr != VK_SUCCESS) { submitFailed = true; break; }
      totalRoundtripUs += (double)std::chrono::duration_cast<std::chrono::nanoseconds>(chronoEnd - chronoStart).count() / 1000.0;

      if (haveTimestamps)
      {
        uint64_t gpuStart = 0;
        if (vkGetQueryPoolResults(dev.device, queryPool, 0, 1,
                                  sizeof(gpuStart), &gpuStart, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
        {
          // Convert everything to nanoseconds, then to microseconds.
          double hostSubmitNs = (double)hostSubmit * hostTickNs;
          double hostCalibNs  = (double)hostCalib  * hostTickNs;
          double gpuDeltaNs   = (double)((int64_t)gpuStart - (int64_t)gpuCalib) * (double)timestampPeriodNs;
          double hostStartNs  = hostCalibNs + gpuDeltaNs;
          double dispatchUs   = (hostStartNs - hostSubmitNs) / 1000.0;
          if (dispatchUs > 0)
          {
            totalDispatchUs += dispatchUs;
            dispatchSamples++;
          }
        }
      }
    }
  }

  if (submitFailed)
  {
    test.skip("dispatch", ResultStatus::Error, "vkQueueSubmit/WaitIdle failed");
  }
  else if (dispatchSamples > 0)
  {
    float dispatchUs = (float)(totalDispatchUs / dispatchSamples);
    test.emit("dispatch", dispatchUs);
  }
  else
  {
    test.skip("dispatch", ResultStatus::Unsupported,
               "Needs VK_EXT_calibrated_timestamps");
  }
  if (submitFailed)
  {
    test.skip("roundtrip", ResultStatus::Error, "vkQueueSubmit/WaitIdle failed");
  }
  else
  {
    double avgRoundtripUs = totalRoundtripUs / static_cast<double>(iters);
    test.emit("roundtrip", (float)avgRoundtripUs);
  }

  vkFreeCommandBuffers(dev.device, dev.commandPool, 1, &cmdBuf);
  if (queryPool != VK_NULL_HANDLE)
    vkDestroyQueryPool(dev.device, queryPool, nullptr);
  vkDestroyPipeline(dev.device, pipeline, nullptr);
  vkDestroyShaderModule(dev.device, shaderModule, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);

  return 0;
}

#endif // ENABLE_VULKAN
