#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/common.h>
#include <cstring>
#include <chrono>
#include <vector>

// ---------------------------------------------------------------------------
// Host<->device transfer bandwidth (Vulkan)
// ---------------------------------------------------------------------------

int vkPeak::runTransferBandwidth(VulkanDevice &dev, benchmark_config_t &cfg)
{
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(dev.physicalDevice, &props);

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(dev.physicalDevice, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(dev.physicalDevice, &queueFamilyCount, queueFamilies.data());

  const bool canUseTimestamps =
      dev.info.computeQueueFamily < queueFamilies.size() &&
      queueFamilies[dev.info.computeQueueFamily].timestampValidBits > 0 &&
      props.limits.timestampPeriod > 0.0f;

  uint64_t bytes = cfg.transferBWMaxSize ? cfg.transferBWMaxSize : (1ull << 27);
  if (dev.info.maxAllocSize > 0)
    bytes = std::min(bytes, dev.info.maxAllocSize);
  bytes &= ~255ull;
  if (bytes == 0)
    bytes = 256;

  unsigned int forced = forceIters ? specifiedIters : 0;

  logger::TestSpec testSpec;
  testSpec.tag = "transfer_bandwidth";
  testSpec.display = "Transfer bandwidth";
  testSpec.unit = "gbps";
  auto test = currentDeviceScope->beginTest(testSpec);

  VkBuffer hostBuf = VK_NULL_HANDLE;
  VkBuffer devBuf = VK_NULL_HANDLE;
  VkDeviceMemory hostMem = VK_NULL_HANDLE;
  VkDeviceMemory devMem = VK_NULL_HANDLE;

  if (!dev.createBuffer(bytes,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                        hostBuf, hostMem) ||
      !dev.createBuffer(bytes,
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        devBuf, devMem))
  {
    log->note("Failed to allocate transfer buffers\n");
    if (hostBuf != VK_NULL_HANDLE) vkDestroyBuffer(dev.device, hostBuf, nullptr);
    if (hostMem != VK_NULL_HANDLE) vkFreeMemory(dev.device, hostMem, nullptr);
    if (devBuf != VK_NULL_HANDLE) vkDestroyBuffer(dev.device, devBuf, nullptr);
    if (devMem != VK_NULL_HANDLE) vkFreeMemory(dev.device, devMem, nullptr);
    return -1;
  }

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = dev.commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(dev.device, &allocInfo, &cmdBuf);

  VkQueryPool queryPool = VK_NULL_HANDLE;
  if (canUseTimestamps)
  {
    VkQueryPoolCreateInfo qpCI = {};
    qpCI.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpCI.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpCI.queryCount = 2;
    vkCreateQueryPool(dev.device, &qpCI, nullptr, &queryPool);
  }

  auto runCopy = [&](VkBuffer src, VkBuffer dst) -> float
  {
    VkBufferCopy region = {};
    region.size = bytes;

    auto recordCopy = [&](bool timed) {
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(cmdBuf, &beginInfo);
      if (timed && queryPool != VK_NULL_HANDLE)
      {
        vkCmdResetQueryPool(cmdBuf, queryPool, 0, 2);
        vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);
      }
      vkCmdCopyBuffer(cmdBuf, src, dst, 1, &region);
      if (timed && queryPool != VK_NULL_HANDLE)
        vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 1);
      vkEndCommandBuffer(cmdBuf);
    };

    // Run `n` timed copies, returning total us (or -1 on submit failure).
    auto runIters = [&](unsigned int n) -> float {
      float totalUs = 0.0f;
      for (unsigned int i = 0; i < n; i++)
      {
        recordCopy(queryPool != VK_NULL_HANDLE);
        auto start = std::chrono::high_resolution_clock::now();
        VkResult sr = dev.submitAndWait(cmdBuf);
        auto end = std::chrono::high_resolution_clock::now();
        if (sr != VK_SUCCESS)
        {
          vkResetCommandBuffer(cmdBuf, 0);
          return -1.0f;
        }

        double iterUs = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        if (queryPool != VK_NULL_HANDLE)
        {
          uint64_t ts[2] = {};
          VkResult qr = vkGetQueryPoolResults(dev.device, queryPool, 0, 2, sizeof(ts), ts,
                                              sizeof(uint64_t),
                                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
          // Strict > (not >=): some drivers (e.g. Adreno+Turnip) return both
          // timestamps with the same value for buffer-copy ranges instead of
          // failing the query.  Treat equality as "timer didn't tick" and
          // fall back to the host wall clock.
          if (qr == VK_SUCCESS && ts[1] > ts[0])
            totalUs += (float)((double)(ts[1] - ts[0]) * props.limits.timestampPeriod / 1000.0);
          else
            totalUs += (float)iterUs;
        }
        else
        {
          totalUs += (float)iterUs;
        }

        vkResetCommandBuffer(cmdBuf, 0);
      }
      return totalUs;
    };

    // Phase 1: untimed warmup.
    for (unsigned int w = 0; w < warmupCount; w++)
    {
      recordCopy(false);
      dev.submitAndWait(cmdBuf);
      vkResetCommandBuffer(cmdBuf, 0);
    }

    // Phase 2: timed probe -> per-iter time -> calibrated iters.
    unsigned int probeIters = 1;
    float probeUs = runIters(probeIters);
    if (probeUs < 0.0f) return -1.0f;
    if (probeUs <= 0.0f) probeUs = 1.0f;
    double per_iter_us = (double)probeUs / (double)probeIters;
    unsigned int iters = pickIters(per_iter_us, cfg.targetTimeUs, forced);

    // Phase 3: real timed run.
    float totalUs = runIters(iters);
    if (totalUs < 0.0f) return -1.0f;
    // A genuine zero reading (no measurable wall-clock time AND no
    // timestamp delta) is the timer-broken case -- don't paper over it
    // with FLT_EPSILON, since (bytes / FLT_EPSILON / 1e3) prints 1.13e12
    // GBPS and looks like a real result.
    if (totalUs <= 0.0f)
      return -1.0f;
    return totalUs / static_cast<float>(iters);
  };

  auto reportCopy = [&](const char *metric, float us)
  {
    if (us <= 0.0f)
    {
      test.skip(metric, ResultStatus::Error,
                 "vkQueueSubmit/WaitIdle failed or timer returned zero");
      return;
    }
    float gbps = (float)bytes / us / 1e3f;
    test.emit(metric, gbps);
  };

  reportCopy("h2d", runCopy(hostBuf, devBuf));
  reportCopy("d2h", runCopy(devBuf, hostBuf));

  if (queryPool != VK_NULL_HANDLE)
    vkDestroyQueryPool(dev.device, queryPool, nullptr);
  vkFreeCommandBuffers(dev.device, dev.commandPool, 1, &cmdBuf);
  vkDestroyBuffer(dev.device, hostBuf, nullptr);
  vkFreeMemory(dev.device, hostMem, nullptr);
  vkDestroyBuffer(dev.device, devBuf, nullptr);
  vkFreeMemory(dev.device, devMem, nullptr);

  return 0;
}

#endif // ENABLE_VULKAN
