#ifdef ENABLE_VULKAN

#include <vk_peak.h>
#include <cstring>
#include <sstream>
#include <chrono>
#include <iostream>

// ---------------------------------------------------------------------------
// VulkanDevice
// ---------------------------------------------------------------------------

VulkanDevice::VulkanDevice()
  : physicalDevice(VK_NULL_HANDLE), device(VK_NULL_HANDLE),
    computeQueue(VK_NULL_HANDLE), commandPool(VK_NULL_HANDLE)
{
}

VulkanDevice::~VulkanDevice()
{
  cleanup();
}

bool VulkanDevice::init(VkPhysicalDevice physDev)
{
  physicalDevice = physDev;

  // Query properties
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physDev, &props);

  info.deviceName = props.deviceName;
  info.apiVersion = std::to_string(VK_VERSION_MAJOR(props.apiVersion)) + "." +
                    std::to_string(VK_VERSION_MINOR(props.apiVersion)) + "." +
                    std::to_string(VK_VERSION_PATCH(props.apiVersion));
  info.driverVersion = std::to_string(VK_VERSION_MAJOR(props.driverVersion)) + "." +
                       std::to_string(VK_VERSION_MINOR(props.driverVersion)) + "." +
                       std::to_string(VK_VERSION_PATCH(props.driverVersion));
  info.maxWGSize = std::min(props.limits.maxComputeWorkGroupSize[0], (uint32_t)MAX_WG_SIZE);
  info.maxAllocSize = props.limits.maxStorageBufferRange;
  info.deviceType = props.deviceType;

  // Estimate CUs from maxComputeWorkGroupCount (rough proxy)
  info.numCUs = 0; // Vulkan doesn't expose CU count directly

  // Get memory heap size (device-local)
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
  info.heapSize = 0;
  for (uint32_t i = 0; i < memProps.memoryHeapCount; i++)
  {
    if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
    {
      info.heapSize = std::max(info.heapSize, (uint64_t)memProps.memoryHeaps[i].size);
    }
  }

  // Find compute queue family
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &queueFamilyCount, queueFamilies.data());

  info.computeQueueFamily = UINT32_MAX;
  for (uint32_t i = 0; i < queueFamilyCount; i++)
  {
    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
    {
      info.computeQueueFamily = i;
      break;
    }
  }
  if (info.computeQueueFamily == UINT32_MAX)
    return false;

  // Query supported extensions
  uint32_t extCount = 0;
  vkEnumerateDeviceExtensionProperties(physDev, nullptr, &extCount, nullptr);
  std::vector<VkExtensionProperties> extProps(extCount);
  vkEnumerateDeviceExtensionProperties(physDev, nullptr, &extCount, extProps.data());
  auto hasExt = [&](const char *name) {
    for (auto &e : extProps)
      if (strcmp(e.extensionName, name) == 0) return true;
    return false;
  };

  std::vector<const char *> enabledExts;
  info.int8DotProductSupported = false;

#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
  // INT8 dot-product: VK_KHR_shader_integer_dot_product + shaderInt8.
  // Compiled in only when glslc could build the shader AND the Vulkan loader
  // exposes vkGetPhysicalDeviceFeatures2 (Vulkan 1.1 core). On older Android
  // API levels (< 28) that symbol isn't present in the NDK sysroot stub, so
  // we don't reference it there.
  VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dpFeatures = {};
  dpFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
  VkPhysicalDeviceShaderFloat16Int8FeaturesKHR f16i8Features = {};
  f16i8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;

  if (hasExt(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) &&
      hasExt(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME))
  {
    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &dpFeatures;
    dpFeatures.pNext = &f16i8Features;
    vkGetPhysicalDeviceFeatures2(physDev, &features2);

    if (dpFeatures.shaderIntegerDotProduct && f16i8Features.shaderInt8)
    {
      enabledExts.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
      enabledExts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
      if (hasExt(VK_KHR_8BIT_STORAGE_EXTENSION_NAME))
        enabledExts.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
      info.int8DotProductSupported = true;
    }
  }
#endif

#if defined(__APPLE__) || defined(__MACOSX)
  if (hasExt("VK_KHR_portability_subset"))
    enabledExts.push_back("VK_KHR_portability_subset");
#endif

  // Create logical device
  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCI = {};
  queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCI.queueFamilyIndex = info.computeQueueFamily;
  queueCI.queueCount = 1;
  queueCI.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceCI = {};
  deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCI.queueCreateInfoCount = 1;
  deviceCI.pQueueCreateInfos = &queueCI;
  deviceCI.enabledExtensionCount = (uint32_t)enabledExts.size();
  deviceCI.ppEnabledExtensionNames = enabledExts.empty() ? nullptr : enabledExts.data();

#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
  VkPhysicalDeviceFeatures2 enabledFeatures2 = {};
  enabledFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  if (info.int8DotProductSupported)
  {
    enabledFeatures2.pNext = &dpFeatures;
    dpFeatures.pNext = &f16i8Features;
    f16i8Features.pNext = nullptr;
    deviceCI.pNext = &enabledFeatures2;
  }
#endif

  if (vkCreateDevice(physDev, &deviceCI, nullptr, &device) != VK_SUCCESS)
    return false;

  vkGetDeviceQueue(device, info.computeQueueFamily, 0, &computeQueue);

  // Create command pool
  VkCommandPoolCreateInfo poolCI = {};
  poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolCI.queueFamilyIndex = info.computeQueueFamily;
  poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(device, &poolCI, nullptr, &commandPool) != VK_SUCCESS)
    return false;

  return true;
}

void VulkanDevice::cleanup()
{
  if (device != VK_NULL_HANDLE)
  {
    vkDeviceWaitIdle(device);
    if (commandPool != VK_NULL_HANDLE)
      vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    device = VK_NULL_HANDLE;
    commandPool = VK_NULL_HANDLE;
  }
}

bool VulkanDevice::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags memProps,
                                VkBuffer &buffer, VkDeviceMemory &memory)
{
  VkBufferCreateInfo bufCI = {};
  bufCI.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufCI.size = size;
  bufCI.usage = usage;
  bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if (vkCreateBuffer(device, &bufCI, nullptr, &buffer) != VK_SUCCESS)
    return false;

  VkMemoryRequirements memReqs;
  vkGetBufferMemoryRequirements(device, buffer, &memReqs);

  VkMemoryAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize = memReqs.size;
  allocInfo.memoryTypeIndex = findMemoryType(memReqs.memoryTypeBits, memProps);

  if (allocInfo.memoryTypeIndex == UINT32_MAX)
  {
    vkDestroyBuffer(device, buffer, nullptr);
    return false;
  }

  if (vkAllocateMemory(device, &allocInfo, nullptr, &memory) != VK_SUCCESS)
  {
    vkDestroyBuffer(device, buffer, nullptr);
    return false;
  }

  vkBindBufferMemory(device, buffer, memory, 0);
  return true;
}

bool VulkanDevice::createComputePipeline(const uint32_t *spirv, size_t spirvSize,
                                         VkDescriptorSetLayout dsLayout,
                                         VkPipelineLayout pipeLayout,
                                         VkPipeline &pipeline)
{
  VkShaderModuleCreateInfo smCI = {};
  smCI.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  smCI.codeSize = spirvSize;
  smCI.pCode = spirv;

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &smCI, nullptr, &shaderModule) != VK_SUCCESS)
    return false;

  VkPipelineShaderStageCreateInfo stageCI = {};
  stageCI.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageCI.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageCI.module = shaderModule;
  stageCI.pName = "main";

  VkComputePipelineCreateInfo pipelineCI = {};
  pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCI.stage = stageCI;
  pipelineCI.layout = pipeLayout;

  VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline);
  vkDestroyShaderModule(device, shaderModule, nullptr);

  return result == VK_SUCCESS;
}

void VulkanDevice::submitAndWait(VkCommandBuffer cmdBuf)
{
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuf;

  vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(computeQueue);
}

uint32_t VulkanDevice::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);

  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
  {
    if ((typeFilter & (1 << i)) &&
        (memProps.memoryTypes[i].propertyFlags & properties) == properties)
    {
      return i;
    }
  }
  return UINT32_MAX;
}

// ---------------------------------------------------------------------------
// vkPeak
// ---------------------------------------------------------------------------

vkPeak::vkPeak()
  : warmupCount(2), specifiedIters(0), forceIters(false), listDevices(false),
    instance(VK_NULL_HANDLE)
{
}

vkPeak::~vkPeak()
{
  cleanup();
}

bool vkPeak::initInstance()
{
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "clpeak-vulkan";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "clpeak";
  // Request 1.1 so vkGetPhysicalDeviceFeatures2 / pNext chaining on features
  // are available core (needed for optional extension feature queries).
  appInfo.apiVersion = VK_API_VERSION_1_1;

  VkInstanceCreateInfo instCI = {};
  instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instCI.pApplicationInfo = &appInfo;

#if defined(__APPLE__) || defined(__MACOSX)
  // MoltenVK portability
  instCI.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
  const char *extensions[] = {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
  instCI.enabledExtensionCount = 1;
  instCI.ppEnabledExtensionNames = extensions;
#endif

  if (vkCreateInstance(&instCI, nullptr, &instance) != VK_SUCCESS)
    return false;

  uint32_t devCount = 0;
  vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
  if (devCount == 0)
    return false;

  physicalDevices.resize(devCount);
  vkEnumeratePhysicalDevices(instance, &devCount, physicalDevices.data());

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
                        uint32_t groupCountX, unsigned int iters,
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

  // Warmup
  for (unsigned int w = 0; w < warmupCount; w++)
  {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descriptorSet, 0, nullptr);
    if (pushData && pushSize > 0)
      vkCmdPushConstants(cmdBuf, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    vkCmdDispatch(cmdBuf, groupCountX, 1, 1);
    vkEndCommandBuffer(cmdBuf);
    dev.submitAndWait(cmdBuf);
    vkResetCommandBuffer(cmdBuf, 0);
  }

  // Timed runs
  float timed = 0;
  for (unsigned int i = 0; i < iters; i++)
  {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmdBuf, &beginInfo);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descriptorSet, 0, nullptr);
    if (pushData && pushSize > 0)
      vkCmdPushConstants(cmdBuf, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    vkCmdDispatch(cmdBuf, groupCountX, 1, 1);
    vkEndCommandBuffer(cmdBuf);

    auto start = std::chrono::high_resolution_clock::now();
    dev.submitAndWait(cmdBuf);
    auto end = std::chrono::high_resolution_clock::now();
    timed += (float)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    vkResetCommandBuffer(cmdBuf, 0);
  }

  vkFreeCommandBuffers(dev.device, dev.commandPool, 1, &cmdBuf);
  return timed / static_cast<float>(iters);
}

int vkPeak::runAll()
{
  if (!initInstance())
  {
    std::cerr << "Vulkan: failed to create instance or no devices found\n";
    return -1;
  }

  if (listDevices)
  {
    for (size_t i = 0; i < physicalDevices.size(); i++)
    {
      VkPhysicalDeviceProperties props;
      vkGetPhysicalDeviceProperties(physicalDevices[i], &props);
      const char *typeStr = (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? "Discrete GPU" :
                            (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? "Integrated GPU" :
                            (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) ? "CPU" : "Other";
      std::cout << "  Vulkan Device " << i << ": " << props.deviceName
                << " [" << typeStr << "]"
                << "\n";
      std::cout << "    API       : " << VK_VERSION_MAJOR(props.apiVersion) << "."
                << VK_VERSION_MINOR(props.apiVersion) << "."
                << VK_VERSION_PATCH(props.apiVersion) << "\n";
    }
    return 0;
  }

  for (size_t d = 0; d < physicalDevices.size(); d++)
  {
    VulkanDevice dev;
    if (!dev.init(physicalDevices[d]))
    {
      std::cerr << "Vulkan: failed to init device " << d << "\n";
      continue;
    }

    benchmark_config_t cfg = benchmark_config_t::forDevice(
        (dev.info.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) ? CL_DEVICE_TYPE_CPU : CL_DEVICE_TYPE_GPU);

    if (forceIters)
    {
      cfg.computeIters = specifiedIters;
      cfg.globalBWIters = specifiedIters;
    }

    log->print(NEWLINE "Vulkan Device: " + dev.info.deviceName + NEWLINE);
    log->print(TAB "API version   : " + dev.info.apiVersion + NEWLINE);
    log->print(TAB "Driver version: " + dev.info.driverVersion + NEWLINE);
    log->print(TAB "VRAM          : ");
    log->print((unsigned int)(dev.info.heapSize / (1024 * 1024)));
    log->print(" MB" NEWLINE);

    log->xmlOpenTag("device");
    log->xmlAppendAttribs("name", dev.info.deviceName);
    log->xmlAppendAttribs("api", "vulkan");
    log->xmlAppendAttribs("driver_version", dev.info.driverVersion);

    runComputeSP(dev, cfg);
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
    runComputeInt8DP(dev, cfg);
#endif
    runGlobalBandwidth(dev, cfg);

    log->print(NEWLINE);
    log->xmlCloseTag(); // device
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Compute SP benchmark (Vulkan)
// ---------------------------------------------------------------------------

int vkPeak::runComputeSP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = cfg.computeIters;

  // Size the dispatch to match OpenCL's sizing: large enough to saturate the
  // device and amortize submit overhead. Vulkan doesn't expose CU count, so
  // target 32M work items (typical of OpenCL's numCUs*2048*maxWGSize on most
  // GPUs once clamped), bounded by maxStorageBufferRange.
  const uint32_t wgSize = 256;
  uint64_t globalWIs = 32ULL * 1024 * 1024;
  if (globalWIs * sizeof(float) > dev.info.maxAllocSize)
    globalWIs = dev.info.maxAllocSize / sizeof(float);
  globalWIs = (globalWIs / wgSize) * wgSize;
  uint32_t numGroups = (uint32_t)(globalWIs / wgSize);

  log->print(NEWLINE TAB "Single-precision compute (GFLOPS)" NEWLINE);
  log->xmlOpenTag("single_precision_compute");
  log->xmlAppendAttribs("unit", "gflops");

  // Create output buffer
  VkBuffer outputBuf;
  VkDeviceMemory outputMem;
  if (!dev.createBuffer(globalWIs * sizeof(float),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        outputBuf, outputMem))
  {
    log->print(TAB TAB "Failed to allocate buffer" NEWLINE);
    log->xmlCloseTag();
    return -1;
  }

  // Descriptor set layout
  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dsLayoutCI = {};
  dsLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsLayoutCI.bindingCount = 1;
  dsLayoutCI.pBindings = &binding;

  VkDescriptorSetLayout dsLayout;
  vkCreateDescriptorSetLayout(dev.device, &dsLayoutCI, nullptr, &dsLayout);

  // Push constant range for the scalar A
  VkPushConstantRange pushRange = {};
  pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushRange.offset = 0;
  pushRange.size = sizeof(float);

  VkPipelineLayoutCreateInfo pipeLayoutCI = {};
  pipeLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeLayoutCI.setLayoutCount = 1;
  pipeLayoutCI.pSetLayouts = &dsLayout;
  pipeLayoutCI.pushConstantRangeCount = 1;
  pipeLayoutCI.pPushConstantRanges = &pushRange;

  VkPipelineLayout pipeLayout;
  vkCreatePipelineLayout(dev.device, &pipeLayoutCI, nullptr, &pipeLayout);

  // Create pipeline
  VkPipeline pipeline;
  if (!dev.createComputePipeline(vk_shaders::compute_sp_v1, vk_shaders::compute_sp_v1_size,
                                  dsLayout, pipeLayout, pipeline))
  {
    log->print(TAB TAB "Failed to create compute pipeline" NEWLINE);
    vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
    vkDestroyBuffer(dev.device, outputBuf, nullptr);
    vkFreeMemory(dev.device, outputMem, nullptr);
    log->xmlCloseTag();
    return -1;
  }

  // Descriptor pool + set
  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 1;

  VkDescriptorPoolCreateInfo dpCI = {};
  dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpCI.maxSets = 1;
  dpCI.poolSizeCount = 1;
  dpCI.pPoolSizes = &poolSize;

  VkDescriptorPool descPool;
  vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &descPool);

  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = descPool;
  dsAI.descriptorSetCount = 1;
  dsAI.pSetLayouts = &dsLayout;

  VkDescriptorSet descSet;
  vkAllocateDescriptorSets(dev.device, &dsAI, &descSet);

  VkDescriptorBufferInfo bufInfo = {};
  bufInfo.buffer = outputBuf;
  bufInfo.offset = 0;
  bufInfo.range = globalWIs * sizeof(float);

  VkWriteDescriptorSet write = {};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = descSet;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &bufInfo;

  vkUpdateDescriptorSets(dev.device, 1, &write, 0, nullptr);

  // Run benchmark
  log->print(TAB TAB "float   : ");

  float A = 1.3f;
  float timed = runKernel(dev, pipeline, pipeLayout, descSet, numGroups, iters, &A, sizeof(float));
  float gflops = (static_cast<float>(globalWIs) * static_cast<float>(COMPUTE_FP_WORK_PER_WI)) / timed / 1e3f;

  log->print(gflops);
  log->print(NEWLINE);
  log->xmlRecord("float", gflops);

  // Cleanup
  vkDestroyPipeline(dev.device, pipeline, nullptr);
  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, outputBuf, nullptr);
  vkFreeMemory(dev.device, outputMem, nullptr);

  log->xmlCloseTag(); // single_precision_compute
  return 0;
}

#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
// ---------------------------------------------------------------------------
// INT8 dot-product benchmark (Vulkan, VK_KHR_shader_integer_dot_product)
// ---------------------------------------------------------------------------

int vkPeak::runComputeInt8DP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  log->print(NEWLINE TAB "INT8 dot-product compute (GIOPS)" NEWLINE);
  log->xmlOpenTag("integer_compute_int8_dp");
  log->xmlAppendAttribs("unit", "giops");

  if (!dev.info.int8DotProductSupported)
  {
    log->print(TAB TAB "VK_KHR_shader_integer_dot_product / shaderInt8 not supported! Skipped" NEWLINE);
    log->xmlCloseTag();
    return 0;
  }

  unsigned int iters = cfg.computeIters;
  const uint32_t wgSize = 256;
  uint64_t globalWIs = 32ULL * 1024 * 1024;
  if (globalWIs * sizeof(int32_t) > dev.info.maxAllocSize)
    globalWIs = dev.info.maxAllocSize / sizeof(int32_t);
  globalWIs = (globalWIs / wgSize) * wgSize;
  uint32_t numGroups = (uint32_t)(globalWIs / wgSize);

  VkBuffer outputBuf;
  VkDeviceMemory outputMem;
  if (!dev.createBuffer(globalWIs * sizeof(int32_t),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        outputBuf, outputMem))
  {
    log->print(TAB TAB "Failed to allocate buffer" NEWLINE);
    log->xmlCloseTag();
    return -1;
  }

  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dsLayoutCI = {};
  dsLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsLayoutCI.bindingCount = 1;
  dsLayoutCI.pBindings = &binding;

  VkDescriptorSetLayout dsLayout;
  vkCreateDescriptorSetLayout(dev.device, &dsLayoutCI, nullptr, &dsLayout);

  VkPushConstantRange pushRange = {};
  pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  pushRange.offset = 0;
  pushRange.size = sizeof(int32_t);

  VkPipelineLayoutCreateInfo pipeLayoutCI = {};
  pipeLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeLayoutCI.setLayoutCount = 1;
  pipeLayoutCI.pSetLayouts = &dsLayout;
  pipeLayoutCI.pushConstantRangeCount = 1;
  pipeLayoutCI.pPushConstantRanges = &pushRange;

  VkPipelineLayout pipeLayout;
  vkCreatePipelineLayout(dev.device, &pipeLayoutCI, nullptr, &pipeLayout);

  VkPipeline pipeline;
  if (!dev.createComputePipeline(vk_shaders::compute_int8_dp_v1, vk_shaders::compute_int8_dp_v1_size,
                                  dsLayout, pipeLayout, pipeline))
  {
    log->print(TAB TAB "Failed to create INT8-DP compute pipeline (driver may not honor extension)" NEWLINE);
    vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
    vkDestroyBuffer(dev.device, outputBuf, nullptr);
    vkFreeMemory(dev.device, outputMem, nullptr);
    log->xmlCloseTag();
    return 0;
  }

  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 1;

  VkDescriptorPoolCreateInfo dpCI = {};
  dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpCI.maxSets = 1;
  dpCI.poolSizeCount = 1;
  dpCI.pPoolSizes = &poolSize;

  VkDescriptorPool descPool;
  vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &descPool);

  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = descPool;
  dsAI.descriptorSetCount = 1;
  dsAI.pSetLayouts = &dsLayout;

  VkDescriptorSet descSet;
  vkAllocateDescriptorSets(dev.device, &dsAI, &descSet);

  VkDescriptorBufferInfo bufInfo = {};
  bufInfo.buffer = outputBuf;
  bufInfo.offset = 0;
  bufInfo.range = globalWIs * sizeof(int32_t);

  VkWriteDescriptorSet write = {};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = descSet;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &bufInfo;

  vkUpdateDescriptorSets(dev.device, 1, &write, 0, nullptr);

  log->print(TAB TAB "int8_dp : ");

  int32_t A = 4;
  float timed = runKernel(dev, pipeline, pipeLayout, descSet, numGroups, iters, &A, sizeof(int32_t));
  float giops = (static_cast<float>(globalWIs) * static_cast<float>(COMPUTE_INT8_DP_WORK_PER_WI)) / timed / 1e3f;

  log->print(giops);
  log->print(NEWLINE);
  log->xmlRecord("int8_dp", giops);

  vkDestroyPipeline(dev.device, pipeline, nullptr);
  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, outputBuf, nullptr);
  vkFreeMemory(dev.device, outputMem, nullptr);

  log->xmlCloseTag(); // integer_compute_int8_dp
  return 0;
}
#endif // CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1

// ---------------------------------------------------------------------------
// Global bandwidth benchmark (Vulkan)
// ---------------------------------------------------------------------------

int vkPeak::runGlobalBandwidth(VulkanDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = cfg.globalBWIters;
  const uint32_t wgSize = 256;

  uint64_t maxItems = dev.info.maxAllocSize / sizeof(float) / 2;
  uint64_t numItems = (maxItems / (wgSize * FETCH_PER_WI)) * (wgSize * FETCH_PER_WI);
  if (numItems > cfg.globalBWMaxSize / sizeof(float))
    numItems = (cfg.globalBWMaxSize / sizeof(float) / (wgSize * FETCH_PER_WI)) * (wgSize * FETCH_PER_WI);

  uint32_t numGroups = (uint32_t)(numItems / FETCH_PER_WI / wgSize);
  if (numGroups == 0) numGroups = 1;

  log->print(NEWLINE TAB "Global memory bandwidth (GBPS)" NEWLINE);
  log->xmlOpenTag("global_memory_bandwidth");
  log->xmlAppendAttribs("unit", "gbps");

  // Create input + output buffers
  VkBuffer inputBuf, outputBuf;
  VkDeviceMemory inputMem, outputMem;

  if (!dev.createBuffer(numItems * sizeof(float),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        inputBuf, inputMem) ||
      !dev.createBuffer(numItems * sizeof(float),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        outputBuf, outputMem))
  {
    log->print(TAB TAB "Failed to allocate buffers" NEWLINE);
    log->xmlCloseTag();
    return -1;
  }

  // Descriptor set layout (2 bindings: input + output)
  VkDescriptorSetLayoutBinding bindings[2] = {};
  bindings[0].binding = 0;
  bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[0].descriptorCount = 1;
  bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bindings[1].binding = 1;
  bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bindings[1].descriptorCount = 1;
  bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dsLayoutCI = {};
  dsLayoutCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dsLayoutCI.bindingCount = 2;
  dsLayoutCI.pBindings = bindings;

  VkDescriptorSetLayout dsLayout;
  vkCreateDescriptorSetLayout(dev.device, &dsLayoutCI, nullptr, &dsLayout);

  VkPipelineLayoutCreateInfo pipeLayoutCI = {};
  pipeLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeLayoutCI.setLayoutCount = 1;
  pipeLayoutCI.pSetLayouts = &dsLayout;

  VkPipelineLayout pipeLayout;
  vkCreatePipelineLayout(dev.device, &pipeLayoutCI, nullptr, &pipeLayout);

  VkPipeline pipeline;
  if (!dev.createComputePipeline(vk_shaders::global_bandwidth_v1, vk_shaders::global_bandwidth_v1_size,
                                  dsLayout, pipeLayout, pipeline))
  {
    log->print(TAB TAB "Failed to create pipeline" NEWLINE);
    vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
    vkDestroyBuffer(dev.device, inputBuf, nullptr);
    vkFreeMemory(dev.device, inputMem, nullptr);
    vkDestroyBuffer(dev.device, outputBuf, nullptr);
    vkFreeMemory(dev.device, outputMem, nullptr);
    log->xmlCloseTag();
    return -1;
  }

  // Descriptor pool + set
  VkDescriptorPoolSize poolSize = {};
  poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  poolSize.descriptorCount = 2;

  VkDescriptorPoolCreateInfo dpCI = {};
  dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpCI.maxSets = 1;
  dpCI.poolSizeCount = 1;
  dpCI.pPoolSizes = &poolSize;

  VkDescriptorPool descPool;
  vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &descPool);

  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = descPool;
  dsAI.descriptorSetCount = 1;
  dsAI.pSetLayouts = &dsLayout;

  VkDescriptorSet descSet;
  vkAllocateDescriptorSets(dev.device, &dsAI, &descSet);

  VkDescriptorBufferInfo bufInfos[2] = {};
  bufInfos[0].buffer = inputBuf;
  bufInfos[0].offset = 0;
  bufInfos[0].range = numItems * sizeof(float);
  bufInfos[1].buffer = outputBuf;
  bufInfos[1].offset = 0;
  bufInfos[1].range = numItems * sizeof(float);

  VkWriteDescriptorSet writes[2] = {};
  for (int i = 0; i < 2; i++)
  {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufInfos[i];
  }
  vkUpdateDescriptorSets(dev.device, 2, writes, 0, nullptr);

  // Run
  log->print(TAB TAB "float   : ");
  float timed = runKernel(dev, pipeline, pipeLayout, descSet, numGroups, iters);
  float gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

  log->print(gbps);
  log->print(NEWLINE);
  log->xmlRecord("float", gbps);

  // Cleanup
  vkDestroyPipeline(dev.device, pipeline, nullptr);
  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, inputBuf, nullptr);
  vkFreeMemory(dev.device, inputMem, nullptr);
  vkDestroyBuffer(dev.device, outputBuf, nullptr);
  vkFreeMemory(dev.device, outputMem, nullptr);

  log->xmlCloseTag(); // global_memory_bandwidth
  return 0;
}

#endif // ENABLE_VULKAN
