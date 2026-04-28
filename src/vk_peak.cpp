#ifdef ENABLE_VULKAN

#include <vk_peak.h>
#include <cstring>
#include <sstream>
#include <chrono>

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

bool VulkanDevice::init(VkInstance inst, VkPhysicalDevice physDev)
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
  info.float16Supported = false;
  info.bfloat16Supported = false;
  info.cooperativeMatrixSupported = false;
  info.fp8Supported = false;
  info.coopmatFP16Supported = false;
  info.coopmatBF16Supported = false;
  info.coopmatINT8K = 0;
  info.coopmatFP8E4M3Supported = false;
  info.coopmatFP8E5M2Supported = false;

  // Feature query for optional shader types.  Uses vkGetPhysicalDeviceFeatures2
  // (Vulkan 1.1 core).  Each optional shader's feature struct gets chained
  // into pNext, queried, and then re-chained at vkCreateDevice time if the
  // driver advertises the capability we need.  Adding a new dtype reduces
  // to: declare the struct + macro + two conditional blocks here and one
  // runCompute* wrapper.
  //
  // On Android API < 28 Features2 isn't in the stub loader, so none of the
  // optional shader defines should be active unless the NDK provides a
  // modern-enough loader (see CompileShaders.cmake).
#if defined(CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1) || defined(CLPEAK_VK_HAS_COMPUTE_MP_V1) || defined(CLPEAK_VK_HAS_COMPUTE_BF16_V1) || defined(CLPEAK_VK_HAS_ANY_COOPMAT)
  VkPhysicalDeviceShaderFloat16Int8FeaturesKHR f16i8Features = {};
  f16i8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
  VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dpFeatures = {};
  dpFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
#endif
#if defined(CLPEAK_VK_HAS_COMPUTE_BF16_V1) || defined(CLPEAK_VK_HAS_COOPMAT_BF16)
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16Features = {};
  bf16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
#endif
#ifdef CLPEAK_VK_HAS_ANY_COOPMAT
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmatFeatures = {};
  coopmatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  // Cooperative-matrix shaders emitted by glslang use MemoryModel Vulkan via
  // GL_KHR_memory_scope_semantics, which requires the vulkanMemoryModel
  // feature bit (core in 1.2, otherwise VK_KHR_vulkan_memory_model).
  VkPhysicalDeviceVulkanMemoryModelFeatures vmmFeatures = {};
  vmmFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
#ifdef CLPEAK_VK_HAS_ANY_COOPMAT_FP8
  VkPhysicalDeviceShaderFloat8FeaturesEXT fp8Features = {};
  fp8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
#endif
#endif

  // Small helper: prepend a features struct onto a pNext chain.
  // All VkPhysicalDevice*Features* structs begin with the (sType, pNext)
  // pair described by VkBaseOutStructure, which is what we rely on here.
  auto chainPNext = [](void *head, void *newNode) -> void * {
    reinterpret_cast<VkBaseOutStructure *>(newNode)->pNext =
        reinterpret_cast<VkBaseOutStructure *>(head);
    return newNode;
  };

  bool f16i8ExtEnabled = false;
  if (hasExt(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME))
  {
    void *chain = nullptr;
    chain = chainPNext(chain, &f16i8Features);
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
    if (hasExt(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
      chain = chainPNext(chain, &dpFeatures);
#endif
#if defined(CLPEAK_VK_HAS_COMPUTE_BF16_V1) || defined(CLPEAK_VK_HAS_COOPMAT_BF16)
    if (hasExt(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME))
      chain = chainPNext(chain, &bf16Features);
#endif
#ifdef CLPEAK_VK_HAS_ANY_COOPMAT
    // Cooperative-matrix requires vulkanMemoryModel.  Chain both feature
    // structs whenever the coopmat extension is present; the memory-model
    // struct is core-1.2 and is what drivers advertising coopmat expose.
    bool hasCoopmatExt = hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    if (hasCoopmatExt)
    {
      chain = chainPNext(chain, &coopmatFeatures);
      chain = chainPNext(chain, &vmmFeatures);
    }
#ifdef CLPEAK_VK_HAS_ANY_COOPMAT_FP8
    bool hasFP8Ext = hasExt(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    if (hasFP8Ext)
      chain = chainPNext(chain, &fp8Features);
#endif
#endif

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = chain;
    // Resolve at runtime: on Android API 26 the NDK's libvulkan.so does not
    // export the 1.1-core symbol statically, but the 1.1 driver underneath
    // still answers the call.  Fall back to the KHR alias (VK_KHR_get_phys-
    // ical_device_properties2, a 1.0 extension) if the core entrypoint is
    // missing.  If neither resolves, leave all optional features disabled.
    auto pfnGetFeat2 = (PFN_vkGetPhysicalDeviceFeatures2)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceFeatures2");
    if (!pfnGetFeat2)
      pfnGetFeat2 = (PFN_vkGetPhysicalDeviceFeatures2)
          vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceFeatures2KHR");
    if (pfnGetFeat2)
      pfnGetFeat2(physDev, &features2);

#ifdef CLPEAK_VK_HAS_COMPUTE_MP_V1
    if (f16i8Features.shaderFloat16)
    {
      enabledExts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
      f16i8ExtEnabled = true;
      info.float16Supported = true;
    }
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
    if (hasExt(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) &&
        dpFeatures.shaderIntegerDotProduct && f16i8Features.shaderInt8)
    {
      enabledExts.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
      if (!f16i8ExtEnabled)
      {
        enabledExts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        f16i8ExtEnabled = true;
      }
      if (hasExt(VK_KHR_8BIT_STORAGE_EXTENSION_NAME))
        enabledExts.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
      info.int8DotProductSupported = true;
    }
#endif
#if defined(CLPEAK_VK_HAS_COMPUTE_BF16_V1) || defined(CLPEAK_VK_HAS_COOPMAT_BF16)
    if (hasExt(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) && bf16Features.shaderBFloat16Type)
    {
      enabledExts.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
      info.bfloat16Supported = true;
    }
#endif
#ifdef CLPEAK_VK_HAS_ANY_COOPMAT
    if (hasCoopmatExt && coopmatFeatures.cooperativeMatrix && vmmFeatures.vulkanMemoryModel)
    {
      enabledExts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
      info.cooperativeMatrixSupported = true;
      // FP16 coopmat inputs also need shaderFloat16.
#ifdef CLPEAK_VK_HAS_COOPMAT_FP16
      if (f16i8Features.shaderFloat16 && !f16i8ExtEnabled)
      {
        enabledExts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        f16i8ExtEnabled = true;
        info.float16Supported = true;
      }
#endif
      // INT8 coopmat inputs also need shaderInt8 + 8-bit storage.
#ifdef CLPEAK_VK_HAS_COOPMAT_INT8
      if (f16i8Features.shaderInt8)
      {
        if (!f16i8ExtEnabled)
        {
          enabledExts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
          f16i8ExtEnabled = true;
        }
        if (hasExt(VK_KHR_8BIT_STORAGE_EXTENSION_NAME))
          enabledExts.push_back(VK_KHR_8BIT_STORAGE_EXTENSION_NAME);
      }
#endif
      // FP8 coopmat inputs additionally need VK_EXT_shader_float8 with
      // shaderFloat8 + shaderFloat8CooperativeMatrix both advertised.
#ifdef CLPEAK_VK_HAS_ANY_COOPMAT_FP8
      if (hasFP8Ext && fp8Features.shaderFloat8 &&
          fp8Features.shaderFloat8CooperativeMatrix)
      {
        enabledExts.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        info.fp8Supported = true;
      }
#endif
    }
#endif
  }
  (void)f16i8ExtEnabled;
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

#if defined(CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1) || defined(CLPEAK_VK_HAS_COMPUTE_MP_V1) || defined(CLPEAK_VK_HAS_COMPUTE_BF16_V1) || defined(CLPEAK_VK_HAS_ANY_COOPMAT)
  // Re-chain the feature structs we actually enabled for vkCreateDevice.
  // The query-phase chain is discarded; these are the features we ask the
  // driver to turn on.
  VkPhysicalDeviceFeatures2 enabledFeatures2 = {};
  enabledFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  void *enabledChain = nullptr;
  if (info.float16Supported || info.int8DotProductSupported)
  {
    f16i8Features.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &f16i8Features);
  }
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
  if (info.int8DotProductSupported)
  {
    dpFeatures.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &dpFeatures);
  }
#endif
#if defined(CLPEAK_VK_HAS_COMPUTE_BF16_V1) || defined(CLPEAK_VK_HAS_COOPMAT_BF16)
  if (info.bfloat16Supported)
  {
    bf16Features.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &bf16Features);
  }
#endif
#ifdef CLPEAK_VK_HAS_ANY_COOPMAT
  if (info.cooperativeMatrixSupported)
  {
    coopmatFeatures.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &coopmatFeatures);
    vmmFeatures.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &vmmFeatures);
  }
#ifdef CLPEAK_VK_HAS_ANY_COOPMAT_FP8
  if (info.fp8Supported)
  {
    fp8Features.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &fp8Features);
  }
#endif
#endif
  if (enabledChain)
  {
    enabledFeatures2.pNext = enabledChain;
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
  // Only request 1.1 when we compiled in code that actually needs 1.1-core
  // symbols (e.g. vkGetPhysicalDeviceFeatures2 for INT8 DP feature query).
  // Otherwise request 1.0 so we work on older drivers / Android API levels
  // where libvulkan only exposes 1.0.
  // 1.1 gets us vkGetPhysicalDeviceFeatures2; cooperative matrix brings its
  // own extension + VK_KHR_vulkan_memory_model, so 1.1 is sufficient as the
  // instance version (MoltenVK 1.2 headers reject some older loaders).
#if defined(CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1) || defined(CLPEAK_VK_HAS_COMPUTE_MP_V1) || defined(CLPEAK_VK_HAS_COMPUTE_BF16_V1) || defined(CLPEAK_VK_HAS_ANY_COOPMAT)
  appInfo.apiVersion = VK_API_VERSION_1_1;
#else
  appInfo.apiVersion = VK_API_VERSION_1_0;
#endif

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
  log->print(NEWLINE "=== Vulkan backend ===" NEWLINE);
  if (!initInstance())
  {
    log->print("Vulkan: failed to create instance or no devices found" NEWLINE);
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
      std::stringstream ss;
      ss << "  Vulkan Device " << i << ": " << props.deviceName << " [" << typeStr << "]" << NEWLINE
         << "    API       : " << VK_VERSION_MAJOR(props.apiVersion) << "."
         << VK_VERSION_MINOR(props.apiVersion) << "."
         << VK_VERSION_PATCH(props.apiVersion) << NEWLINE;
      log->print(ss.str());
    }
    return 0;
  }

  // Mirror the OpenCL context stack so logger_android recordMetric() can
  // reach contextStack depth 4 (clpeak > platform > device > test_group).
  log->xmlOpenTag("clpeak");
  log->xmlAppendAttribs("os", OS_NAME);
  log->xmlOpenTag("platform");
  log->xmlAppendAttribs("name", "Vulkan");
  log->xmlAppendAttribs("backend", "Vulkan");

  for (size_t d = 0; d < physicalDevices.size(); d++)
  {
    VulkanDevice dev;
    if (!dev.init(instance, physicalDevices[d]))
    {
      log->print(NEWLINE "Vulkan: failed to init device " + std::to_string(d) + NEWLINE);
      continue;
    }

#ifdef CLPEAK_VK_HAS_ANY_COOPMAT
    // Enumerate cooperative-matrix properties to decide which dtype
    // combinations are advertised at the canonical 16x16x16 subgroup-scope
    // tile.  Done here (not in VulkanDevice::init) because the entry point
    // is an instance-level extension function that must be resolved via
    // vkGetInstanceProcAddr against the real VkInstance.
    if (dev.info.cooperativeMatrixSupported)
    {
      auto pfn = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
          vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
      uint32_t propCount = 0;
      if (pfn && pfn(physicalDevices[d], &propCount, nullptr) == VK_SUCCESS && propCount > 0)
      {
        std::vector<VkCooperativeMatrixPropertiesKHR> props(propCount);
        for (auto &p : props) p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        if (pfn(physicalDevices[d], &propCount, props.data()) == VK_SUCCESS)
        {
          // Tile matcher: subgroup-scope property at the requested MxNxK
          // with matching A/B input type and C/Result accumulator type.
          auto matches = [](const VkCooperativeMatrixPropertiesKHR &p,
                            uint32_t m, uint32_t n, uint32_t k,
                            VkComponentTypeKHR ab, VkComponentTypeKHR c) {
            return p.MSize == m && p.NSize == n && p.KSize == k &&
                   p.scope == VK_SCOPE_SUBGROUP_KHR &&
                   p.AType == ab && p.BType == ab &&
                   p.CType == c && p.ResultType == c;
          };
          for (auto &p : props)
          {
            if (matches(p, 16,16,16, VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR))
              dev.info.coopmatFP16Supported = true;
            if (matches(p, 16,16,16, VK_COMPONENT_TYPE_BFLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR))
              dev.info.coopmatBF16Supported = true;
            if (matches(p, 16,16,16, VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT, VK_COMPONENT_TYPE_FLOAT32_KHR))
              dev.info.coopmatFP8E4M3Supported = true;
            if (matches(p, 16,16,16, VK_COMPONENT_TYPE_FLOAT8_E5M2_EXT, VK_COMPONENT_TYPE_FLOAT32_KHR))
              dev.info.coopmatFP8E5M2Supported = true;
            // INT8: prefer K=16 (AMD/Intel) but fall back to K=32 (NVIDIA
            // Turing+ advertises INT8 tensor-core tiles only with K=32).
            if (dev.info.coopmatINT8K == 0 &&
                matches(p, 16,16,16, VK_COMPONENT_TYPE_SINT8_KHR, VK_COMPONENT_TYPE_SINT32_KHR))
              dev.info.coopmatINT8K = 16;
            if (dev.info.coopmatINT8K != 16 &&
                matches(p, 16,16,32, VK_COMPONENT_TYPE_SINT8_KHR, VK_COMPONENT_TYPE_SINT32_KHR))
              dev.info.coopmatINT8K = 32;
          }
        }
      }
    }
#endif

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
#ifdef CLPEAK_VK_HAS_COMPUTE_MP_V1
    runComputeMP(dev, cfg);
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
    runComputeInt8DP(dev, cfg);
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_INT4_PACKED_V1
    runComputeInt4Packed(dev, cfg);
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_BF16_V1
    runComputeBF16(dev, cfg);
#endif
#ifdef CLPEAK_VK_HAS_ANY_COOPMAT
    runCoopMatrix(dev, cfg);
#endif
    runGlobalBandwidth(dev, cfg);
    runLocalBandwidth(dev, cfg);
    runImageBandwidth(dev, cfg);
    runAtomicThroughput(dev, cfg);

    log->print(NEWLINE);
    log->xmlCloseTag(); // device
  }

  log->xmlCloseTag(); // platform
  log->xmlCloseTag(); // clpeak
  return 0;
}

// ---------------------------------------------------------------------------
// Shared compute-peak driver.
//
// Every compute-peak benchmark (runComputeSP / MP / INT8-DP / INT4-packed /
// coop-matrix / ...) shares the same Vulkan scaffolding: allocate a single
// device-local output buffer, build a one-binding descriptor set, create a
// pipeline from the shader's SPIR-V, dispatch repeatedly with a push
// constant, and report work-per-WI / elapsed time.  The only differences
// are the shader, the buffer-element size, the push-constant payload, and
// the strings used for display / XML.  All of those are bundled into
// vk_compute_desc_t so each concrete benchmark becomes a few-line wrapper.
// ---------------------------------------------------------------------------

int vkPeak::runComputeKernel(VulkanDevice &dev, benchmark_config_t &cfg,
                             const vk_compute_desc_t &d)
{
  log->print(NEWLINE TAB);
  log->print(d.title);
  log->print(NEWLINE);
  log->xmlOpenTag(d.xmlTag);
  log->xmlAppendAttribs("unit", d.unit);
  if (d.extraAttribKey && d.extraAttribVal)
    log->xmlAppendAttribs(d.extraAttribKey, d.extraAttribVal);

  if (d.skip)
  {
    log->print(TAB TAB);
    log->print(d.skipMsg ? d.skipMsg : "Skipped");
    log->print(NEWLINE);
    log->xmlCloseTag();
    return 0;
  }

  // Collect variants.  Multi-variant path (e.g. fp16 v1/v2/v4) shares one
  // buffer + descriptor set and swaps only the pipeline between dispatches;
  // single-variant benchmarks materialize a one-entry list.
  struct Variant { const char *label; const uint32_t *spirv; size_t spirvSize; };
  std::vector<Variant> variants;
  if (d.variants && d.numVariants > 0)
  {
    for (uint32_t i = 0; i < d.numVariants; i++)
      variants.push_back({d.variants[i].label, d.variants[i].spirv, d.variants[i].spirvSize});
  }
  else
  {
    variants.push_back({d.metricLabel, d.spirv, d.spirvSize});
  }

  // Size the dispatch to saturate the device and amortize submit overhead.
  // Vulkan doesn't expose a CU count, so target 32M work-items (matches
  // OpenCL's numCUs*2048*maxWGSize on most GPUs once clamped), bounded
  // by maxStorageBufferRange.  Cooperative-matrix shaders run one subgroup
  // per work-group (32 threads on NVIDIA / AMD RDNA3+ / Intel Arc); other
  // compute kernels use the classic 256.
  const uint32_t wgSize = d.wgSize ? d.wgSize : 256;
  const uint32_t outPerWG = d.outElemsPerWG ? d.outElemsPerWG : wgSize;
  uint64_t globalWIs = 32ULL * 1024 * 1024;
  // Buffer footprint = numGroups * outPerWG * elemSize.  Bound by allocation.
  uint64_t bytesPerWG = (uint64_t)outPerWG * d.elemSize;
  uint64_t maxWGs = dev.info.maxAllocSize / bytesPerWG;
  uint64_t wantWGs = globalWIs / wgSize;
  uint32_t numGroups = (uint32_t)std::min(wantWGs, maxWGs);
  globalWIs = (uint64_t)numGroups * wgSize;
  uint64_t bufferBytes = (uint64_t)numGroups * bytesPerWG;

  VkBuffer outputBuf;
  VkDeviceMemory outputMem;
  if (!dev.createBuffer(bufferBytes,
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
  pushRange.size = d.pushSize;

  VkPipelineLayoutCreateInfo pipeLayoutCI = {};
  pipeLayoutCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeLayoutCI.setLayoutCount = 1;
  pipeLayoutCI.pSetLayouts = &dsLayout;
  if (d.pushSize > 0)
  {
    pipeLayoutCI.pushConstantRangeCount = 1;
    pipeLayoutCI.pPushConstantRanges = &pushRange;
  }

  VkPipelineLayout pipeLayout;
  vkCreatePipelineLayout(dev.device, &pipeLayoutCI, nullptr, &pipeLayout);

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
  bufInfo.range = bufferBytes;

  VkWriteDescriptorSet write = {};
  write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write.dstSet = descSet;
  write.dstBinding = 0;
  write.descriptorCount = 1;
  write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write.pBufferInfo = &bufInfo;

  vkUpdateDescriptorSets(dev.device, 1, &write, 0, nullptr);

  // Build + dispatch each variant's pipeline.  Variants failing pipeline
  // creation are printed as skipped but don't abort the group -- some
  // drivers accept the v1 shader but choke on a wider packed variant.
  for (const auto &v : variants)
  {
    log->print(TAB TAB);
    log->print(v.label);
    log->print(" : ");

    VkPipeline pipeline;
    if (!dev.createComputePipeline(v.spirv, v.spirvSize, dsLayout, pipeLayout, pipeline))
    {
      log->print("pipeline creation failed (driver may not honor extension)" NEWLINE);
      continue;
    }

    float timed = runKernel(dev, pipeline, pipeLayout, descSet, numGroups,
                            cfg.computeIters, d.pushData, d.pushSize);
    float value = (static_cast<float>(globalWIs) * static_cast<float>(d.workPerWI)) / timed / 1e3f;

    log->print(value);
    log->print(NEWLINE);
    log->xmlRecord(v.label, value);

    vkDestroyPipeline(dev.device, pipeline, nullptr);
  }

  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, outputBuf, nullptr);
  vkFreeMemory(dev.device, outputMem, nullptr);

  log->xmlCloseTag();
  return 0;
}

// ---------------------------------------------------------------------------
// Concrete compute benchmarks.  Each is a thin wrapper that fills in a
// vk_compute_desc_t and delegates to runComputeKernel above.
// ---------------------------------------------------------------------------

int vkPeak::runComputeSP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  float A = 1.3f;
  vk_compute_desc_t d = {};
  d.title       = "Single-precision compute (GFLOPS)";
  d.xmlTag      = "single_precision_compute";
  d.metricLabel = "float";
  d.unit        = "gflops";
  d.spirv       = vk_shaders::compute_sp_v1;
  d.spirvSize   = vk_shaders::compute_sp_v1_size;
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

#ifdef CLPEAK_VK_HAS_COMPUTE_MP_V1
int vkPeak::runComputeMP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  // v1 = scalar fp16 (baseline; no HFMA2 packing).
  // v2 = f16vec2  (unlocks NVIDIA HFMA2 at 2x FP32 rate on shader cores).
  // v4 = f16vec4  (wider packing; informs AMD/Intel where issue rate
  //                exceeds two lanes per slot).
  static const vk_compute_variant_t variants[] = {
    { "mp",  vk_shaders::compute_mp_v1, vk_shaders::compute_mp_v1_size },
#ifdef CLPEAK_VK_HAS_COMPUTE_MP_V2
    { "mp2", vk_shaders::compute_mp_v2, vk_shaders::compute_mp_v2_size },
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_MP_V4
    { "mp4", vk_shaders::compute_mp_v4, vk_shaders::compute_mp_v4_size },
#endif
  };
  float A = 1.3f;
  vk_compute_desc_t d = {};
  d.title       = "Mixed-precision compute fp16xfp16+fp32 (GFLOPS)";
  d.xmlTag      = "mixed_precision_compute";
  d.unit        = "gflops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  d.skip        = !dev.info.float16Supported;
  d.skipMsg     = "shaderFloat16 not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}
#endif

#ifdef CLPEAK_VK_HAS_COMPUTE_INT4_PACKED_V1
int vkPeak::runComputeInt4Packed(VulkanDevice &dev, benchmark_config_t &cfg)
{
  int32_t A = 3;
  vk_compute_desc_t d = {};
  d.title           = "Packed INT4 compute (emulated) (GOPS)";
  d.xmlTag          = "int4_packed_compute";
  d.metricLabel     = "int4_packed";
  d.unit            = "gops";
  d.spirv           = vk_shaders::compute_int4_packed_v1;
  d.spirvSize       = vk_shaders::compute_int4_packed_v1_size;
  d.workPerWI       = COMPUTE_INT4_PACKED_WORK_PER_WI;
  d.elemSize        = sizeof(int32_t);
  d.pushData        = &A;
  d.pushSize        = sizeof(A);
  d.extraAttribKey  = "emulated";
  d.extraAttribVal  = "true";
  return runComputeKernel(dev, cfg, d);
}
#endif

#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V1
int vkPeak::runComputeInt8DP(VulkanDevice &dev, benchmark_config_t &cfg)
{
  // v1 = single dp4a chain (serial through REPACK; dep-stall bound).
  // v2 = two parallel dp4a chains (double the independent work for the
  //       instruction issue queue to pipeline).
  // v4 = four parallel chains (enough to saturate dp4a issue rate on
  //       NVIDIA Turing+ / AMD RDNA2+ / Intel Xe+).
  static const vk_compute_variant_t variants[] = {
    { "int8_dp",  vk_shaders::compute_int8_dp_v1, vk_shaders::compute_int8_dp_v1_size },
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V2
    { "int8_dp2", vk_shaders::compute_int8_dp_v2, vk_shaders::compute_int8_dp_v2_size },
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_INT8_DP_V4
    { "int8_dp4", vk_shaders::compute_int8_dp_v4, vk_shaders::compute_int8_dp_v4_size },
#endif
  };
  int32_t A = 4;
  vk_compute_desc_t d = {};
  d.title       = "INT8 dot-product compute (GOPS)";
  d.xmlTag      = "integer_compute_int8_dp";
  d.unit        = "gops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_INT8_DP_WORK_PER_WI;
  d.elemSize    = sizeof(int32_t);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  d.skip        = !dev.info.int8DotProductSupported;
  d.skipMsg     = "VK_KHR_shader_integer_dot_product / shaderInt8 not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}
#endif

#ifdef CLPEAK_VK_HAS_COMPUTE_BF16_V1
int vkPeak::runComputeBF16(VulkanDevice &dev, benchmark_config_t &cfg)
{
  // v1 / v2 / v4: same packing story as MP.  NVIDIA shader-core BF16
  // peaks at bf16vec2 via BMMA2-style packed multiply.
  static const vk_compute_variant_t variants[] = {
    { "bf16",  vk_shaders::compute_bf16_v1, vk_shaders::compute_bf16_v1_size },
#ifdef CLPEAK_VK_HAS_COMPUTE_BF16_V2
    { "bf16_2", vk_shaders::compute_bf16_v2, vk_shaders::compute_bf16_v2_size },
#endif
#ifdef CLPEAK_VK_HAS_COMPUTE_BF16_V4
    { "bf16_4", vk_shaders::compute_bf16_v4, vk_shaders::compute_bf16_v4_size },
#endif
  };
  float A = 1.3f;
  vk_compute_desc_t d = {};
  d.title       = "BF16 compute bf16xbf16+fp32 (GFLOPS)";
  d.xmlTag      = "bfloat16_compute";
  d.unit        = "gflops";
  d.variants    = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
  d.elemSize    = sizeof(float);
  d.pushData    = &A;
  d.pushSize    = sizeof(A);
  d.skip        = !dev.info.bfloat16Supported;
  d.skipMsg     = "VK_KHR_shader_bfloat16 / shaderBFloat16Type not supported! Skipped";
  return runComputeKernel(dev, cfg, d);
}
#endif

// ---------------------------------------------------------------------------
// Cooperative matrix (tensor-core) umbrella.
//
// Runs every dtype combination the driver advertises at the canonical
// 16x16x16 subgroup-scope tile (queried once in VulkanDevice::init).  Each
// dtype shares the same scaffolding via runComputeKernel -- only the shader,
// buffer element type, and label strings differ.  Adding FP8 / INT4 in
// Phase 2 reduces to: compile a coopmat_fp8.comp, query the matching
// component-type enums, and add one more entry here.
// ---------------------------------------------------------------------------

int vkPeak::runCoopMatrix(VulkanDevice &dev, benchmark_config_t &cfg)
{
  if (!dev.info.cooperativeMatrixSupported)
  {
    log->print(NEWLINE TAB "Cooperative matrix (GFLOPS/GOPS)" NEWLINE);
    log->xmlOpenTag("cooperative_matrix");
    log->xmlAppendAttribs("tile", "16x16x16");
    log->print(TAB TAB "VK_KHR_cooperative_matrix not supported! Skipped" NEWLINE);
    log->xmlCloseTag();
    return 0;
  }

  // Coopmat shape constants: shaders hard-code 16x16x16 with 256 iters and
  // local_size_x=32 (one subgroup per work-group).  See COOPMAT_WORK_PER_WI.
  const uint32_t coopWGSize  = 32;
  const uint32_t coopOutElems = 16 * 16;  // M*N tile written per WG
  const uint32_t coopWork    = COOPMAT_WORK_PER_WI;

#ifdef CLPEAK_VK_HAS_COOPMAT_FP16
  {
    float A = 1.3f;
    vk_compute_desc_t d = {};
    d.title          = "Cooperative-matrix fp16xfp16+fp32 16x16x16 (GFLOPS)";
    d.xmlTag         = "coopmat_fp16";
    d.metricLabel    = "coopmat_fp16";
    d.unit           = "gflops";
    d.spirv          = vk_shaders::coopmat_fp16;
    d.spirvSize      = vk_shaders::coopmat_fp16_size;
    d.workPerWI      = coopWork;
    d.elemSize       = sizeof(float);
    d.wgSize         = coopWGSize;
    d.outElemsPerWG  = coopOutElems;
    d.pushData       = &A;
    d.pushSize       = sizeof(A);
    d.skip           = !dev.info.coopmatFP16Supported;
    d.skipMsg        = "No 16x16x16 fp16xfp16+fp32 coopmat property! Skipped";
    d.extraAttribKey = "tile";
    d.extraAttribVal = "16x16x16";
    runComputeKernel(dev, cfg, d);
  }
#endif
#ifdef CLPEAK_VK_HAS_COOPMAT_BF16
  {
    float A = 1.3f;
    vk_compute_desc_t d = {};
    d.title          = "Cooperative-matrix bf16xbf16+fp32 16x16x16 (GFLOPS)";
    d.xmlTag         = "coopmat_bf16";
    d.metricLabel    = "coopmat_bf16";
    d.unit           = "gflops";
    d.spirv          = vk_shaders::coopmat_bf16;
    d.spirvSize      = vk_shaders::coopmat_bf16_size;
    d.workPerWI      = coopWork;
    d.elemSize       = sizeof(float);
    d.wgSize         = coopWGSize;
    d.outElemsPerWG  = coopOutElems;
    d.pushData       = &A;
    d.pushSize       = sizeof(A);
    d.skip           = !dev.info.coopmatBF16Supported;
    d.skipMsg        = "No 16x16x16 bf16xbf16+fp32 coopmat property! Skipped";
    d.extraAttribKey = "tile";
    d.extraAttribVal = "16x16x16";
    runComputeKernel(dev, cfg, d);
  }
#endif
#ifdef CLPEAK_VK_HAS_COOPMAT_FP8_E4M3
  {
    float A = 1.3f;
    vk_compute_desc_t d = {};
    d.title          = "Cooperative-matrix fp8(E4M3)xfp8(E4M3)+fp32 16x16x16 (GFLOPS)";
    d.xmlTag         = "coopmat_fp8_e4m3";
    d.metricLabel    = "coopmat_fp8_e4m3";
    d.unit           = "gflops";
    d.spirv          = vk_shaders::coopmat_fp8_e4m3;
    d.spirvSize      = vk_shaders::coopmat_fp8_e4m3_size;
    d.workPerWI      = coopWork;
    d.elemSize       = sizeof(float);
    d.wgSize         = coopWGSize;
    d.outElemsPerWG  = coopOutElems;
    d.pushData       = &A;
    d.pushSize       = sizeof(A);
    d.skip           = !(dev.info.fp8Supported && dev.info.coopmatFP8E4M3Supported);
    d.skipMsg        = "No fp8-E4M3 coopmat support (VK_EXT_shader_float8 or property)! Skipped";
    d.extraAttribKey = "tile";
    d.extraAttribVal = "16x16x16";
    runComputeKernel(dev, cfg, d);
  }
#endif
#ifdef CLPEAK_VK_HAS_COOPMAT_FP8_E5M2
  {
    float A = 1.3f;
    vk_compute_desc_t d = {};
    d.title          = "Cooperative-matrix fp8(E5M2)xfp8(E5M2)+fp32 16x16x16 (GFLOPS)";
    d.xmlTag         = "coopmat_fp8_e5m2";
    d.metricLabel    = "coopmat_fp8_e5m2";
    d.unit           = "gflops";
    d.spirv          = vk_shaders::coopmat_fp8_e5m2;
    d.spirvSize      = vk_shaders::coopmat_fp8_e5m2_size;
    d.workPerWI      = coopWork;
    d.elemSize       = sizeof(float);
    d.wgSize         = coopWGSize;
    d.outElemsPerWG  = coopOutElems;
    d.pushData       = &A;
    d.pushSize       = sizeof(A);
    d.skip           = !(dev.info.fp8Supported && dev.info.coopmatFP8E5M2Supported);
    d.skipMsg        = "No fp8-E5M2 coopmat support (VK_EXT_shader_float8 or property)! Skipped";
    d.extraAttribKey = "tile";
    d.extraAttribVal = "16x16x16";
    runComputeKernel(dev, cfg, d);
  }
#endif
#if defined(CLPEAK_VK_HAS_COOPMAT_INT8) || defined(CLPEAK_VK_HAS_COOPMAT_INT8_K32)
  {
    // Select the shader variant matching whichever INT8 tile the driver
    // advertised.  K=16 is the generic path; NVIDIA tensor cores need K=32.
    int32_t A = 3;
    vk_compute_desc_t d = {};
    d.xmlTag         = "coopmat_int8";
    d.metricLabel    = "coopmat_int8";
    d.unit           = "gops";
    d.workPerWI      = coopWork;
    d.elemSize       = sizeof(int32_t);
    d.wgSize         = coopWGSize;
    d.outElemsPerWG  = coopOutElems;
    d.pushData       = &A;
    d.pushSize       = sizeof(A);
    d.extraAttribKey = "tile";

    const char *titleK16 = "Cooperative-matrix int8xint8+int32 16x16x16 (GOPS)";
    const char *titleK32 = "Cooperative-matrix int8xint8+int32 16x16x32 (GOPS)";
    bool haveShaderK16 = false, haveShaderK32 = false;
#ifdef CLPEAK_VK_HAS_COOPMAT_INT8
    haveShaderK16 = true;
#endif
#ifdef CLPEAK_VK_HAS_COOPMAT_INT8_K32
    haveShaderK32 = true;
#endif

    if (dev.info.coopmatINT8K == 16 && haveShaderK16)
    {
#ifdef CLPEAK_VK_HAS_COOPMAT_INT8
      d.title          = titleK16;
      d.spirv          = vk_shaders::coopmat_int8;
      d.spirvSize      = vk_shaders::coopmat_int8_size;
      d.extraAttribVal = "16x16x16";
#endif
    }
    else if (dev.info.coopmatINT8K == 32 && haveShaderK32)
    {
#ifdef CLPEAK_VK_HAS_COOPMAT_INT8_K32
      d.title          = titleK32;
      d.spirv          = vk_shaders::coopmat_int8_k32;
      d.spirvSize      = vk_shaders::coopmat_int8_k32_size;
      d.extraAttribVal = "16x16x32";
#endif
    }
    else
    {
      // Driver advertised neither 16x16x16 nor 16x16x32 INT8 -- or the
      // corresponding shader didn't compile in this build.  Skip with a
      // label that names the probed tiles so the reason is obvious.
      d.title          = titleK16;
      d.skip           = true;
      d.skipMsg        = "No 16x16x{16,32} int8xint8+int32 coopmat property! Skipped";
      d.extraAttribVal = "16x16x16";
      d.spirv          = nullptr;
      d.spirvSize      = 0;
    }
    runComputeKernel(dev, cfg, d);
  }
#endif
  return 0;
}

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

// ---------------------------------------------------------------------------
// Local memory bandwidth (Vulkan -- shared memory)
// ---------------------------------------------------------------------------
//
// Same single-output-buffer scaffolding as runComputeKernel; only the bytes-
// per-WI calculation differs per variant.  Width = 1/2/4/8 floats per slot.

int vkPeak::runLocalBandwidth(VulkanDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = cfg.computeIters;

  log->print(NEWLINE TAB "Local memory bandwidth (GBPS)" NEWLINE);
  log->xmlOpenTag("local_memory_bandwidth");
  log->xmlAppendAttribs("unit", "gbps");

  const uint32_t wgSize = 256;
  uint64_t globalWIs = 32ULL * 1024 * 1024;
  uint32_t numGroups = (uint32_t)(globalWIs / wgSize);
  uint64_t bufferBytes = (uint64_t)globalWIs * sizeof(float);

  VkBuffer outBuf;
  VkDeviceMemory outMem;
  if (!dev.createBuffer(bufferBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuf, outMem))
  {
    log->print(TAB TAB "Failed to allocate buffer" NEWLINE);
    log->xmlCloseTag();
    return -1;
  }

  // Descriptor set layout: 1 storage buffer.
  VkDescriptorSetLayoutBinding binding = {};
  binding.binding = 0;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  binding.descriptorCount = 1;
  binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutCreateInfo dslCI = {};
  dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslCI.bindingCount = 1; dslCI.pBindings = &binding;
  VkDescriptorSetLayout dsLayout;
  vkCreateDescriptorSetLayout(dev.device, &dslCI, nullptr, &dsLayout);

  VkPipelineLayoutCreateInfo plCI = {};
  plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plCI.setLayoutCount = 1; plCI.pSetLayouts = &dsLayout;
  VkPipelineLayout pipeLayout;
  vkCreatePipelineLayout(dev.device, &plCI, nullptr, &pipeLayout);

  VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
  VkDescriptorPoolCreateInfo dpCI = {};
  dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpCI.maxSets = 1; dpCI.poolSizeCount = 1; dpCI.pPoolSizes = &ps;
  VkDescriptorPool descPool;
  vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &descPool);

  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = descPool; dsAI.descriptorSetCount = 1; dsAI.pSetLayouts = &dsLayout;
  VkDescriptorSet descSet;
  vkAllocateDescriptorSets(dev.device, &dsAI, &descSet);

  VkDescriptorBufferInfo bi = {outBuf, 0, bufferBytes};
  VkWriteDescriptorSet w = {};
  w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w.dstSet = descSet; w.dstBinding = 0; w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
  vkUpdateDescriptorSets(dev.device, 1, &w, 0, nullptr);

  struct V { const char *label; const uint32_t *spv; size_t sz; uint32_t width; };
  const V variants[] = {
    {"float  ", vk_shaders::local_bandwidth_v1, vk_shaders::local_bandwidth_v1_size, 1},
    {"float2 ", vk_shaders::local_bandwidth_v2, vk_shaders::local_bandwidth_v2_size, 2},
    {"float4 ", vk_shaders::local_bandwidth_v4, vk_shaders::local_bandwidth_v4_size, 4},
    {"float8 ", vk_shaders::local_bandwidth_v8, vk_shaders::local_bandwidth_v8_size, 8},
  };
  for (const auto &v : variants)
  {
    log->print(TAB TAB);
    log->print(v.label);
    log->print(": ");
    VkPipeline pipe;
    if (!dev.createComputePipeline(v.spv, v.sz, dsLayout, pipeLayout, pipe))
    {
      log->print("pipeline failed" NEWLINE);
      continue;
    }
    float us = runKernel(dev, pipe, pipeLayout, descSet, numGroups, iters);
    // Each rep: 1 write + 1 read per WI = 2 * width * sizeof(float) bytes.
    uint64_t bytes = (uint64_t)LMEM_REPS * 2 * v.width * sizeof(float) * globalWIs;
    float gbps = (float)bytes / us / 1e3f;
    log->print(gbps);
    log->print(NEWLINE);
    // strip padding from label for the xml record key
    std::string key(v.label);
    while (!key.empty() && key.back() == ' ') key.pop_back();
    log->xmlRecord(key, gbps);
    vkDestroyPipeline(dev.device, pipe, nullptr);
  }

  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, outBuf, nullptr);
  vkFreeMemory(dev.device, outMem, nullptr);
  log->xmlCloseTag();
  return 0;
}

// ---------------------------------------------------------------------------
// Image (texture) bandwidth (Vulkan)
// ---------------------------------------------------------------------------
//
// Combined image-sampler descriptor + storage-buffer output.  Image is
// VK_FORMAT_R32G32B32A32_SFLOAT, sampled with NEAREST + CLAMP_TO_EDGE.

int vkPeak::runImageBandwidth(VulkanDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = cfg.globalBWIters;

  log->print(NEWLINE TAB "Image memory bandwidth (GBPS)" NEWLINE);
  log->xmlOpenTag("image_memory_bandwidth");
  log->xmlAppendAttribs("unit", "gbps");

  const uint32_t imgW = 4096, imgH = 4096;
  const uint32_t wgSize = 256;
  uint64_t globalWIs = 32ULL * 1024 * 1024;
  uint32_t numGroups = (uint32_t)(globalWIs / wgSize);
  uint64_t outBytes  = globalWIs * sizeof(float);

  // Create image (RGBA32F, sampled, transfer-dst so we can clear it).
  VkImageCreateInfo imgCI = {};
  imgCI.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imgCI.imageType = VK_IMAGE_TYPE_2D;
  imgCI.format    = VK_FORMAT_R32G32B32A32_SFLOAT;
  imgCI.extent    = {imgW, imgH, 1};
  imgCI.mipLevels = 1; imgCI.arrayLayers = 1;
  imgCI.samples   = VK_SAMPLE_COUNT_1_BIT;
  imgCI.tiling    = VK_IMAGE_TILING_OPTIMAL;
  imgCI.usage     = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  imgCI.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  imgCI.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImage img;
  if (vkCreateImage(dev.device, &imgCI, nullptr, &img) != VK_SUCCESS)
  {
    log->print(TAB TAB "Image create failed" NEWLINE);
    log->xmlCloseTag();
    return -1;
  }

  // Allocate device-local memory for the image.
  VkMemoryRequirements imReq;
  vkGetImageMemoryRequirements(dev.device, img, &imReq);
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(dev.physicalDevice, &memProps);
  uint32_t typeIdx = UINT32_MAX;
  for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
    if ((imReq.memoryTypeBits & (1u << i)) &&
        (memProps.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
    { typeIdx = i; break; }

  VkMemoryAllocateInfo aI = {};
  aI.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  aI.allocationSize = imReq.size;
  aI.memoryTypeIndex = typeIdx;
  VkDeviceMemory imgMem;
  vkAllocateMemory(dev.device, &aI, nullptr, &imgMem);
  vkBindImageMemory(dev.device, img, imgMem, 0);

  // Transition UNDEFINED -> SHADER_READ_ONLY_OPTIMAL.  We don't need to
  // upload any data; the image contents being unspecified is fine for a
  // bandwidth measurement (the cache lines still get fetched).
  VkCommandBufferAllocateInfo cbAI = {};
  cbAI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cbAI.commandPool = dev.commandPool;
  cbAI.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cbAI.commandBufferCount = 1;
  VkCommandBuffer transCmd;
  vkAllocateCommandBuffers(dev.device, &cbAI, &transCmd);
  VkCommandBufferBeginInfo cbBI = {};
  cbBI.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  cbBI.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(transCmd, &cbBI);
  VkImageMemoryBarrier b = {};
  b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  b.image = img;
  b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  b.srcAccessMask = 0;
  b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(transCmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                       0, nullptr, 0, nullptr, 1, &b);
  vkEndCommandBuffer(transCmd);
  dev.submitAndWait(transCmd);
  vkFreeCommandBuffers(dev.device, dev.commandPool, 1, &transCmd);

  VkImageViewCreateInfo ivCI = {};
  ivCI.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  ivCI.image = img; ivCI.viewType = VK_IMAGE_VIEW_TYPE_2D;
  ivCI.format = VK_FORMAT_R32G32B32A32_SFLOAT;
  ivCI.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  VkImageView imgView;
  vkCreateImageView(dev.device, &ivCI, nullptr, &imgView);

  VkSamplerCreateInfo smCI = {};
  smCI.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  smCI.magFilter = VK_FILTER_NEAREST;
  smCI.minFilter = VK_FILTER_NEAREST;
  smCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  smCI.addressModeU = smCI.addressModeV = smCI.addressModeW =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  smCI.unnormalizedCoordinates = VK_FALSE;
  VkSampler sampler;
  vkCreateSampler(dev.device, &smCI, nullptr, &sampler);

  VkBuffer outBuf; VkDeviceMemory outMem;
  if (!dev.createBuffer(outBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, outBuf, outMem))
  {
    log->print(TAB TAB "Output buffer alloc failed" NEWLINE);
    vkDestroySampler(dev.device, sampler, nullptr);
    vkDestroyImageView(dev.device, imgView, nullptr);
    vkDestroyImage(dev.device, img, nullptr);
    vkFreeMemory(dev.device, imgMem, nullptr);
    log->xmlCloseTag();
    return -1;
  }

  VkDescriptorSetLayoutBinding bs[2] = {};
  bs[0].binding = 0; bs[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  bs[0].descriptorCount = 1; bs[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  bs[1].binding = 1; bs[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  bs[1].descriptorCount = 1; bs[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  VkDescriptorSetLayoutCreateInfo dslCI = {};
  dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  dslCI.bindingCount = 2; dslCI.pBindings = bs;
  VkDescriptorSetLayout dsLayout;
  vkCreateDescriptorSetLayout(dev.device, &dslCI, nullptr, &dsLayout);

  VkPipelineLayoutCreateInfo plCI = {};
  plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  plCI.setLayoutCount = 1; plCI.pSetLayouts = &dsLayout;
  VkPipelineLayout pipeLayout;
  vkCreatePipelineLayout(dev.device, &plCI, nullptr, &pipeLayout);

  VkDescriptorPoolSize ps[2] = {
    {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
  };
  VkDescriptorPoolCreateInfo dpCI = {};
  dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  dpCI.maxSets = 1; dpCI.poolSizeCount = 2; dpCI.pPoolSizes = ps;
  VkDescriptorPool descPool;
  vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &descPool);

  VkDescriptorSetAllocateInfo dsAI = {};
  dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  dsAI.descriptorPool = descPool; dsAI.descriptorSetCount = 1; dsAI.pSetLayouts = &dsLayout;
  VkDescriptorSet descSet;
  vkAllocateDescriptorSets(dev.device, &dsAI, &descSet);

  VkDescriptorImageInfo ii = {};
  ii.imageView = imgView; ii.sampler = sampler;
  ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  VkDescriptorBufferInfo bi = {outBuf, 0, outBytes};
  VkWriteDescriptorSet ws[2] = {};
  ws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  ws[0].dstSet = descSet; ws[0].dstBinding = 0; ws[0].descriptorCount = 1;
  ws[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  ws[0].pImageInfo = &ii;
  ws[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  ws[1].dstSet = descSet; ws[1].dstBinding = 1; ws[1].descriptorCount = 1;
  ws[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  ws[1].pBufferInfo = &bi;
  vkUpdateDescriptorSets(dev.device, 2, ws, 0, nullptr);

  VkPipeline pipe;
  bool ok = dev.createComputePipeline(vk_shaders::image_bandwidth_v1,
      vk_shaders::image_bandwidth_v1_size, dsLayout, pipeLayout, pipe);
  log->print(TAB TAB "float4 : ");
  if (!ok)
  {
    log->print("pipeline failed" NEWLINE);
  }
  else
  {
    float us = runKernel(dev, pipe, pipeLayout, descSet, numGroups, iters);
    uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalWIs;
    float gbps = (float)bytes / us / 1e3f;
    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("float4", gbps);
    vkDestroyPipeline(dev.device, pipe, nullptr);
  }

  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, outBuf, nullptr);
  vkFreeMemory(dev.device, outMem, nullptr);
  vkDestroySampler(dev.device, sampler, nullptr);
  vkDestroyImageView(dev.device, imgView, nullptr);
  vkDestroyImage(dev.device, img, nullptr);
  vkFreeMemory(dev.device, imgMem, nullptr);
  log->xmlCloseTag();
  return 0;
}

// ---------------------------------------------------------------------------
// Atomic throughput (Vulkan -- global + local atomics)
// ---------------------------------------------------------------------------

int vkPeak::runAtomicThroughput(VulkanDevice &dev, benchmark_config_t &cfg)
{
  unsigned int iters = cfg.computeIters;

  log->print(NEWLINE TAB "Atomic throughput (GOPS)" NEWLINE);
  log->xmlOpenTag("atomic_throughput");
  log->xmlAppendAttribs("unit", "gops");

  const uint32_t wgSize = 256;
  uint64_t globalWIs = 32ULL * 1024 * 1024;
  uint32_t numGroups = (uint32_t)(globalWIs / wgSize);

  // Helper: allocate a single-storage-buffer descriptor + dispatch + time.
  auto runOne = [&](const char *label, const uint32_t *spv, size_t spvSize,
                    uint64_t bufBytes) -> float
  {
    VkBuffer buf; VkDeviceMemory mem;
    if (!dev.createBuffer(bufBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buf, mem))
      return -1.0f;
    VkDescriptorSetLayoutBinding b = {};
    b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslCI = {};
    dslCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslCI.bindingCount = 1; dslCI.pBindings = &b;
    VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev.device, &dslCI, nullptr, &dsl);
    VkPipelineLayoutCreateInfo plCI = {};
    plCI.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plCI.setLayoutCount = 1; plCI.pSetLayouts = &dsl;
    VkPipelineLayout pl; vkCreatePipelineLayout(dev.device, &plCI, nullptr, &pl);
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpCI = {};
    dpCI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpCI.maxSets = 1; dpCI.poolSizeCount = 1; dpCI.pPoolSizes = &ps;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev.device, &dpCI, nullptr, &dp);
    VkDescriptorSetAllocateInfo dsAI = {};
    dsAI.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsAI.descriptorPool = dp; dsAI.descriptorSetCount = 1; dsAI.pSetLayouts = &dsl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(dev.device, &dsAI, &ds);
    VkDescriptorBufferInfo bi = {buf, 0, bufBytes};
    VkWriteDescriptorSet w = {};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = ds; w.dstBinding = 0; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(dev.device, 1, &w, 0, nullptr);

    VkPipeline pipe;
    if (!dev.createComputePipeline(spv, spvSize, dsl, pl, pipe))
    {
      log->print("pipeline failed" NEWLINE);
      vkDestroyDescriptorPool(dev.device, dp, nullptr);
      vkDestroyPipelineLayout(dev.device, pl, nullptr);
      vkDestroyDescriptorSetLayout(dev.device, dsl, nullptr);
      vkDestroyBuffer(dev.device, buf, nullptr);
      vkFreeMemory(dev.device, mem, nullptr);
      return -1.0f;
    }
    float us = runKernel(dev, pipe, pl, ds, numGroups, iters);
    vkDestroyPipeline(dev.device, pipe, nullptr);
    vkDestroyDescriptorPool(dev.device, dp, nullptr);
    vkDestroyPipelineLayout(dev.device, pl, nullptr);
    vkDestroyDescriptorSetLayout(dev.device, dsl, nullptr);
    vkDestroyBuffer(dev.device, buf, nullptr);
    vkFreeMemory(dev.device, mem, nullptr);
    (void)label;
    return us;
  };

  // Global atomics: one int counter per WI -> 128 MB output.
  log->print(TAB TAB "global : ");
  float us_g = runOne("global", vk_shaders::atomic_throughput_global,
      vk_shaders::atomic_throughput_global_size, globalWIs * sizeof(int32_t));
  if (us_g > 0)
  {
    float gops = ((float)globalWIs * (float)ATOMIC_REPS) / us_g / 1e3f;
    log->print(gops); log->print(NEWLINE);
    log->xmlRecord("global", gops);
  }

  // Local atomics: one int counter per workgroup.
  log->print(TAB TAB "local  : ");
  float us_l = runOne("local", vk_shaders::atomic_throughput_local,
      vk_shaders::atomic_throughput_local_size, (uint64_t)numGroups * sizeof(int32_t));
  if (us_l > 0)
  {
    float gops = ((float)globalWIs * (float)ATOMIC_REPS) / us_l / 1e3f;
    log->print(gops); log->print(NEWLINE);
    log->xmlRecord("local", gops);
  }

  log->xmlCloseTag();
  return 0;
}

#endif // ENABLE_VULKAN
