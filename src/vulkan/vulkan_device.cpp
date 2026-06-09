#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <cstring>
#include <functional>
#include <vector>

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

// ---------------------------------------------------------------------------
// Step 1: basic device properties, memory heaps, queue family discovery
// ---------------------------------------------------------------------------
static bool queryBasicInfo(VulkanDevice *self, VkPhysicalDevice physDev)
{
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(physDev, &props);

  self->info.deviceName = props.deviceName;
  self->info.apiVersion = std::to_string(VK_VERSION_MAJOR(props.apiVersion)) + "." +
                          std::to_string(VK_VERSION_MINOR(props.apiVersion)) + "." +
                          std::to_string(VK_VERSION_PATCH(props.apiVersion));
  self->info.driverVersion = std::to_string(VK_VERSION_MAJOR(props.driverVersion)) + "." +
                             std::to_string(VK_VERSION_MINOR(props.driverVersion)) + "." +
                             std::to_string(VK_VERSION_PATCH(props.driverVersion));
  self->info.maxWGSize = std::min(props.limits.maxComputeWorkGroupSize[0], (uint32_t)MAX_WG_SIZE);
  self->info.maxAllocSize = props.limits.maxStorageBufferRange;
  self->info.vkDeviceType = props.deviceType;

  // Set neutral DeviceType
  if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
      self->info.deviceType = DeviceType::Cpu;
  else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
           props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      self->info.deviceType = DeviceType::Gpu;
  else
      self->info.deviceType = DeviceType::Unknown;

  self->info.numCUs = 0;

  // Get memory heap size (device-local)
  VkPhysicalDeviceMemoryProperties memProps;
  vkGetPhysicalDeviceMemoryProperties(physDev, &memProps);
  self->info.heapSize = 0;
  for (uint32_t i = 0; i < memProps.memoryHeapCount; i++)
  {
    if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
    {
      self->info.heapSize = std::max(self->info.heapSize, (uint64_t)memProps.memoryHeaps[i].size);
    }
  }

  // Find compute queue family
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(physDev, &queueFamilyCount, queueFamilies.data());

  self->info.computeQueueFamily = UINT32_MAX;
  for (uint32_t i = 0; i < queueFamilyCount; i++)
  {
    if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
    {
      self->info.computeQueueFamily = i;
      break;
    }
  }
  if (self->info.computeQueueFamily == UINT32_MAX)
    return false;

  return true;
}

// ---------------------------------------------------------------------------
// Step 2: probe vendor extension properties for CU/SM count
// ---------------------------------------------------------------------------
static void probeComputeUnitCount(VulkanDevice *self, VkPhysicalDevice physDev,
                                  const std::function<bool(const char*)> &hasExt)
{
  VkPhysicalDeviceProperties2 p2 = {};
  p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

  if (hasExt("VK_NV_shader_sm_builtins"))
  {
    VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smProps = {};
    smProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV;
    p2.pNext = &smProps;
    vkGetPhysicalDeviceProperties2(physDev, &p2);
    self->info.numCUs = smProps.shaderSMCount;
  }
  else if (hasExt("VK_AMD_shader_core_properties2"))
  {
    VkPhysicalDeviceShaderCoreProperties2AMD coreProps2 = {};
    coreProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD;
    p2.pNext = &coreProps2;
    vkGetPhysicalDeviceProperties2(physDev, &p2);
    self->info.numCUs = coreProps2.activeComputeUnitCount;
  }
  else if (hasExt("VK_AMD_shader_core_properties"))
  {
    VkPhysicalDeviceShaderCorePropertiesAMD coreProps = {};
    coreProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;
    p2.pNext = &coreProps;
    vkGetPhysicalDeviceProperties2(physDev, &p2);
    self->info.numCUs = coreProps.shaderEngineCount *
                        coreProps.shaderArraysPerEngineCount *
                        coreProps.computeUnitsPerShaderArray;
  }
  else if (hasExt("VK_ARM_shader_core_builtins"))
  {
    VkPhysicalDeviceShaderCoreBuiltinsPropertiesARM armProps = {};
    armProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES_ARM;
    p2.pNext = &armProps;
    vkGetPhysicalDeviceProperties2(physDev, &p2);
    self->info.numCUs = armProps.shaderCoreCount;
  }
}

// Small helper: prepend a features struct onto a pNext chain.
static void *chainPNext(void *head, void *newNode)
{
  reinterpret_cast<VkBaseOutStructure *>(newNode)->pNext =
      reinterpret_cast<VkBaseOutStructure *>(head);
  return newNode;
}

// ---------------------------------------------------------------------------
// Step 3: query optional shader features and collect enabled extensions
// ---------------------------------------------------------------------------
static void queryOptionalFeatures(VulkanDevice *self, VkPhysicalDevice physDev,
                                  VkInstance inst,
                                  const std::function<bool(const char*)> &hasExt,
                                  std::vector<const char *> &enabledExts)
{
  // Reset all feature gates
  self->info.int8DotProductSupported = false;
  self->info.float16Supported = false;
  self->info.float64Supported = false;
  self->info.bfloat16Supported = false;
  self->info.cooperativeMatrixSupported = false;
  self->info.fp8Supported = false;
  self->info.coopmatFP32 = {};
  self->info.coopmatFP16 = {};
  self->info.coopmatBF16 = {};
  self->info.coopmatFP8E4M3 = {};
  self->info.coopmatFP8E5M2 = {};
  self->info.coopmatINT8 = {};
  self->info.calibratedTimestampsSupported = false;

#if defined(VK_HAS_COMPUTE_INT8_DP_V1) || defined(VK_HAS_COMPUTE_MP_V1) || defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_ANY_COOPMAT) || defined(VK_HAS_COMPUTE_DP_V1)
  VkPhysicalDeviceShaderFloat16Int8FeaturesKHR f16i8Features = {};
  f16i8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
#ifdef VK_HAS_COMPUTE_INT8_DP_V1
  VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dpFeatures = {};
  dpFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
#endif
#if defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_COOPMAT_BF16)
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16Features = {};
  bf16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
#endif
#ifdef VK_HAS_ANY_COOPMAT
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmatFeatures = {};
  coopmatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  VkPhysicalDeviceVulkanMemoryModelFeatures vmmFeatures = {};
  vmmFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
#ifdef VK_HAS_ANY_COOPMAT_FP8
  VkPhysicalDeviceShaderFloat8FeaturesEXT fp8Features = {};
  fp8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
#endif
#endif
  VkPhysicalDeviceFeatures2 baseFeatures2 = {};
  baseFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

  bool f16i8ExtEnabled = false;
  bool hasF16I8Ext = hasExt(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
#ifdef VK_HAS_ANY_COOPMAT
  bool hasCoopmatExt = false;
#endif
#ifdef VK_HAS_ANY_COOPMAT_FP8
  bool hasFP8Ext = false;
#endif
  {
    void *chain = nullptr;
    if (hasF16I8Ext)
      chain = chainPNext(chain, &f16i8Features);
#ifdef VK_HAS_COMPUTE_INT8_DP_V1
    if (hasExt(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
      chain = chainPNext(chain, &dpFeatures);
#endif
#if defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_COOPMAT_BF16)
    if (hasExt(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME))
      chain = chainPNext(chain, &bf16Features);
#endif
#ifdef VK_HAS_ANY_COOPMAT
    hasCoopmatExt = hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    if (hasCoopmatExt)
    {
      chain = chainPNext(chain, &coopmatFeatures);
      chain = chainPNext(chain, &vmmFeatures);
    }
#ifdef VK_HAS_ANY_COOPMAT_FP8
    hasFP8Ext = hasExt(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    if (hasFP8Ext)
      chain = chainPNext(chain, &fp8Features);
#endif
#endif

    VkPhysicalDeviceFeatures2 features2 = {};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = chain;
    auto pfnGetFeat2 = (PFN_vkGetPhysicalDeviceFeatures2)
        vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceFeatures2");
    if (!pfnGetFeat2)
      pfnGetFeat2 = (PFN_vkGetPhysicalDeviceFeatures2)
          vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceFeatures2KHR");
    if (pfnGetFeat2)
      pfnGetFeat2(physDev, &features2);
    baseFeatures2.features = features2.features;

#ifdef VK_HAS_COMPUTE_DP_V1
    if (features2.features.shaderFloat64)
      self->info.float64Supported = true;
#endif
#ifdef VK_HAS_COMPUTE_MP_V1
    if (f16i8Features.shaderFloat16)
    {
      enabledExts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
      f16i8ExtEnabled = true;
      self->info.float16Supported = true;
    }
#endif
#ifdef VK_HAS_COMPUTE_INT8_DP_V1
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
      self->info.int8DotProductSupported = true;
    }
#endif
#if defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_COOPMAT_BF16)
    if (hasExt(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) && bf16Features.shaderBFloat16Type)
    {
      enabledExts.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
      self->info.bfloat16Supported = true;
    }
#endif
#ifdef VK_HAS_ANY_COOPMAT
    if (hasCoopmatExt && coopmatFeatures.cooperativeMatrix && vmmFeatures.vulkanMemoryModel)
    {
      enabledExts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
      self->info.cooperativeMatrixSupported = true;
#ifdef VK_HAS_COOPMAT_FP16
      if (f16i8Features.shaderFloat16 && !f16i8ExtEnabled)
      {
        enabledExts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        f16i8ExtEnabled = true;
        self->info.float16Supported = true;
      }
#endif
#ifdef VK_HAS_COOPMAT_INT8
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
#ifdef VK_HAS_ANY_COOPMAT_FP8
      if (hasFP8Ext && fp8Features.shaderFloat8 &&
          fp8Features.shaderFloat8CooperativeMatrix)
      {
        enabledExts.push_back(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
        self->info.fp8Supported = true;
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

  if (hasExt("VK_EXT_calibrated_timestamps"))
  {
    enabledExts.push_back("VK_EXT_calibrated_timestamps");
    self->info.calibratedTimestampsSupported = true;
  }
}

// ---------------------------------------------------------------------------
// Step 4: create logical device with enabled features/extensions
// ---------------------------------------------------------------------------
static bool createLogicalDevice(VulkanDevice *self, VkPhysicalDevice physDev,
                                const std::vector<const char *> &enabledExts)
{
  float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCI = {};
  queueCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCI.queueFamilyIndex = self->info.computeQueueFamily;
  queueCI.queueCount = 1;
  queueCI.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceCI = {};
  deviceCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCI.queueCreateInfoCount = 1;
  deviceCI.pQueueCreateInfos = &queueCI;
  deviceCI.enabledExtensionCount = (uint32_t)enabledExts.size();
  deviceCI.ppEnabledExtensionNames = enabledExts.empty() ? nullptr : enabledExts.data();

#if defined(VK_HAS_COMPUTE_INT8_DP_V1) || defined(VK_HAS_COMPUTE_MP_V1) || defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_ANY_COOPMAT) || defined(VK_HAS_COMPUTE_DP_V1)
  // Re-chain the feature structs we actually enabled for vkCreateDevice.
  VkPhysicalDeviceShaderFloat16Int8FeaturesKHR f16i8Features = {};
  f16i8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES_KHR;
#ifdef VK_HAS_COMPUTE_INT8_DP_V1
  VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR dpFeatures = {};
  dpFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
#endif
#if defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_COOPMAT_BF16)
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bf16Features = {};
  bf16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
#endif
#ifdef VK_HAS_ANY_COOPMAT
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmatFeatures = {};
  coopmatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  VkPhysicalDeviceVulkanMemoryModelFeatures vmmFeatures = {};
  vmmFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
#ifdef VK_HAS_ANY_COOPMAT_FP8
  VkPhysicalDeviceShaderFloat8FeaturesEXT fp8Features = {};
  fp8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
#endif
#endif

  VkPhysicalDeviceFeatures2 enabledFeatures2 = {};
  enabledFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  void *enabledChain = nullptr;
#ifdef VK_HAS_COMPUTE_DP_V1
  if (self->info.float64Supported)
    enabledFeatures2.features.shaderFloat64 = VK_TRUE;
#endif
  if (self->info.float16Supported || self->info.int8DotProductSupported)
  {
    f16i8Features.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &f16i8Features);
  }
#ifdef VK_HAS_COMPUTE_INT8_DP_V1
  if (self->info.int8DotProductSupported)
  {
    dpFeatures.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &dpFeatures);
  }
#endif
#if defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_COOPMAT_BF16)
  if (self->info.bfloat16Supported)
  {
    bf16Features.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &bf16Features);
  }
#endif
#ifdef VK_HAS_ANY_COOPMAT
  if (self->info.cooperativeMatrixSupported)
  {
    coopmatFeatures.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &coopmatFeatures);
    vmmFeatures.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &vmmFeatures);
  }
#ifdef VK_HAS_ANY_COOPMAT_FP8
  if (self->info.fp8Supported)
  {
    fp8Features.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &fp8Features);
  }
#endif
#endif
  bool haveBaseFeats = enabledFeatures2.features.shaderFloat64 != VK_FALSE;
  if (enabledChain || haveBaseFeats)
  {
    enabledFeatures2.pNext = enabledChain;
    deviceCI.pNext = &enabledFeatures2;
  }
#endif

  if (vkCreateDevice(physDev, &deviceCI, nullptr, &self->device) != VK_SUCCESS)
    return false;

  vkGetDeviceQueue(self->device, self->info.computeQueueFamily, 0, &self->computeQueue);

  // Create command pool
  VkCommandPoolCreateInfo poolCI = {};
  poolCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolCI.queueFamilyIndex = self->info.computeQueueFamily;
  poolCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(self->device, &poolCI, nullptr, &self->commandPool) != VK_SUCCESS)
    return false;

  return true;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
bool VulkanDevice::init(VkInstance inst, VkPhysicalDevice physDev)
{
  physicalDevice = physDev;

  // Step 1: basic properties, memory, queue family
  if (!queryBasicInfo(this, physDev))
    return false;

  // Query supported extensions (used by steps 2-4)
  uint32_t extCount = 0;
  vkEnumerateDeviceExtensionProperties(physDev, nullptr, &extCount, nullptr);
  std::vector<VkExtensionProperties> extProps(extCount);
  vkEnumerateDeviceExtensionProperties(physDev, nullptr, &extCount, extProps.data());
  auto hasExt = [&](const char *name) {
    for (auto &e : extProps)
      if (strcmp(e.extensionName, name) == 0) return true;
    return false;
  };

  // Step 2: probe CU/SM count from vendor extensions
  probeComputeUnitCount(this, physDev, hasExt);

  // Step 3: query optional shader features
  std::vector<const char *> enabledExts;
  queryOptionalFeatures(this, physDev, inst, hasExt, enabledExts);

  // Step 4: create logical device + queue + command pool
  if (!createLogicalDevice(this, physDev, enabledExts))
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
                                         VkPipeline &pipeline,
                                         const VkSpecializationInfo *specInfo)
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
  stageCI.pSpecializationInfo = specInfo;

  VkComputePipelineCreateInfo pipelineCI = {};
  pipelineCI.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineCI.stage = stageCI;
  pipelineCI.layout = pipeLayout;

  VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline);
  vkDestroyShaderModule(device, shaderModule, nullptr);

  return result == VK_SUCCESS;
}

VkResult VulkanDevice::submitAndWait(VkCommandBuffer cmdBuf)
{
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmdBuf;

  VkResult sr = vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
  VkResult wr = vkQueueWaitIdle(computeQueue);
  return sr != VK_SUCCESS ? sr : wr;
}

bool VulkanDevice::zeroBuffer(VkBuffer buffer)
{
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer cmdBuf;
  if (vkAllocateCommandBuffers(device, &allocInfo, &cmdBuf) != VK_SUCCESS)
    return false;

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmdBuf, &beginInfo);

  vkCmdFillBuffer(cmdBuf, buffer, 0, VK_WHOLE_SIZE, 0);

  VkBufferMemoryBarrier barrier = {};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = buffer;
  barrier.offset = 0;
  barrier.size = VK_WHOLE_SIZE;
  vkCmdPipelineBarrier(cmdBuf,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, nullptr, 1, &barrier, 0, nullptr);

  vkEndCommandBuffer(cmdBuf);
  VkResult result = submitAndWait(cmdBuf);
  vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
  return result == VK_SUCCESS;
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

#endif // ENABLE_VULKAN
