#ifdef ENABLE_VULKAN

#include <vulkan/vk_peak.h>
#include <common/options.h>
#include <common/inventory.h>
#include <common/common.h>
#include <cstring>
#include <sstream>
#include <chrono>
#include <cfloat>
#if defined(_WIN32)
#include <windows.h>
#else
#include <time.h>
#endif

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
  info.vkDeviceType = props.deviceType;
  // Set neutral DeviceType
  if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
      info.deviceType = DeviceType::Cpu;
  else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
           props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      info.deviceType = DeviceType::Gpu;
  else
      info.deviceType = DeviceType::Unknown;

  // CU/SM count is filled in below from vendor property extensions if the
  // implementation advertises them.  When unavailable, dispatch helpers pick a
  // device-type-specific fallback.
  info.numCUs = 0;

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

  // Probe vendor property extensions for CU/SM count.  These are query-only
  // (no feature bits, no need to chain at vkCreateDevice -- the spec lets us
  // read them via vkGetPhysicalDeviceProperties2 as long as the extension is
  // advertised).  Order matters: prefer the most specific count.
  {
    VkPhysicalDeviceProperties2 p2 = {};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

    if (hasExt("VK_NV_shader_sm_builtins"))
    {
      VkPhysicalDeviceShaderSMBuiltinsPropertiesNV smProps = {};
      smProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_PROPERTIES_NV;
      p2.pNext = &smProps;
      vkGetPhysicalDeviceProperties2(physDev, &p2);
      info.numCUs = smProps.shaderSMCount;
    }
    else if (hasExt("VK_AMD_shader_core_properties2"))
    {
      VkPhysicalDeviceShaderCoreProperties2AMD coreProps2 = {};
      coreProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_2_AMD;
      p2.pNext = &coreProps2;
      vkGetPhysicalDeviceProperties2(physDev, &p2);
      info.numCUs = coreProps2.activeComputeUnitCount;
    }
    else if (hasExt("VK_AMD_shader_core_properties"))
    {
      VkPhysicalDeviceShaderCorePropertiesAMD coreProps = {};
      coreProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;
      p2.pNext = &coreProps;
      vkGetPhysicalDeviceProperties2(physDev, &p2);
      info.numCUs = coreProps.shaderEngineCount *
                    coreProps.shaderArraysPerEngineCount *
                    coreProps.computeUnitsPerShaderArray;
    }
    else if (hasExt("VK_ARM_shader_core_builtins"))
    {
      VkPhysicalDeviceShaderCoreBuiltinsPropertiesARM armProps = {};
      armProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES_ARM;
      p2.pNext = &armProps;
      vkGetPhysicalDeviceProperties2(physDev, &p2);
      info.numCUs = armProps.shaderCoreCount;
    }
  }

  std::vector<const char *> enabledExts;
  info.int8DotProductSupported = false;
  info.float16Supported = false;
  info.float64Supported = false;
  info.bfloat16Supported = false;
  info.cooperativeMatrixSupported = false;
  info.fp8Supported = false;
  info.atomicFloat32Supported = false;
  info.atomicInt64Supported = false;
  info.coopmatFP16Supported = false;
  info.coopmatBF16Supported = false;
  info.coopmatINT8K = 0;
  info.coopmatFP8E4M3Supported = false;
  info.coopmatFP8E5M2Supported = false;
  info.coopmatFP32Supported = false;
  info.calibratedTimestampsSupported = false;

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
#if defined(VK_HAS_COMPUTE_INT8_DP_V1) || defined(VK_HAS_COMPUTE_MP_V1) || defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_ANY_COOPMAT) || defined(VK_HAS_COMPUTE_DP_V1) || defined(VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_FLOAT) || defined(VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_UINT64)
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
  // Cooperative-matrix shaders emitted by glslang use MemoryModel Vulkan via
  // GL_KHR_memory_scope_semantics, which requires the vulkanMemoryModel
  // feature bit (core in 1.2, otherwise VK_KHR_vulkan_memory_model).
  VkPhysicalDeviceVulkanMemoryModelFeatures vmmFeatures = {};
  vmmFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES;
#ifdef VK_HAS_ANY_COOPMAT_FP8
  VkPhysicalDeviceShaderFloat8FeaturesEXT fp8Features = {};
  fp8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
#endif
#endif
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_FLOAT
  VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures = {};
  atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
#endif
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_UINT64
  VkPhysicalDeviceShaderAtomicInt64Features atomicInt64Features = {};
  atomicInt64Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES;
#endif
  // Base features (shaderFloat64 etc.) come back inside Features2.features.
  VkPhysicalDeviceFeatures2 baseFeatures2 = {};
  baseFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

  // Small helper: prepend a features struct onto a pNext chain.
  // All VkPhysicalDevice*Features* structs begin with the (sType, pNext)
  // pair described by VkBaseOutStructure, which is what we rely on here.
  auto chainPNext = [](void *head, void *newNode) -> void * {
    reinterpret_cast<VkBaseOutStructure *>(newNode)->pNext =
        reinterpret_cast<VkBaseOutStructure *>(head);
    return newNode;
  };

  bool f16i8ExtEnabled = false;
  bool hasF16I8Ext = hasExt(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
#ifdef VK_HAS_ANY_COOPMAT
  bool hasCoopmatExt = false;
#endif
#ifdef VK_HAS_ANY_COOPMAT_FP8
  bool hasFP8Ext = false;
#endif
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_FLOAT
  bool hasAtomicFloatExt = false;
#endif
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_UINT64
  bool hasAtomicInt64Ext = false;
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
    // Cooperative-matrix requires vulkanMemoryModel.  Chain both feature
    // structs whenever the coopmat extension is present; the memory-model
    // struct is core-1.2 and is what drivers advertising coopmat expose.
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
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_FLOAT
    hasAtomicFloatExt = hasExt("VK_EXT_shader_atomic_float");
    if (hasAtomicFloatExt)
      chain = chainPNext(chain, &atomicFloatFeatures);
#endif
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_UINT64
    hasAtomicInt64Ext = hasExt(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
    if (hasAtomicInt64Ext)
      chain = chainPNext(chain, &atomicInt64Features);
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
    baseFeatures2.features = features2.features;

#ifdef VK_HAS_COMPUTE_DP_V1
    if (features2.features.shaderFloat64)
      info.float64Supported = true;
#endif
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_FLOAT
    if (hasAtomicFloatExt && atomicFloatFeatures.shaderBufferFloat32AtomicAdd)
    {
      enabledExts.push_back("VK_EXT_shader_atomic_float");
      info.atomicFloat32Supported = true;
    }
#endif
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_UINT64
    if (hasAtomicInt64Ext && atomicInt64Features.shaderBufferInt64Atomics)
    {
      enabledExts.push_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
      info.atomicInt64Supported = true;
    }
#endif
#ifdef VK_HAS_COMPUTE_MP_V1
    if (f16i8Features.shaderFloat16)
    {
      enabledExts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
      f16i8ExtEnabled = true;
      info.float16Supported = true;
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
      info.int8DotProductSupported = true;
    }
#endif
#if defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_COOPMAT_BF16)
    if (hasExt(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME) && bf16Features.shaderBFloat16Type)
    {
      enabledExts.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
      info.bfloat16Supported = true;
    }
#endif
#ifdef VK_HAS_ANY_COOPMAT
    if (hasCoopmatExt && coopmatFeatures.cooperativeMatrix && vmmFeatures.vulkanMemoryModel)
    {
      enabledExts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
      info.cooperativeMatrixSupported = true;
      // FP16 coopmat inputs also need shaderFloat16.
#ifdef VK_HAS_COOPMAT_FP16
      if (f16i8Features.shaderFloat16 && !f16i8ExtEnabled)
      {
        enabledExts.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
        f16i8ExtEnabled = true;
        info.float16Supported = true;
      }
#endif
      // INT8 coopmat inputs also need shaderInt8 + 8-bit storage.
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
      // FP8 coopmat inputs additionally need VK_EXT_shader_float8 with
      // shaderFloat8 + shaderFloat8CooperativeMatrix both advertised.
#ifdef VK_HAS_ANY_COOPMAT_FP8
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

  // VK_EXT_calibrated_timestamps lets us pin a host time into the GPU
  // timestamp domain, which is exactly what's needed to measure one-way
  // dispatch latency (host submit -> GPU kernel start) the same way
  // OpenCL does via CL_PROFILING_COMMAND_QUEUED.  Optional.
  if (hasExt("VK_EXT_calibrated_timestamps"))
  {
    enabledExts.push_back("VK_EXT_calibrated_timestamps");
    info.calibratedTimestampsSupported = true;
  }

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

#if defined(VK_HAS_COMPUTE_INT8_DP_V1) || defined(VK_HAS_COMPUTE_MP_V1) || defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_ANY_COOPMAT) || defined(VK_HAS_COMPUTE_DP_V1) || defined(VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_FLOAT) || defined(VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_UINT64)
  // Re-chain the feature structs we actually enabled for vkCreateDevice.
  // The query-phase chain is discarded; these are the features we ask the
  // driver to turn on.
  VkPhysicalDeviceFeatures2 enabledFeatures2 = {};
  enabledFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  void *enabledChain = nullptr;
#ifdef VK_HAS_COMPUTE_DP_V1
  if (info.float64Supported)
    enabledFeatures2.features.shaderFloat64 = VK_TRUE;
#endif
  if (info.float16Supported || info.int8DotProductSupported)
  {
    f16i8Features.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &f16i8Features);
  }
#ifdef VK_HAS_COMPUTE_INT8_DP_V1
  if (info.int8DotProductSupported)
  {
    dpFeatures.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &dpFeatures);
  }
#endif
#if defined(VK_HAS_COMPUTE_BF16_V1) || defined(VK_HAS_COOPMAT_BF16)
  if (info.bfloat16Supported)
  {
    bf16Features.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &bf16Features);
  }
#endif
#ifdef VK_HAS_ANY_COOPMAT
  if (info.cooperativeMatrixSupported)
  {
    coopmatFeatures.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &coopmatFeatures);
    vmmFeatures.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &vmmFeatures);
  }
#ifdef VK_HAS_ANY_COOPMAT_FP8
  if (info.fp8Supported)
  {
    fp8Features.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &fp8Features);
  }
#endif
#endif
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_FLOAT
  if (info.atomicFloat32Supported)
  {
    atomicFloatFeatures.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &atomicFloatFeatures);
  }
#endif
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_UINT64
  if (info.atomicInt64Supported)
  {
    atomicInt64Features.pNext = nullptr;
    enabledChain = chainPNext(enabledChain, &atomicInt64Features);
  }
#endif
  // Always pass enabledFeatures2 if any enabled state present (chain or base feats).
  bool haveBaseFeats = enabledFeatures2.features.shaderFloat64 != VK_FALSE;
  if (enabledChain || haveBaseFeats)
  {
    enabledFeatures2.pNext = enabledChain;
    deviceCI.pNext = &enabledFeatures2;
  }
  (void)baseFeatures2;
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

// ---------------------------------------------------------------------------
// vkPeak
// ---------------------------------------------------------------------------

vkPeak::vkPeak()
  : deviceIndex(-1),
    instance(VK_NULL_HANDLE)
{
}

vkPeak::~vkPeak()
{
  cleanup();
}

void vkPeak::applyOptions(const CliOptions &opts)
{
    Peak::applyOptions(opts);
    deviceIndex = opts.vkDeviceIndex;
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
                        uint32_t groupCountX,
                        unsigned int targetTimeUsLocal, unsigned int forcedIters,
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
    for (unsigned int i = 0; i < n; i++)
      vkCmdDispatch(cmdBuf, groupCountX, 1, 1);
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
    log->note("Vulkan: failed to create instance or no devices found\n");
    return -1;
  }

  for (size_t d = 0; d < physicalDevices.size(); d++)
  {
    if (deviceIndex >= 0 && static_cast<size_t>(deviceIndex) != d)
      continue;

    VulkanDevice dev;
    if (!dev.init(instance, physicalDevices[d]))
    {
      log->note("Vulkan: failed to init device " + std::to_string(d) + "\n");
      continue;
    }

#ifdef VK_HAS_ANY_COOPMAT
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
            if (matches(p, 16,16,16, VK_COMPONENT_TYPE_FLOAT32_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR))
              dev.info.coopmatFP32Supported = true;
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
            (dev.info.vkDeviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) ? DeviceType::Cpu : DeviceType::Gpu);
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
#ifdef VK_HAS_ATOMIC_THROUGHPUT_GLOBAL_FLOAT
    // Float atomic add is reported in the fp-compute phase (mirrors Metal).
    if (isAllowedAs(Benchmark::AtomicThroughput, Category::FpCompute))
        runAtomicThroughputFp(dev, cfg);
#endif

    // ---- Phase 2: integer compute (GOPS / TOPS) --------------------
#ifdef VK_HAS_COMPUTE_INT32_V1
    if (isAllowed(Benchmark::ComputeInt))        runComputeInt32(dev, cfg);
#endif
#ifdef VK_HAS_COMPUTE_INT8_DP_V1
    if (isAllowed(Benchmark::ComputeInt8DP))     runComputeInt8DP(dev, cfg);
#endif
#ifdef VK_HAS_COMPUTE_INT4_PACKED_V1
    if (isAllowed(Benchmark::ComputeInt4Packed)) runComputeInt4Packed(dev, cfg);
#endif
#ifdef VK_HAS_ANY_COOPMAT
    if (isAllowedAs(Benchmark::CoopMatrix, Category::IntCompute))
        runCoopMatrix(dev, cfg, /*intPart=*/true);
#endif
    if (isAllowedAs(Benchmark::AtomicThroughput, Category::IntCompute))
        runAtomicThroughput(dev, cfg);

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
// Shared compute-peak driver.
//
// Every compute-peak benchmark (runComputeSP / MP / INT8-DP / INT4-packed /
// coop-matrix / ...) shares the same Vulkan scaffolding: allocate a single
// device-local output buffer, build a one-binding descriptor set, create a
// pipeline from the shader's SPIR-V, dispatch repeatedly with a push
// constant, and report work-per-WI / elapsed time.  The only differences
// are the shader, the buffer-element size, the push-constant payload, and
// the strings used for display / result output.  All of those are bundled into
// vk_compute_desc_t so each concrete benchmark becomes a few-line wrapper.
// ---------------------------------------------------------------------------

int vkPeak::runComputeKernel(VulkanDevice &dev, benchmark_config_t &cfg,
                             const vk_compute_desc_t &d)
{
  logger::TestSpec testSpec;
  testSpec.tag = d.resultTag;
  testSpec.display = d.title;
  testSpec.unit = d.unit;
  auto test = currentDeviceScope->beginTest(testSpec);

  if (d.skip)
  {
    const char *msg = d.skipMsg ? d.skipMsg : "Skipped";
    if (d.variants && d.numVariants > 0)
    {
      for (uint32_t i = 0; i < d.numVariants; i++)
        test.skip(d.variants[i].label, ResultStatus::Unsupported, msg);
    }
    else
    {
      test.skip(d.metricLabel, ResultStatus::Unsupported, msg);
    }
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
  // When Vulkan exposes a CU count, mirror OpenCL's
  // numCUs*2048*maxWGSize formula.  Unknown integrated/mobile GPUs use a
  // smaller floor so the calibration probe does not become a watchdog-sized
  // dispatch.  Cooperative-matrix shaders run one subgroup per work-group
  // (32 threads on NVIDIA / AMD RDNA3+ / Intel Arc); other compute kernels
  // use the classic 256.
  const uint32_t wgSize = d.wgSize ? d.wgSize : 256;
  const uint32_t outPerWG = d.outElemsPerWG ? d.outElemsPerWG : wgSize;
  uint64_t globalWIs = targetVulkanGlobalThreads(dev.info);
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
    log->note("Failed to allocate buffer\n");
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
  // creation are skipped but don't abort the group -- some drivers accept
  // the v1 shader but choke on a wider packed variant.
  for (const auto &v : variants)
  {
    VkPipeline pipeline;
    if (!dev.createComputePipeline(v.spirv, v.spirvSize, dsLayout, pipeLayout, pipeline))
    {
      test.skip(v.label, ResultStatus::Error, "Pipeline creation failed");
      continue;
    }

    float timed = runKernel(dev, pipeline, pipeLayout, descSet, numGroups,
                            cfg.targetTimeUs, forceIters ? specifiedIters : 0,
                            d.pushData, d.pushSize);
    if (timed <= 0.0f)
    {
      test.skip(v.label, ResultStatus::Error, "vkQueueSubmit/WaitIdle failed");
      vkDestroyPipeline(dev.device, pipeline, nullptr);
      continue;
    }
    double divider = d.unitDivider > 0.0 ? d.unitDivider : 1e9;
    float value = (float)((double)globalWIs * (double)d.workPerWI * 1e6 / timed / divider);

    test.emit(v.label, value);

    vkDestroyPipeline(dev.device, pipeline, nullptr);
  }

  vkDestroyDescriptorPool(dev.device, descPool, nullptr);
  vkDestroyPipelineLayout(dev.device, pipeLayout, nullptr);
  vkDestroyDescriptorSetLayout(dev.device, dsLayout, nullptr);
  vkDestroyBuffer(dev.device, outputBuf, nullptr);
  vkFreeMemory(dev.device, outputMem, nullptr);

  return 0;
}

// ---------------------------------------------------------------------------
// Benchmark methods live in separate category files:
//   compute_float.cpp    compute_int.cpp    coopmat.cpp
//   global_bandwidth.cpp local_bandwidth.cpp image_bandwidth.cpp
//   transfer_bandwidth.cpp atomic_throughput.cpp kernel_latency.cpp
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
