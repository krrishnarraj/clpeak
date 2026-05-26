#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <algorithm>
#include <cstdio>

OneapiDevice::OneapiDevice() = default;
OneapiDevice::~OneapiDevice() { cleanup(); }

bool OneapiDevice::init(int devIndex, const sycl::device &d)
{
  deviceIndex = devIndex;
  dev = d;

  try
  {
    // in_order avoids implicit dep tracking; we wait between submissions.
    stream = sycl::queue(dev, sycl::property::queue::in_order{});
  }
  catch (const sycl::exception &e)
  {
    fprintf(stderr, "SYCL queue create failed: %s\n", e.what());
    return false;
  }

  try
  {
    info.deviceName       = dev.get_info<sycl::info::device::name>();
    info.vendor           = dev.get_info<sycl::info::device::vendor>();
    info.driverVersion    = dev.get_info<sycl::info::device::driver_version>();
    info.numCUs           = (int)dev.get_info<sycl::info::device::max_compute_units>();
    info.maxWorkGroupSize = dev.get_info<sycl::info::device::max_work_group_size>();
    info.totalGlobalMem   = dev.get_info<sycl::info::device::global_mem_size>();
    info.clockRateMHz     = (int)dev.get_info<sycl::info::device::max_clock_frequency>();
  }
  catch (const sycl::exception &)
  {
    // Non-fatal: leave fields at defaults if the runtime declines a query.
  }

  // Backend label (Level Zero / OpenCL / CUDA) lets the user see which
  // SYCL runtime path is exercising the device.
  switch (dev.get_backend())
  {
#if defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO) || defined(SYCL_BACKEND_LEVEL_ZERO)
    case sycl::backend::ext_oneapi_level_zero: info.backendName = "Level Zero"; break;
#endif
    case sycl::backend::opencl:                info.backendName = "OpenCL";     break;
#if defined(SYCL_EXT_ONEAPI_BACKEND_CUDA)
    case sycl::backend::ext_oneapi_cuda:       info.backendName = "CUDA";       break;
#endif
#if defined(SYCL_EXT_ONEAPI_BACKEND_HIP)
    case sycl::backend::ext_oneapi_hip:        info.backendName = "HIP";        break;
#endif
    default:                                   info.backendName = "SYCL";       break;
  }

  if (dev.is_gpu())              info.deviceType = DeviceType::Gpu;
  else if (dev.is_cpu())         info.deviceType = DeviceType::Cpu;
  else if (dev.is_accelerator()) info.deviceType = DeviceType::Accelerator;
  else                           info.deviceType = DeviceType::Unknown;

  info.fp16Supported = dev.has(sycl::aspect::fp16);
  info.fp64Supported = dev.has(sycl::aspect::fp64);

  // bf16: aspect was promoted from ext_oneapi_bfloat16_math_functions on
  // newer SYCL releases.  Probe both, ignore unknown-aspect throws.
  try { info.bf16Supported = dev.has(sycl::aspect::ext_oneapi_bfloat16_math_functions); }
  catch (...) { info.bf16Supported = false; }

  try
  {
    info.subGroupSizes = dev.get_info<sycl::info::device::sub_group_sizes>();
    if (!info.subGroupSizes.empty())
    {
      // Pick the largest reported sub-group size as the preferred lane width;
      // Intel iGPU exposes {8,16,32}, Arc/PVC {8,16,32} also, Battlemage {16,32}.
      info.preferredSubGroupSize =
          (uint32_t)*std::max_element(info.subGroupSizes.begin(), info.subGroupSizes.end());
    }
  }
  catch (const sycl::exception &) {}

  // XMX (Xe Matrix eXtensions) presence is inferred from joint_matrix
  // dimension tables exposing at least one supported size.  We can't probe
  // that without including <sycl/ext/oneapi/matrix>, so use a name-based
  // heuristic: Intel discrete GPUs (Arc, Data Center, Battlemage) all
  // advertise XMX; integrated Xe-LP / UHD do not.
  if (info.deviceType == DeviceType::Gpu &&
      info.vendor.find("Intel") != std::string::npos)
  {
    const std::string &n = info.deviceName;
    info.xmxSupported =
        n.find("Arc")        != std::string::npos ||
        n.find("Data Center") != std::string::npos ||
        n.find("Max ")        != std::string::npos ||
        n.find("Flex ")       != std::string::npos ||
        n.find("Battlemage")  != std::string::npos ||
        n.find("Ponte Vecchio") != std::string::npos;
  }

  return true;
}

void OneapiDevice::cleanup()
{
  // sycl::queue / sycl::device clean up via RAII; nothing else to release.
}

#endif // ENABLE_ONEAPI
