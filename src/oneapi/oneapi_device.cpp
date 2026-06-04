#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <algorithm>
#include <cstdio>

#ifdef CLPEAK_ONEAPI_HAS_JOINT_MATRIX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#endif

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
    CLPEAK_VLOG("SYCL queue create failed: %s\n", e.what());
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

  // bf16: no SYCL aspect exists in oneAPI 2026.0+ — Intel removed it
  // because all SYCL-capable Intel GPUs (Xe-LP / DG2 / Arc / PVC) have
  // native bf16 hardware.
  info.bf16Supported = true;

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

  // XMX (Xe Matrix eXtensions) presence — name-based heuristic first: Intel
  // discrete GPUs (Arc, Data Center, Battlemage) advertise XMX; integrated
  // Xe-LP / UHD do not.
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

#ifdef CLPEAK_ONEAPI_HAS_JOINT_MATRIX
  // Authoritative override: a non-empty joint_matrix combination table means a
  // matrix engine is present, regardless of marketing name.  This catches
  // unnamed / engineering parts that enumerate as "Intel(R) Graphics [0xXXXX]"
  // and would otherwise be missed by the name heuristic above.
  try
  {
    auto combos = dev.get_info<
        sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
    if (!combos.empty())
      info.xmxSupported = true;
    CLPEAK_VLOG("oneAPI: device '%s' reports %zu joint_matrix combination(s)\n",
                info.deviceName.c_str(), combos.size());
  }
  catch (const std::exception &e)
  {
    CLPEAK_VLOG("oneAPI: matrix_combinations query unavailable: %s\n", e.what());
  }
#endif

  return true;
}

bool OneapiDevice::resetQueue()
{
  try
  {
    stream = sycl::queue(dev, sycl::property::queue::in_order{});
    return true;
  }
  catch (const sycl::exception &e)
  {
    CLPEAK_VLOG("SYCL queue reset failed: %s\n", e.what());
    return false;
  }
}

void OneapiDevice::cleanup()
{
  // sycl::queue / sycl::device clean up via RAII; nothing else to release.
}

#endif // ENABLE_ONEAPI
