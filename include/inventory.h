#ifndef CLPEAK_INVENTORY_H
#define CLPEAK_INVENTORY_H

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

struct CliOptions;

// Backend-neutral description of one device. Per-backend enumerators fill the
// fields that make sense for them and leave the rest at their defaults; the
// printer / JSON serializer skip empty fields. This keeps a single struct
// usable for OpenCL (rich info) and Vulkan (just name + type + API version)
// without forcing a discriminated union.
struct InventoryDevice
{
  int           index = -1;
  std::string   name;
  std::string   typeStr;          // "GPU" / "CPU" / "Discrete GPU" / ...
  std::string   driverVersion;    // OpenCL
  std::string   apiVersion;       // Vulkan ("1.2.3")
  unsigned int  numComputeUnits = 0;
  unsigned int  maxClockMHz     = 0;
  std::uint64_t globalMemBytes  = 0;
  std::uint64_t maxAllocBytes   = 0;
  bool          hasFp16 = false;
  bool          hasFp64 = false;
};

struct InventoryPlatform
{
  int                          index = -1;
  std::string                  name;          // OpenCL: real platform; Vulkan: "Vulkan"
  std::vector<InventoryDevice> devices;
};

struct BackendInventory
{
  std::string                    backend;     // "OpenCL" / "Vulkan" / ...
  bool                           available = false;
  std::vector<InventoryPlatform> platforms;   // Vulkan/CUDA: a single synthetic platform
};

// Implemented in clpeak.cpp.
BackendInventory enumerateOpenCL();

#ifdef ENABLE_VULKAN
// Implemented in vk_peak.cpp.
BackendInventory enumerateVulkan();
#endif

#ifdef ENABLE_CUDA
// Implemented in cuda_peak.cpp.
BackendInventory enumerateCuda();
#endif

#ifdef ENABLE_METAL
// Implemented in mtl_peak.mm.
BackendInventory enumerateMetal();
#endif

// Aggregator: enumerates every backend not skipped in opts.
std::vector<BackendInventory> enumerateAllBackends(const CliOptions &opts);

// Pretty-printer used by the desktop --list-devices flow. Output format
// matches the legacy per-backend listDevices output.
void printInventory(const std::vector<BackendInventory> &inv, std::ostream &os);

// JSON serializer used by the Android JNI surface. Schema is stable and
// consumed by BackendCatalog.kt.
std::string inventoryToJson(const std::vector<BackendInventory> &inv);

#endif // CLPEAK_INVENTORY_H
