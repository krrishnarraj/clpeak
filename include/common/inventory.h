#ifndef CLPEAK_INVENTORY_H
#define CLPEAK_INVENTORY_H

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>
#include <common/benchmark_enums.h>  // Backend

// Backend-neutral description of one device. Per-backend enumerators fill the
// fields that make sense for them and leave the rest at their defaults; the
// printer / JSON serializer skip empty fields. This keeps a single struct
// usable for OpenCL (rich info) and Vulkan (just name + type + API version)
// without forcing a discriminated union.
struct InventoryDevice
{
  int           index = -1;       // the backend's own numbering: what --device takes
  std::string   name;
  std::string   typeStr;          // "GPU" / "CPU" / "NPU" / "Discrete GPU" / ...
  std::string   arch;             // "sm_120" (CUDA), "gfx1201" (ROCm)
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
  std::string                  name;          // OpenCL: real platform; others: one synthetic platform
  std::vector<InventoryDevice> devices;
};

struct BackendInventory
{
  Backend                        id = Backend::COUNT;
  bool                           available = false;
  // Why `available` is false: "onnxruntime library not found", "driver init
  // failed or no devices found".  Printed in place of the device list.
  std::string                    unavailableReason;
  // Backend-level fact worth a line in the listing: the runtime version
  // ("ONNX Runtime 1.29.0"), the OS release Core ML comes with.
  std::string                    info;
  // Diagnostics from enumeration that are not devices -- a provider the
  // runtime names but nothing here can run, with the reason.  Shown under
  // --verbose only, so a missing NPU reads as absent hardware in the
  // default listing.
  std::vector<std::string>       notes;
  std::vector<InventoryPlatform> platforms;   // Vulkan/CUDA: a single synthetic platform
};

// JSON serializer used by the GUI catalog (clpeak_copy_backend_catalog_json).
// Schema is stable and consumed by app/lib/src/model/catalog.dart.
std::string inventoryToJson(const std::vector<BackendInventory> &inv);

// --list-devices.  One format for every backend: a header per backend, then
// one line per device that starts with the exact `backend:index` token
// --device takes, so what a user reads is what they pass.  Backends print
// in the order given, which is the registry's -- the run order.
void printInventory(const std::vector<BackendInventory> &inv, std::ostream &os);

#endif // CLPEAK_INVENTORY_H
