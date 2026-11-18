#ifndef XPUPEAK_INVENTORY_H
#define XPUPEAK_INVENTORY_H

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>
#include <common/benchmark_enums.h>  // Backend, DeviceType
#include <common/hardware_facts.h>   // HardwareFacts, DeviceEvidence, PackageInfo, ProcessorInfo
#include <common/host_info.h>        // SystemInfo

// Backend-neutral description of one device. Per-backend enumerators fill the
// fields that make sense for them and leave the rest at their defaults; the
// printer / JSON serializer skip empty fields. This keeps a single struct
// usable for OpenCL (rich info) and Vulkan (just name + type + API version)
// without forcing a discriminated union.
struct InventoryDevice
{
  int           index = -1;       // the backend's own numbering: what --devices takes
  std::string   name;
  std::string   typeStr;          // "GPU" / "CPU" / "NPU" / "Discrete GPU" / ...
  std::string   arch;             // "sm_120" (CUDA), "gfx1201" (ROCm)
  std::string   driverVersion;    // OpenCL
  std::string   apiVersion;       // Vulkan ("1.2.3")
  std::string   origin;           // how the device arrived: "EP plugin (...)"
                                  // (ONNX plugin EPs only; empty otherwise)
  unsigned int  numComputeUnits = 0;
  unsigned int  maxClockMHz     = 0;
  std::uint64_t globalMemBytes  = 0;
  std::uint64_t maxAllocBytes   = 0;
  bool          hasFp16 = false;
  bool          hasFp64 = false;

  // What the linker reads, as a run's device has them
  // (include/common/hardware_link.h): the type the backend's DeviceSpec
  // gives the device, its API's facts from the backend's *Identify()
  // helper, and its evidence -- the identity flags, and tokens of its
  // identifiers under this process's key (xpupeak::listingEvidence), which
  // are never written anywhere.  Filled by each backend's enumerate().
  DeviceType     type = DeviceType::Unknown;
  HardwareFacts  facts;
  DeviceEvidence evidence;

  // The processor it is a view of ("gpu0"), once linkInventory() has run.
  std::string    processor;
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

// The hardware a listing's devices are views of: the system, its packages
// and its processors, linked from the listing alone, by the rules a run's
// devices are linked by -- so the ids are the ones a run of the same
// backends, seeing the same devices, gives (docs/format-v4.md, Hardware).
struct InventoryHardware
{
  SystemInfo                 system;
  std::vector<PackageInfo>   packages;
  std::vector<ProcessorInfo> processors;

  bool empty() const { return processors.empty(); }
};

// Link a listing: every device's `processor`, and the hardware, from
// probeSystem() and probeHost() as a run's linker has them.
InventoryHardware linkInventory(std::vector<BackendInventory> &inv);

// The `backend:index` token --devices takes for a device of `b`.
std::string inventoryDeviceToken(Backend b, int index);

// JSON serializer used by the GUI catalog (xpupeak_copy_backend_catalog_json):
//   {"backends":[{"name","flag","available",info?,reason?,notes?,
//   "platforms":[{"index","name","devices":[{"index","name","type",arch?,
//       driver?,api?,origin?,compute_units?,clock_mhz?,global_mem_bytes?,
//       max_alloc_bytes?,fp16?,fp64?,processor?}]}]}],
//    packages?, processors?}
// `packages` and `processors` are the result document's own arrays
// (docs/format-v4.md, Hardware), present when `hw` holds a linked listing;
// each device then names its `processor`.  Consumed by
// app/lib/src/model/catalog.dart.
std::string inventoryToJson(const std::vector<BackendInventory> &inv,
                            const InventoryHardware *hw = nullptr);

// The `backends` array of the same document, written into an open array of
// a caller's JsonWriter -- how the run document embeds the inventory under
// --verbose (run_document.cpp), so the two never drift apart.
class JsonWriter;
void writeInventoryBackends(JsonWriter &w, const std::vector<BackendInventory> &inv);

// --list-devices.  With `hw`, the hardware first: the system, then each
// package and its processors -- a CPU's clusters, a GPU's units -- and under
// each processor the `backend:index` tokens that reach it, its own id being
// what --devices takes for all of them.  Then one format for every backend:
// a header per backend, then one line per device that starts with the exact
// `backend:index` token --devices takes, so what a user reads is what they
// pass.  Backends print in the order given, which is the registry's -- the
// run order.
void printInventory(const std::vector<BackendInventory> &inv, std::ostream &os,
                    const InventoryHardware *hw = nullptr);

// The --devices items that name processors (CliOptions::deviceProcessors),
// as the devices of this linked listing on each: added to opts.devices, the
// allow-list every backend reads.  False, with `error` saying why, for a
// processor the listing does not have, or one none of whose devices is on a
// backend that runs.
struct CliOptions;
bool resolveProcessorSelectors(const std::vector<BackendInventory> &inv,
                               const InventoryHardware &hw, CliOptions &opts,
                               std::string &error);

#endif // XPUPEAK_INVENTORY_H
