#ifndef XPUPEAK_HARDWARE_LINK_H
#define XPUPEAK_HARDWARE_LINK_H

#include <cstdint>
#include <string>
#include <vector>

#include <common/logger.h>        // logger::Identity
#include <common/run_document.h>
#include <common/vocabulary.h>

// ── Linking devices to the hardware they ran on ────────────────────────────
//
// A device is one backend's view of a processor: the M1 Pro's GPU is the
// Metal, Vulkan, OpenCL, Core ML and LiteRT devices of one run.  Each backend
// passes what its API says about who the device is (logger::Identity: a UUID,
// a LUID, a PCI bus address, a Metal registry ID, "runs on the host CPU", "is
// the SoC's NPU", "is the platform's default GPU") with what it says about the
// hardware (HardwareFacts), and the linker merges the views into processors
// and the processors into packages, by the rules of docs/format-v4.md,
// Linking:
//
//   Strong -- an equal UUID, LUID, PCI bus address or registry ID merges.
//   Host   -- a device that executes on the CPU is the host CPU.
//   SoC    -- an ML runtime's NPU is the SoC's NPU; an Android GPU delegate
//             the SoC's GPU.
//   Sole   -- a device the strong rules leave alone joins the only candidate
//             of its kind and vendor: never one of several.
//
// Nothing else merges, and a device nothing links is a processor of its own.
//
// The identifiers themselves never leave the process: a GPU's UUID is in
// effect a serial number.  The logger turns them into tokens when the device
// is opened (evidenceFrom()), HMACs under the run's evidence key, so the
// document, its checkpoint and its log hold only tokens -- two devices in a
// file can be seen to share an identifier, and nothing correlates across
// files.  The linker reads only the tokens, so a file re-links the same way.

namespace xpupeak {

// ---- Identifiers, in the one form every API's is written in ----------------
//
// The same identifier read through two APIs has to be the same text, or its
// tokens differ.  Each returns "" for an identifier the API did not really
// give: all zero, or unparseable.

// 16 bytes as 32 lower-case hex digits: CUDA's CUuuid, Vulkan's deviceUUID,
// OpenCL's CL_DEVICE_UUID_KHR, HIP's hipUUID, Level Zero's UUID.
std::string uuidText(const unsigned char bytes[16]);

// A Windows adapter LUID as one 64-bit number, HighPart:LowPart, in 16 hex
// digits.  The 8 bytes Vulkan, OpenCL, CUDA and HIP give are the LUID struct
// in memory (LowPart first, little-endian); DXGI's is the two parts.
std::string luidText(const unsigned char bytes[8]);
std::string luidText(std::uint32_t lowPart, std::int32_t highPart);

// A PCI bus address, "0000:01:00.0": domain, bus, device, function, in hex.
std::string pciBusText(unsigned domain, unsigned bus, unsigned device, unsigned function);
// The same from an API's own string: "0000:01:00.0", "01:00.0", "0:1:0.0";
// "" when it is not one.
std::string pciBusFromText(const std::string &text);

// Metal's registryID (and OpenCL's CL_DEVICE_REGISTRY_ID_APPLE): 16 hex digits.
std::string registryText(std::uint64_t id);

// A PCI vendor or device id as the facts write it: "10de", "10005".
std::string pciIdText(std::uint32_t id);

// ---- Facts -------------------------------------------------------------------

// The vendor a device's facts name: their PCI (or Khronos) vendor id, else
// the first word of the vendor's own string ("NVIDIA Corporation", "Apple").
bool vendorOf(const HardwareFacts &facts, Vendor &out);

// What one of a generic API's compute units -- OpenCL's, SYCL's -- is on a
// device of `type` whose facts name its vendor: an SM on NVIDIA's GPUs, a CU
// on AMD's, an EU on Intel's, a core on Apple's and Arm's, a thread on a CPU;
// "" where that is not known.
std::string computeUnitName(const HardwareFacts &facts, DeviceType type);

// ---- Tokens ----------------------------------------------------------------

// The token of one identifier under a run's key (hex, RunMeta::evidenceKey):
// the first 64 bits of HMAC-SHA256(key, kind '\0' value), as 16 hex digits.
// `kind` is the evidence key it is filed under ("uuid", "luid", "pci_bus",
// "registry"), so one value under two kinds gives two tokens.  An empty or
// malformed key is replaced by this process's own (processEvidenceKey()).
std::string evidenceToken(const std::string &keyHex, const std::string &kind,
                          const std::string &value);

// A key made once per process, for evidence that is never written anywhere --
// a device listing linked before any run -- and for a device opened with no
// run to take its key from.
const std::string &processEvidenceKey();

// A device's evidence as the document keeps it: the flags as they are, and a
// token for each identifier `id` holds.  The logger calls it as the device
// opens, with the run's key, so a checkpoint already holds tokens.
DeviceEvidence evidenceFrom(const logger::Identity &id, const std::string &keyHex);

// ---- The linker --------------------------------------------------------------

// One device as the linker sees it.
struct LinkDevice {
    std::string    backend;           // as the document names it: "Vulkan"
    int            deviceIndex = -1;
    std::string    name;
    DeviceType     type = DeviceType::Unknown;
    DeviceEvidence evidence;
    HardwareFacts  facts;
};

// What the host CPU processor starts from when no CPU backend device is
// there to give its facts: the brand, the thread count and the ISA.
struct HostCpu {
    std::string name;
    unsigned    threads = 0;
    std::string isa;                  // "arm64", "x86_64"
};

// The run's hardware: its packages and processors, ids assigned, and for
// each input device the processor it is a view of and what linked it.
struct HardwareLink {
    std::vector<PackageInfo>   packages;
    std::vector<ProcessorInfo> processors;
    std::vector<std::string>   processorOf;   // per input device: ProcessorInfo::id
    std::vector<std::string>   linkOf;        // per input device: a LinkEvidence word

    const ProcessorInfo *processor(const std::string &id) const;
    // The input devices that are views of a processor, in input order.
    std::vector<size_t> devicesOf(const std::string &processorId) const;
};

// Link `devices` (in any order: run order is taken from the Backend enum
// and the device index).  `socName` names the host package where the system
// gives one (SystemInfo::soc).  Every decision, and every plausible pair
// left apart, is a --verbose line; none holds an identifier.
HardwareLink linkDevices(const std::vector<LinkDevice> &devices, const HostCpu &host,
                         const std::string &socName);

// Where the host CPU processor's start comes from when no CPU backend ran.
enum class LinkHost {
    Probe,      // this process's machine: a run's own document, before it is saved
    FromFile,   // the document alone: re-linking a saved file, wherever it came from
};

// Link a run's document: replaces `packages` and `processors`, and sets
// every device's `processor` and `link`.  Both hosts call it once, after the
// last backend and before the document is saved; a resumed run's devices
// are linked with the ones the parts before measured, from their tokens.
void linkHardware(RunDocument &doc, LinkHost host = LinkHost::Probe);

// ---- Reading a linked document -----------------------------------------------

// "gpu0" from (gpu, 0), and back; false for a string that is not an id.
std::string processorIdFor(ProcessorKind kind, unsigned ordinal);
bool        parseProcessorId(const std::string &id, ProcessorKind &kind, unsigned &ordinal);

// The processor a device is a view of; nullptr in a file nothing linked.
const ProcessorInfo *processorOf(const RunDocument &doc, const DeviceResult &device);
// The devices that are views of a processor, in document order.
std::vector<const DeviceResult *> devicesOf(const RunDocument &doc, const std::string &processorId);
// A processor's package, and a package's processors.
const PackageInfo *packageOf(const RunDocument &doc, const ProcessorInfo &processor);
std::vector<const ProcessorInfo *> processorsIn(const RunDocument &doc, const std::string &packageId);

} // namespace xpupeak

#endif  // XPUPEAK_HARDWARE_LINK_H
