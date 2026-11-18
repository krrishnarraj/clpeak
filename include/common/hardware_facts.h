#ifndef XPUPEAK_HARDWARE_FACTS_H
#define XPUPEAK_HARDWARE_FACTS_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// ── Hardware facts ─────────────────────────────────────────────────────────
//
// What an API -- or for a CPU, the OS and CPUID -- says about the hardware,
// typed.  A device keeps its own API's (`facts`), and a processor the facts
// merged from the most native backend that ran, written on the processor
// itself; so a consumer with other rules can merge them again.  Vocabulary
// fields hold their strings.  None identifies a unit: a PCI device id is a
// model's, not a card's.
//
// Apart from run_document.h, which includes it, so that the device listing
// (inventory.h) carries the same facts and evidence a run's devices do, and
// the listing links like a run (include/common/hardware_link.h).

struct CacheInfo {
    std::string   level;           // CacheLevel: "l1d", "l2", "slc"
    std::uint64_t sizeBytes = 0;   // one instance
    unsigned      instances = 0;   // how many the processor, or the cluster, has
    unsigned      lineBytes = 0;
};

struct CpuCluster {
    std::string kind;              // ClusterKind; "" for a CPU of one kind of core
    unsigned    cores = 0;
    unsigned    threadsPerCore = 0;
    unsigned    maxMhz = 0;
    std::string microarch;         // as MIDR or CPUID decode it: "Firestorm", "Zen 2"
    std::string midr;              // ARM: MIDR_EL1 of its cores, hex
    std::vector<CacheInfo> caches; // the cluster's own
};

struct CpuEngine {
    std::string kind;              // EngineKind: "matrix"
    std::string name;              // "AMX", "SME"
    std::string scope;             // EngineScope: one per "core" | per "cluster"
    std::string reachedBy;         // EngineReach: "instructions" | "library"
};

struct CpuFacts {
    std::string isa;               // "arm64", "x86_64"
    unsigned    cores   = 0;
    unsigned    threads = 0;
    unsigned    sockets = 0;
    std::vector<CpuCluster> clusters;
    std::vector<CacheInfo>  caches;    // the ones the whole processor shares
    // x86: CPUID's vendor string and family / model / stepping; -1 unknown.
    std::string cpuidVendor;
    int         cpuidFamily   = -1;
    int         cpuidModel    = -1;
    int         cpuidStepping = -1;
    std::vector<CpuEngine>   engines;
    std::vector<std::string> features; // normalised: "neon", "dotprod", "avx512_vnni"

    bool empty() const;
};

struct HardwareFacts {
    std::string   vendorId;        // the API's vendor id, lower-case hex: "10de", "10005"
    std::string   deviceId;        // the PCI device id, lower-case hex: a model id
    // The vendor's own string, as the API gives it: "NVIDIA Corporation".  A
    // device's; a processor keeps its own in ProcessorInfo::vendorName.
    std::string   vendorName;
    std::string   arch;            // "sm_120", "gfx1201", "Apple7"
    unsigned      computeUnits = 0;
    std::string   computeUnit;     // what one is called: "SM", "CU", "Xe-core", "core"
    unsigned      clockMhz = 0;
    std::uint64_t memoryBytes = 0;
    int           integrated = -1; // -1 unknown, 0 discrete, 1 integrated
    CpuFacts      cpu;             // a CPU's

    bool empty() const;
};

// What links a device to its processor, as the document keeps it: tokens for
// the strong identifiers its API gave -- HMACs under a key made per run and
// never written to the result, so two devices in one file can be seen to
// share an identifier while nothing correlates across files -- and the
// weaker evidence, which identifies nothing.  The raw identifiers never
// leave the process (logger::DeviceSpec::Identity).
struct DeviceEvidence {
    std::map<std::string, std::string> tokens;   // "uuid" | "luid" | "pci_bus" | "registry"
    bool        host = false;      // executes on the host CPU
    std::string soc;               // ProcessorKind: the SoC's "npu" or "gpu"
    bool        defaultGpu = false;// the platform's default GPU

    bool empty() const { return tokens.empty() && !host && soc.empty() && !defaultGpu; }
};

// ---- Hardware ---------------------------------------------------------------
//
// The run's hardware, which the devices are views of: packages (an SoC, a CPU
// package, a discrete card) and the processors in them, with run-local ids
// that are deterministic for one machine -- `cpu0`, `gpu0`, `npu0`.

struct PackageInfo {
    std::string id;               // "pkg0"
    std::string kind;             // PackageKind: "soc" | "cpu" | "discrete"
    std::string vendor;           // Vendor, when known
    std::string name;             // "Apple M1 Pro", "NVIDIA GeForce RTX 5060"
    std::string ext;
};

struct ProcessorInfo {
    std::string   id;             // "cpu0", "gpu0", "npu0"
    std::string   kind;           // ProcessorKind
    std::string   package;        // PackageInfo::id
    std::string   vendor;         // Vendor, when known
    std::string   vendorName;     // the vendor's own string: "NVIDIA Corporation"
    std::string   name;
    HardwareFacts facts;          // merged; written on the processor itself
    std::string   ext;
};

#endif  // XPUPEAK_HARDWARE_FACTS_H
