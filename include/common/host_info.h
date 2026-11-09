#ifndef XPUPEAK_HOST_INFO_H
#define XPUPEAK_HOST_INFO_H

#include <cstdint>
#include <string>

// ── The machine the run happened on ────────────────────────────────────────
//
// Recorded once per run into the result document, so a file is comparable to
// another file without the reader having to be told where it came from.  A
// GPU number means something different on a laptop that thermally throttles
// than on a workstation, and "which OS build" is the first question asked
// about a driver-shaped anomaly.
//
// Deliberately NOT here: hostname, username, machine serial, MAC address,
// DMI product_uuid, IORegistry product-id, and any name the user gave the
// device.  Result files get uploaded and shared -- nothing in one should
// identify the unit or the person who ran it.  Every field below is a
// property of the hardware model or the OS build, not of its owner.
//
// Fields the platform cannot answer are left empty / zero; the writer omits
// them rather than emitting a placeholder.

// What the probe reads, raw.  The document's `system` is built from it
// (probeSystem); the host CPU's brand and thread count are what the host CPU
// processor starts from when no CPU backend ran.
struct HostInfo {
    std::string os;             // "Macintosh" | "Linux ARM64" | "Win64" | …
    std::string osVersion;      // "26.6.2", "6.8.0-40-generic", "22631"
    std::string arch;           // "arm64" | "x86_64" | "x86" | "arm"
    std::string cpu;            // marketing name of the host CPU
    unsigned    logicalCores = 0;
    std::uint64_t memoryBytes = 0;
};

// Probe the host.  Cheap, but not free (a few sysctls / one /proc read), so
// call it once per run rather than per device.
HostInfo probeHost();

// ── `system` ───────────────────────────────────────────────────────────────
//
// The run's machine as the document records it (docs/format-v4.md, System).
// Vocabulary-valued fields hold their strings (include/common/vocabulary.h),
// so a newer file's values survive a read.

struct OsInfo {
    std::string name;           // "macOS" (today the probe's "Macintosh"), "Android", "Linux"
    std::string version;        // the user-visible release: "27.2", "16", "24.04"
    std::string build;          // the OS build: "25C56"
    std::string kernel;         // the kernel release, where it is not the version
    std::string distribution;   // Linux: /etc/os-release's PRETTY_NAME
};

struct SystemInfo {
    // FormFactor, FormFactorDetail and FormFactorEvidence strings.
    std::string formFactor;
    std::string formFactorDetail;
    std::string formFactorFrom;
    bool        isVirtual  = false;   // a virtual machine
    bool        translated = false;   // an x86 binary under Rosetta or Prism

    std::string manufacturer;   // "Apple", "samsung", DMI sys_vendor
    std::string model;          // the model id: "MacBookPro18,3", "SM-S921B"
    std::string name;           // the OS's marketing name, never a table's
    std::string family;         // DMI product_family, SMBIOS SystemFamily
    std::string board;          // the board, where the product name is an OEM placeholder
    std::string soc;            // "Apple M1 Pro"; the CPU brand where there is no SoC name

    OsInfo        os;
    std::string   arch;         // the binary's: "arm64", "x86_64"
    std::uint64_t memoryBytes = 0;

    std::string ext;            // experimental fields, raw JSON (docs/format-v4.md)
};

// The system as far as today's probe goes: the OS name and version, the
// binary's architecture, the CPU brand as the SoC, and the memory.
SystemInfo probeSystem();

// ── `conditions` ───────────────────────────────────────────────────────────
//
// What the machine was doing that changes every number: on battery or not,
// a low-power mode or power profile, the Linux CPU governor, how hot it was
// at the start and at the end.
struct RunConditions {
    std::string powerSource;    // PowerSource: "ac" | "battery"
    int         lowPowerMode = -1;   // -1 unknown, 0 off, 1 on
    std::string powerProfile;   // the OS's own name: "balanced", "performance", "low-power"
    std::string cpuGovernor;    // Linux: "schedutil", "performance"
    std::string thermalStart;   // the OS's thermal state at the start: "nominal", "fair", …
    std::string thermalEnd;     // and at the end
    std::string ext;            // experimental fields, raw JSON

    bool empty() const
    {
        return powerSource.empty() && lowPowerMode < 0 && powerProfile.empty() &&
               cpuGovernor.empty() && thermalStart.empty() && thermalEnd.empty() &&
               ext.empty();
    }
};

#endif  // XPUPEAK_HOST_INFO_H
