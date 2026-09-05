#ifndef CLPEAK_HOST_INFO_H
#define CLPEAK_HOST_INFO_H

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
// Deliberately NOT here: hostname, username, machine serial, MAC address.
// Result files get uploaded and shared -- nothing in one should identify the
// person who ran it.  Every field below is a property of the hardware or the
// OS build, not of its owner.
//
// Fields the platform cannot answer are left empty / zero; the writer omits
// them rather than emitting a placeholder.

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

#endif  // CLPEAK_HOST_INFO_H
