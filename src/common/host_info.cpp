#include <common/host_info.h>
#include <common/common.h>

#include <fstream>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/utsname.h>
#else
#include <sys/utsname.h>
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#endif

namespace {

// The machine's instruction set, from the compiler rather than from a runtime
// query: this is the binary's own architecture, which is what a reader wants
// when comparing two runs (an x86_64 build under Rosetta is a different beast
// from a native arm64 one, and only the compiler knows which this is).
const char *hostArch()
{
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#elif defined(__arm__) || defined(_M_ARM)
    return "arm";
#elif defined(__riscv) && __riscv_xlen == 64
    return "riscv64";
#else
    return "";
#endif
}

#if defined(__APPLE__)

std::string sysctlStr(const char *name)
{
    size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0)
        return "";
    std::string out(len, '\0');
    if (sysctlbyname(name, &out[0], &len, nullptr, 0) != 0)
        return "";
    // sysctl reports the length including the terminator.
    while (!out.empty() && out.back() == '\0')
        out.pop_back();
    return out;
}

#elif !defined(_WIN32)

// First "key<sep>value" line of /proc/cpuinfo, value trimmed.
std::string cpuinfoValue(const std::string &text, const char *key)
{
    std::istringstream in(text);
    std::string line;
    const std::string k(key);
    while (std::getline(in, line))
    {
        if (line.compare(0, k.size(), k) != 0) continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        size_t b = line.find_first_not_of(" \t", colon + 1);
        if (b == std::string::npos) continue;
        size_t e = line.find_last_not_of(" \t\r");
        return line.substr(b, e - b + 1);
    }
    return "";
}

#endif

#if defined(_WIN32)

std::string regString(const char *subkey, const char *value)
{
    char  buf[256];
    DWORD size = sizeof(buf);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, subkey, value, RRF_RT_REG_SZ,
                     nullptr, buf, &size) != ERROR_SUCCESS)
        return "";
    return std::string(buf);
}

#endif

} // namespace

HostInfo probeHost()
{
    HostInfo h;
    h.os   = OS_NAME;
    h.arch = hostArch();

    h.logicalCores = std::thread::hardware_concurrency();
    h.memoryBytes  = clpeak::systemMemoryBytes();

#if defined(__APPLE__)
    // kern.osproductversion is the user-visible number ("26.6.2"); the Darwin
    // kernel release underneath it is no use to anyone reading a result file.
    h.osVersion = sysctlStr("kern.osproductversion");
    if (h.osVersion.empty())
    {
        utsname u{};
        if (uname(&u) == 0) h.osVersion = u.release;
    }
    h.cpu = sysctlStr("machdep.cpu.brand_string");

#elif defined(_WIN32)
    const char *cv = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
    h.osVersion = regString(cv, "DisplayVersion");
    if (h.osVersion.empty()) h.osVersion = regString(cv, "CurrentBuild");
    h.cpu = regString("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      "ProcessorNameString");

#else
    utsname u{};
    if (uname(&u) == 0) h.osVersion = u.release;
#if defined(__ANDROID__)
    // The Android release ("14") is what a phone's owner recognises; the
    // kernel release is what uname gives, so prefer the property and keep the
    // kernel as the fallback above.
    char prop[PROP_VALUE_MAX] = {0};
    if (__system_property_get("ro.build.version.release", prop) > 0 && prop[0])
        h.osVersion = prop;
#endif
    {
        std::ifstream f("/proc/cpuinfo");
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string cpuinfo = ss.str();
        h.cpu = cpuinfoValue(cpuinfo, "model name");
        // ARM boards usually have no "model name" line at all.
        if (h.cpu.empty()) h.cpu = cpuinfoValue(cpuinfo, "Hardware");
    }
#endif

    return h;
}
