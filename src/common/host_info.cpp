#include <common/host_info.h>
#include <common/common.h>
#include <common/json.h>
#include <common/vocabulary.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <powrprof.h>
#elif defined(__APPLE__)
#include <TargetConditionals.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>
#else
#include <dirent.h>
#include <sys/utsname.h>
#if defined(__ANDROID__)
#include <dlfcn.h>
#include <sys/system_properties.h>
#endif
#endif

// What a probe never reads, so that it can never be written (host_info.h):
// serial numbers (DMI product_serial, board_serial, ro.serialno), DMI
// product_uuid, IORegistry product-id and IOPlatformUUID, hostnames, MAC
// addresses, and names the user set (Android's device_name, iOS's
// UIDevice.name, a simulator's SIMULATOR_DEVICE_NAME).

#if defined(__APPLE__)
// host_info_apple.mm: what only IOKit and Foundation answer.
namespace xpupeak {
namespace apple {
std::string productProperty(const char *key);   // macOS IODeviceTree:/product; "" elsewhere
int         internalBattery();                  // 1, 0, or -1 unknown; macOS only
std::string providingPowerSource();             // "ac", "battery" or ""; macOS only
int         lowPowerMode();                     // 1, 0, or -1 unknown
std::string thermalState();                     // "nominal" … "critical", or ""
} // namespace apple
} // namespace xpupeak
#endif

namespace {

using xpupeak::FormFactor;
using xpupeak::FormFactorDetail;
using xpupeak::FormFactorEvidence;
using xpupeak::toString;

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

[[maybe_unused]] bool binaryIsX86()
{
    const std::string a = hostArch();
    return a == "x86_64" || a == "x86";
}

[[maybe_unused]] std::string trim(const std::string &s)
{
    const char *ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const size_t e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

[[maybe_unused]] std::string lower(std::string s)
{
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

[[maybe_unused]] bool startsWith(const std::string &s, const char *prefix)
{
    return s.compare(0, std::strlen(prefix), prefix) == 0;
}

[[maybe_unused]] bool contains(const std::string &s, const char *needle)
{
    return s.find(needle) != std::string::npos;
}

void setForm(SystemInfo &s, FormFactor f, FormFactorEvidence from)
{
    s.formFactor     = toString(f);
    s.formFactorFrom = toString(from);
}

[[maybe_unused]] void setDetail(SystemInfo &s, FormFactorDetail d)
{
    s.formFactorDetail = toString(d);
}

// ── DMI and SMBIOS ─────────────────────────────────────────────────────────
//
// Linux's /sys/class/dmi/id and Windows' SMBIOS registry copy are the same
// firmware table, read by the same rules.

#if defined(_WIN32) || (defined(__linux__) && !defined(__ANDROID__))

// What firmware writes where the OEM wrote nothing: never a model, a name or
// a maker.  Compared trimmed and case-blind; a string with no letter or
// digit in it at all is one too.
bool isPlaceholder(const std::string &raw)
{
    const std::string s = lower(trim(raw));
    if (std::none_of(s.begin(), s.end(),
                     [](unsigned char c) { return std::isalnum(c) != 0; }))
        return true;
    static const char *const kPlaceholders[] = {
        "to be filled by o.e.m.", "to be filled by oem", "system product name",
        "system manufacturer", "system name", "system version", "system sku",
        "system family", "default string", "default", "not applicable",
        "not specified", "not available", "none", "n/a", "na", "oem", "o.e.m.",
        "unknown", "undefined", "invalid", "empty", "type1productconfigid",
        "type2 - board vendor name1", "type2 - board product name1",
        "base board product name", "base board manufacturer",
        "baseboard manufacturer", "chassis manufacturer", "x.x", "0", "1",
        "0123456789", "123456789",
    };
    for (const char *p : kPlaceholders)
        if (s == p) return true;
    return false;
}

// A hypervisor's virtual hardware, by the maker and product it reports.
// Not the CPUID hypervisor bit on Windows: virtualization-based security
// sets it on bare metal there.
bool namesVirtualMachine(const std::string &vendorRaw, const std::string &productRaw)
{
    const std::string vendor = lower(trim(vendorRaw)), product = lower(trim(productRaw));
    static const char *const kVendors[] = {
        "qemu", "vmware", "innotek", "xen", "parallels", "bochs", "bhyve",
        "openstack", "digitalocean", "ovirt", "nutanix", "red hat",
        "alibaba cloud", "tencent cloud",
    };
    for (const char *v : kVendors)
        if (startsWith(vendor, v)) return true;
    // EC2's bare-metal instances report the same maker as its VMs.
    if (startsWith(vendor, "amazon ec2") &&
        !(product.size() > 6 && product.compare(product.size() - 6, 6, ".metal") == 0))
        return true;
    // "Virtual Machine" (Hyper-V), "VMware Virtual Platform", "VirtualBox",
    // "Parallels ARM Virtual Machine", "Apple Virtualization …".
    static const char *const kProducts[] = {
        "virtual", "kvm", "qemu", "bochs", "bhyve", "hvm domu",
        "google compute engine", "droplet", "openstack", "vserver",
    };
    for (const char *p : kProducts)
        if (contains(product, p)) return true;
    return product == "ahv";
}

// Lenovo is the one maker that keeps its marketing name out of the product
// name: that holds the machine type ("21DD"), and the product version holds
// "ThinkPad X1 Carbon Gen 10" -- in DMI and in SMBIOS alike.  Another maker's
// version is a version ("1.0", "A01"), never a name.
std::string lenovoName(const std::string &vendor, const std::string &version)
{
    if (lower(trim(vendor)) != "lenovo" || isPlaceholder(version)) return "";
    const auto letters = std::count_if(version.begin(), version.end(),
                                       [](unsigned char c) { return std::isalpha(c) != 0; });
    return letters >= 3 ? trim(version) : "";
}

// The system's identity from the SMBIOS strings.  A custom build's product
// name is a placeholder: then there is no model, and the board stands in for
// it, its maker for the system's.
void fromSmbios(SystemInfo &s, const std::string &vendor, const std::string &product,
                const std::string &family, const std::string &version,
                const std::string &boardVendor, const std::string &boardName)
{
    if (!isPlaceholder(vendor)) s.manufacturer = trim(vendor);
    else if (!isPlaceholder(boardVendor)) s.manufacturer = trim(boardVendor);
    if (!isPlaceholder(product)) s.model = trim(product);
    else if (!isPlaceholder(boardName)) s.board = trim(boardName);
    if (!isPlaceholder(family)) s.family = trim(family);
    s.name = lenovoName(vendor, version);
    s.isVirtual = s.isVirtual || namesVirtualMachine(vendor, product) ||
                  namesVirtualMachine(boardVendor, boardName);
}

// ACPI's preferred power-management profile, which Windows reports as the
// platform role (POWER_PLATFORM_ROLE has the same values) and Linux as
// /sys/firmware/acpi/pm_profile.  0 is unspecified.
bool formFromPlatformRole(int role, SystemInfo &s)
{
    switch (role)
    {
    case 1:  // desktop
    case 3:  // workstation
    case 6:  // appliance PC
        setForm(s, FormFactor::Desktop, FormFactorEvidence::PlatformRole);
        return true;
    case 2:  // mobile
        setForm(s, FormFactor::Laptop, FormFactorEvidence::PlatformRole);
        return true;
    case 4:  // enterprise server
    case 5:  // SOHO server
    case 7:  // performance server
        setForm(s, FormFactor::Server, FormFactorEvidence::PlatformRole);
        return true;
    case 8:  // slate: a Windows tablet, which runs what a laptop runs
        setForm(s, FormFactor::Laptop, FormFactorEvidence::PlatformRole);
        setDetail(s, FormFactorDetail::Tablet);
        return true;
    default:
        return false;
    }
}

#endif  // SMBIOS

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

[[maybe_unused]] int sysctlInt(const char *name)
{
    int v = 0;
    size_t len = sizeof(v);
    if (sysctlbyname(name, &v, &len, nullptr, 0) != 0) return -1;
    return v;
}

[[maybe_unused]] std::string envStr(const char *name)
{
    const char *v = std::getenv(name);
    return v ? trim(v) : "";
}

#elif defined(_WIN32)

std::string regString(const char *subkey, const char *value)
{
    char  buf[512];
    DWORD size = sizeof(buf);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, subkey, value, RRF_RT_REG_SZ,
                     nullptr, buf, &size) != ERROR_SUCCESS)
        return "";
    return trim(std::string(buf));
}

bool regDword(const char *subkey, const char *value, DWORD &out)
{
    DWORD size = sizeof(out);
    return RegGetValueA(HKEY_LOCAL_MACHINE, subkey, value, RRF_RT_REG_DWORD,
                        nullptr, &out, &size) == ERROR_SUCCESS;
}

const char *const kCurrentVersion = "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion";
const char *const kSmbios         = "HARDWARE\\DESCRIPTION\\System\\BIOS";

std::string guidText(const GUID &g)
{
    char buf[40];
    std::snprintf(buf, sizeof buf, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                  static_cast<unsigned long>(g.Data1), g.Data2, g.Data3, g.Data4[0], g.Data4[1],
                  g.Data4[2], g.Data4[3], g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7]);
    return buf;
}

// A power plan or slider overlay by Windows' own name for it.  A plan the
// user made has a GUID of its own, unique to the install -- an identifier,
// so it is never written: any GUID not listed reads "custom".
std::string schemeWord(const GUID &g, bool overlay)
{
    struct Scheme { const char *guid; const char *word; };
    static const Scheme kPlans[] = {
        {"381b4222-f694-41f0-9685-ff5bb260df2e", "balanced"},
        {"8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c", "high-performance"},
        {"a1841308-3541-4fab-bc81-f71556f20b4a", "power-saver"},
        {"e9a42b02-d5df-448d-aa00-03f14749eb61", "ultimate-performance"},
    };
    static const Scheme kOverlays[] = {
        {"961cc777-2547-4f9d-8174-7d86181b8a7a", "best-power-efficiency"},
        {"00000000-0000-0000-0000-000000000000", "balanced"},
        {"3af9b8d9-7c97-431d-ad78-34a8bfea439f", "better-performance"},
        {"ded574b5-45a0-4f42-8737-46345c09c238", "best-performance"},
    };
    const Scheme *const list = overlay ? kOverlays : kPlans;
    const size_t count = overlay ? sizeof kOverlays / sizeof kOverlays[0]
                                 : sizeof kPlans / sizeof kPlans[0];
    const std::string text = guidText(g);
    for (size_t i = 0; i < count; i++)
        if (text == list[i].guid) return list[i].word;
    return "custom";
}

// The power plan, and under Balanced -- the only plan the power-mode slider
// works on -- the slider's overlay.
std::string windowsPowerProfile()
{
    std::string plan;
    GUID *active = nullptr;
    if (PowerGetActiveScheme(nullptr, &active) == ERROR_SUCCESS && active)
    {
        plan = schemeWord(*active, /*overlay=*/false);
        LocalFree(active);
    }
    if (plan != "balanced") return plan;
    // Windows 10 1709 and later; looked up so an older powrprof still loads.
    using Overlay = DWORD(WINAPI *)(GUID *);
    HMODULE powrprof = GetModuleHandleA("powrprof.dll");
    auto effectiveOverlay = powrprof ? reinterpret_cast<Overlay>(reinterpret_cast<void *>(
                                           GetProcAddress(powrprof, "PowerGetEffectiveOverlayScheme")))
                                     : nullptr;
    GUID overlay{};
    if (effectiveOverlay && effectiveOverlay(&overlay) == ERROR_SUCCESS)
        return schemeWord(overlay, /*overlay=*/true);
    return plan;
}

// An x86 or x64 binary on an ARM64 Windows runs under Prism.
// IsWow64Process2 (Windows 10 1709) names the machine underneath.
bool windowsTranslated()
{
    if (!binaryIsX86()) return false;
    using Wow64Process2 = BOOL(WINAPI *)(HANDLE, USHORT *, USHORT *);
    HMODULE kernel32 = GetModuleHandleA("kernel32.dll");
    auto isWow64Process2 = kernel32 ? reinterpret_cast<Wow64Process2>(reinterpret_cast<void *>(
                                          GetProcAddress(kernel32, "IsWow64Process2")))
                                    : nullptr;
    USHORT process = 0, native = 0;
    if (!isWow64Process2 || !isWow64Process2(GetCurrentProcess(), &process, &native))
        return false;
    return native == 0xAA64;   // IMAGE_FILE_MACHINE_ARM64
}

#else  // Linux, Android and the other POSIX systems

std::string readFile(const std::string &path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// One sysfs value: a single line.
std::string readLine(const std::string &path)
{
    return trim(readFile(path));
}

// A device-tree property: NUL-separated strings.
[[maybe_unused]] std::vector<std::string> readDtStrings(const std::string &path)
{
    std::vector<std::string> out;
    const std::string text = readFile(path);
    size_t b = 0;
    while (b < text.size())
    {
        size_t e = text.find('\0', b);
        if (e == std::string::npos) e = text.size();
        const std::string s = trim(text.substr(b, e - b));
        if (!s.empty()) out.push_back(s);
        b = e + 1;
    }
    return out;
}

std::string unameRelease()
{
    utsname u{};
    return uname(&u) == 0 ? std::string(u.release) : "";
}

// The binary's ISA family against the one the kernel runs: an ARM build on
// an x86 kernel runs under a translator (Android's native bridge), and an
// x86 build on an ARM kernel under one that does not hide the machine.
bool translatedByKernel()
{
    utsname u{};
    if (uname(&u) != 0) return false;
    const std::string m = u.machine;
    const bool machineX86 = m == "x86_64" || m == "amd64" ||
                            (m.size() == 4 && m[0] == 'i' && m.compare(2, 2, "86") == 0);
    const bool machineArm = startsWith(m, "aarch64") || startsWith(m, "arm");
    const std::string a = hostArch();
    const bool binaryArm = a == "arm64" || a == "arm";
    return (binaryIsX86() && machineArm) || (binaryArm && machineX86);
}

// The x86 `hypervisor` CPU flag.  Reliable on Linux, which only a hypervisor
// sets it for; Windows is the OS where it is not (see namesVirtualMachine).
bool cpuinfoHasHypervisorFlag(const std::string &cpuinfo)
{
    std::istringstream in(cpuinfo);
    std::string line;
    while (std::getline(in, line))
    {
        const size_t colon = line.find(':');
        if (!startsWith(line, "flags") || colon == std::string::npos) continue;
        std::istringstream words(line.substr(colon + 1));
        std::string w;
        while (words >> w)
            if (w == "hypervisor") return true;
        return false;
    }
    return false;
}

struct PowerSupplies
{
    bool listed        = false;   // /sys/class/power_supply could be read
    bool battery       = false;   // a battery that powers the system
    bool discharging   = false;
    bool charging      = false;   // charging, full or held: external power
    bool adapterOnline = false;   // mains, USB or wireless power present
};

// The supplies that power the system.  A peripheral's battery (a mouse, a
// pen, a game pad) says scope "Device" and is no evidence of anything.
PowerSupplies readPowerSupplies()
{
    PowerSupplies p;
    const std::string root = "/sys/class/power_supply/";
    DIR *dir = opendir(root.c_str());
    if (!dir) return p;
    p.listed = true;
    while (dirent *e = readdir(dir))
    {
        if (e->d_name[0] == '.') continue;
        const std::string base = root + e->d_name + "/";
        const std::string type = readLine(base + "type");
        if (type.empty()) continue;
        if (lower(readLine(base + "scope")) == "device") continue;
        if (type == "Battery")
        {
            p.battery = true;
            const std::string status = readLine(base + "status");
            if (status == "Discharging")
                p.discharging = true;
            else if (status == "Charging" || status == "Full" || status == "Not charging")
                p.charging = true;
        }
        else if (type == "Mains" || type == "Wireless" || startsWith(type, "USB"))
        {
            if (readLine(base + "online") == "1") p.adapterOnline = true;
        }
    }
    closedir(dir);
    return p;
}

// The supplies' answer.  A machine with no battery at all runs from the
// mains -- but only where an absent battery means one: not in a VM, whose
// host may well be on battery, and not on Android, where an app usually
// cannot read the supplies at all.
std::string powerSourceFrom(const PowerSupplies &p, bool noBatteryMeansMains)
{
    if (p.adapterOnline) return toString(xpupeak::PowerSource::Ac);
    if (p.battery)
    {
        if (p.discharging) return toString(xpupeak::PowerSource::Battery);
        if (p.charging) return toString(xpupeak::PowerSource::Ac);
        return "";
    }
    return p.listed && noBatteryMeansMains ? toString(xpupeak::PowerSource::Ac) : "";
}

// The governor of each cpufreq policy -- one per cluster on a big.LITTLE
// part -- in policy order, each distinct one once: "schedutil", or
// "schedutil,performance" where the clusters differ.
std::string cpuGovernor()
{
    const std::string root = "/sys/devices/system/cpu/cpufreq/";
    std::vector<std::pair<int, std::string>> policies;
    if (DIR *dir = opendir(root.c_str()))
    {
        while (dirent *e = readdir(dir))
        {
            const std::string name = e->d_name;
            if (!startsWith(name, "policy")) continue;
            const std::string g = readLine(root + name + "/scaling_governor");
            if (!g.empty()) policies.emplace_back(std::atoi(name.c_str() + 6), g);
        }
        closedir(dir);
    }
    if (policies.empty())
        return readLine("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
    std::sort(policies.begin(), policies.end());
    std::string out;
    std::vector<std::string> seen;
    for (const auto &p : policies)
    {
        if (std::find(seen.begin(), seen.end(), p.second) != seen.end()) continue;
        seen.push_back(p.second);
        if (!out.empty()) out += ",";
        out += p.second;
    }
    return out;
}

#if defined(__ANDROID__)

std::string prop(const char *name)
{
    char buf[PROP_VALUE_MAX] = {0};
    if (__system_property_get(name, buf) <= 0) return "";
    return trim(buf);
}

// The build a person sees in Settings: ro.build.display.id on a production
// build.  A local build's display id carries the builder's name
// ("eng.<user>.<date>"), so such a build says only its AOSP build id.
std::string androidBuild()
{
    const std::string display = prop("ro.build.display.id");
    if (prop("ro.build.type") == "user" && !display.empty() && !contains(display, "eng."))
        return display;
    return prop("ro.build.id");
}

bool androidEmulator(const std::string &characteristics)
{
    const std::string hardware = prop("ro.hardware");
    return prop("ro.kernel.qemu") == "1" || prop("ro.boot.qemu") == "1" ||
           hardware == "ranchu" || hardware == "goldfish" ||
           contains(characteristics, "emulator");
}

// AThermal_getCurrentThermalStatus (API 30), in the NDK's words.  libandroid
// is opened at run time, as thermal.cpp does, so an older API level links.
std::string androidThermalState()
{
    struct AThermalManager;
    using Acquire = AThermalManager *(*)();
    using Release = void (*)(AThermalManager *);
    using Status  = int (*)(AThermalManager *);
    void *lib = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) return "";
    auto acquire = reinterpret_cast<Acquire>(dlsym(lib, "AThermal_acquireManager"));
    auto release = reinterpret_cast<Release>(dlsym(lib, "AThermal_releaseManager"));
    auto status  = reinterpret_cast<Status>(dlsym(lib, "AThermal_getCurrentThermalStatus"));
    std::string out;
    if (acquire && release && status)
        if (AThermalManager *m = acquire())
        {
            static const char *const kStates[] = {"none", "light", "moderate", "severe",
                                                  "critical", "emergency", "shutdown"};
            const int s = status(m);
            if (s >= 0 && s <= 6) out = kStates[s];
            release(m);
        }
    dlclose(lib);
    return out;
}

#elif defined(__linux__)

std::string dmi(const char *field)
{
    return readLine(std::string("/sys/class/dmi/id/") + field);
}

// /etc/os-release, KEY=value per line, the value quoted or not.
std::map<std::string, std::string> osRelease()
{
    std::string text = readFile("/etc/os-release");
    if (text.empty()) text = readFile("/usr/lib/os-release");
    std::map<std::string, std::string> out;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string value = line.substr(eq + 1);
        if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') &&
            value.back() == value.front())
            value = value.substr(1, value.size() - 2);
        std::string unescaped;
        for (size_t i = 0; i < value.size(); i++)
        {
            if (value[i] == '\\' && i + 1 < value.size()) i++;
            unescaped += value[i];
        }
        out[line.substr(0, eq)] = unescaped;
    }
    return out;
}

bool linuxVirtual(const std::string &cpuinfo)
{
    if (namesVirtualMachine(dmi("sys_vendor"), dmi("product_name")) ||
        namesVirtualMachine(dmi("board_vendor"), dmi("board_name")))
        return true;
    // Xen, and z/VM on s390.
    if (!readLine("/sys/hypervisor/type").empty()) return true;
    if (cpuinfoHasHypervisorFlag(cpuinfo)) return true;
    // QEMU's and crosvm's ARM machine, which has no DMI.
    for (const std::string &c : readDtStrings("/proc/device-tree/compatible"))
        if (c == "linux,dummy-virt") return true;
    return false;
}

bool formFromChassis(int type, SystemInfo &s)
{
    FormFactor f = FormFactor::Desktop;
    switch (type)
    {
    case 8: case 9: case 10: case 14: case 31: case 32:   // portable … detachable
        f = FormFactor::Laptop;
        break;
    case 17: case 23: case 25: case 28: case 29:          // server, rack, blade
        f = FormFactor::Server;
        break;
    case 3: case 4: case 5: case 6: case 7: case 13: case 15: case 35: case 36:
        f = FormFactor::Desktop;                          // desktop … stick PC
        break;
    case 30:                                              // tablet
        setForm(s, FormFactor::Mobile, FormFactorEvidence::Chassis);
        setDetail(s, FormFactorDetail::Tablet);
        return true;
    case 33: case 34:                                     // IoT gateway, embedded PC
        setForm(s, FormFactor::Desktop, FormFactorEvidence::Chassis);
        setDetail(s, FormFactorDetail::Embedded);
        return true;
    default:                                              // other, unknown, …
        return false;
    }
    setForm(s, f, FormFactorEvidence::Chassis);
    return true;
}

#endif  // __ANDROID__ / __linux__

#endif  // platform

// The host CPU's marketing name.
std::string cpuBrand()
{
#if defined(__APPLE__)
    return sysctlStr("machdep.cpu.brand_string");
#elif defined(_WIN32)
    return regString("HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString");
#else
    // First "key<sep>value" line of /proc/cpuinfo, value trimmed.  ARM
    // kernels usually have no "model name" at all; Android's often say
    // "Hardware".
    std::istringstream in(readFile("/proc/cpuinfo"));
    std::string line, hardware;
    while (std::getline(in, line))
    {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon));
        if (key == "model name") return trim(line.substr(colon + 1));
        if (key == "Hardware" && hardware.empty()) hardware = trim(line.substr(colon + 1));
    }
    return hardware;
#endif
}

OsInfo probeOs()
{
    OsInfo os;
#if defined(__APPLE__)
#if TARGET_OS_OSX
    os.name = "macOS";
#else
    os.name = "iOS";
#endif
#if TARGET_OS_SIMULATOR
    // The simulator runs on the Mac's kernel, whose sysctls are the Mac's
    // macOS; the iOS it simulates says itself in the environment.
    os.version = envStr("SIMULATOR_RUNTIME_VERSION");
    os.build   = envStr("SIMULATOR_RUNTIME_BUILD_VERSION");
#else
    // kern.osproductversion is the user-visible number ("27.2"); the build
    // ("26B123") and the Darwin release under it are their own fields.
    os.version = sysctlStr("kern.osproductversion");
    os.build   = sysctlStr("kern.osversion");
#endif
    os.kernel = sysctlStr("kern.osrelease");

#elif defined(_WIN32)
    // What Settings > About says: Windows 11, version 24H2, OS build
    // 26100.2314.  The registry's ProductName still says "Windows 10" on 11,
    // whose builds start at 22000.
    const std::string build = regString(kCurrentVersion, "CurrentBuild");
    std::string feature = regString(kCurrentVersion, "DisplayVersion");
    if (feature.empty()) feature = regString(kCurrentVersion, "ReleaseId");
    DWORD ubr = 0;
    os.build = build;
    if (!build.empty() && regDword(kCurrentVersion, "UBR", ubr))
        os.build += "." + std::to_string(ubr);
    if (contains(lower(regString(kCurrentVersion, "InstallationType")), "server"))
    {
        os.name    = "Windows Server";
        os.version = feature;
    }
    else
    {
        os.name = "Windows";
        const int b = std::atoi(build.c_str());
        const std::string release = b >= 22000 ? "11" : b >= 10240 ? "10" : "";
        os.version = release.empty() ? feature : feature.empty() ? release : release + " " + feature;
    }

#elif defined(__ANDROID__)
    os.name    = "Android";
    os.version = prop("ro.build.version.release");
    os.build   = androidBuild();
    os.kernel  = unameRelease();

#elif defined(__linux__)
    os.name = "Linux";
    const auto rel = osRelease();
    auto field = [&](const char *k) {
        const auto it = rel.find(k);
        return it == rel.end() ? std::string() : trim(it->second);
    };
    os.version      = field("VERSION_ID");
    os.distribution = field("PRETTY_NAME");
    if (os.distribution.empty())
        os.distribution = trim(field("NAME") + " " + field("VERSION"));
    os.kernel = unameRelease();

#else
    utsname u{};
    if (uname(&u) == 0)
    {
        os.name    = u.sysname;
        os.version = u.release;
    }
#endif
    return os;
}

// The OS's thermal state, in its own words.  Only Apple's and Android's OSes
// keep one; a desktop OS has temperatures, not a state.
std::string thermalStateNow()
{
#if defined(__APPLE__)
    return xpupeak::apple::thermalState();
#elif defined(__ANDROID__)
    return androidThermalState();
#else
    return "";
#endif
}

// ── Facts from the app ─────────────────────────────────────────────────────

std::mutex  g_appFactsMutex;
std::string g_appPowerSource;   // a PowerSource word, or empty
int         g_appLowPowerMode = -1;

} // namespace

HostInfo probeHost()
{
    HostInfo h;
    const OsInfo os = probeOs();
    h.os        = os.name;
    h.osVersion = os.version;
    h.arch      = hostArch();
    h.cpu       = cpuBrand();

    h.logicalCores = std::thread::hardware_concurrency();
    h.memoryBytes  = xpupeak::systemMemoryBytes();
    return h;
}

SystemInfo probeSystem()
{
    SystemInfo s;
    s.os          = probeOs();
    s.arch        = hostArch();
    s.memoryBytes = xpupeak::systemMemoryBytes();

#if defined(__APPLE__)
    s.manufacturer = "Apple";
#if TARGET_OS_OSX
    s.model = sysctlStr("hw.model");
    // Apple silicon's device tree names the product and its SoC; an Intel
    // Mac's has no product node, and its SoC is its CPU.
    s.name = xpupeak::apple::productProperty("product-name");
    s.soc  = xpupeak::apple::productProperty("product-soc-name");
    const int battery = xpupeak::apple::internalBattery();
    if (battery >= 0)
        setForm(s, battery ? FormFactor::Laptop : FormFactor::Desktop, FormFactorEvidence::Battery);
    s.isVirtual  = sysctlInt("kern.hv_vmm_present") == 1;
    s.translated = sysctlInt("sysctl.proc_translated") == 1;
#else
    // The model id names the device; iOS gives no marketing name.  In the
    // simulator the hardware is the Mac's, which is what `virtual` says --
    // the SoC below is the Mac's too, because that is what ran.
#if TARGET_OS_SIMULATOR
    s.model     = envStr("SIMULATOR_MODEL_IDENTIFIER");
    s.isVirtual = true;
#else
    s.model = sysctlStr("hw.machine");
#endif
    setForm(s, FormFactor::Mobile, FormFactorEvidence::Model);
    if (startsWith(s.model, "iPhone"))
        setDetail(s, FormFactorDetail::Phone);
    else if (startsWith(s.model, "iPad"))
        setDetail(s, FormFactorDetail::Tablet);
#endif

#elif defined(_WIN32)
    fromSmbios(s, regString(kSmbios, "SystemManufacturer"), regString(kSmbios, "SystemProductName"),
               regString(kSmbios, "SystemFamily"), regString(kSmbios, "SystemVersion"),
               regString(kSmbios, "BaseBoardManufacturer"), regString(kSmbios, "BaseBoardProduct"));
    if (!formFromPlatformRole(static_cast<int>(PowerDeterminePlatformRoleEx(POWER_PLATFORM_ROLE_V2)), s))
    {
        // An unspecified role: a battery, where Windows knows (128 is "no
        // system battery", 255 "unknown").
        SYSTEM_POWER_STATUS ps{};
        if (GetSystemPowerStatus(&ps) && ps.BatteryFlag != 255)
            setForm(s, (ps.BatteryFlag & 128) ? FormFactor::Desktop : FormFactor::Laptop,
                    FormFactorEvidence::Battery);
    }
    s.translated = windowsTranslated();

#elif defined(__ANDROID__)
    s.manufacturer = prop("ro.product.manufacturer");
    s.model        = prop("ro.product.model");
    s.codename     = prop("ro.product.device");
    // A brand of the maker's own name says nothing the maker does not.
    const std::string brand = prop("ro.product.brand");
    if (lower(brand) != lower(s.manufacturer)) s.brand = brand;
    // The marketing name, where the OEM sets one: the AOSP property and its
    // partition copies, and the two OEM spellings in wide use.
    for (const char *key : {"ro.product.marketname", "ro.product.vendor.marketname",
                            "ro.product.odm.marketname", "ro.product.product.marketname",
                            "ro.config.marketing_name", "ro.vendor.oplus.market.name"})
    {
        s.name = prop(key);
        if (!s.name.empty()) break;
    }
    // Android 12 and later: "QTI" + "SM8650", "Google" + "GS201".
    s.soc = trim(prop("ro.soc.manufacturer") + " " + prop("ro.soc.model"));
    const std::string characteristics = lower(prop("ro.build.characteristics"));
    setForm(s, FormFactor::Mobile, characteristics.empty() ? FormFactorEvidence::Os
                                                           : FormFactorEvidence::Characteristics);
    if (contains(characteristics, "tablet"))
        setDetail(s, FormFactorDetail::Tablet);
    else if (!contains(characteristics, "tv") && !contains(characteristics, "watch") &&
             !contains(characteristics, "automotive"))
        setDetail(s, FormFactorDetail::Phone);
    const std::string cpuinfo = readFile("/proc/cpuinfo");
    s.isVirtual  = androidEmulator(characteristics) || cpuinfoHasHypervisorFlag(cpuinfo);
    s.translated = translatedByKernel();

#elif defined(__linux__)
    const std::string vendor = dmi("sys_vendor"), product = dmi("product_name"),
                      boardName = dmi("board_name");
    const bool hasDmi = !vendor.empty() || !product.empty() || !boardName.empty();
    if (hasDmi)
        fromSmbios(s, vendor, product, dmi("product_family"), dmi("product_version"),
                   dmi("board_vendor"), boardName);
    // An ARM board with no DMI names itself in its device tree: "Raspberry
    // Pi 5 Model B Rev 1.0", made by the vendor its first `compatible` names.
    const std::vector<std::string> dtModel = readDtStrings("/proc/device-tree/model");
    const bool deviceTree = !dtModel.empty();
    if (!hasDmi && deviceTree)
    {
        s.model = dtModel.front();
        const std::vector<std::string> compatible = readDtStrings("/proc/device-tree/compatible");
        if (!compatible.empty())
        {
            const size_t comma = compatible.front().find(',');
            if (comma != std::string::npos && comma > 0)
                s.manufacturer = compatible.front().substr(0, comma);
        }
    }
    const std::string cpuinfo = readFile("/proc/cpuinfo");
    s.isVirtual  = s.isVirtual || linuxVirtual(cpuinfo);
    s.translated = translatedByKernel();

    // The chassis, the ACPI platform role, then a battery.  A device-tree
    // board with none of those is an embedded board (a Raspberry Pi, a
    // Jetson): one with a battery is a laptop (an ARM Chromebook, Asahi on a
    // MacBook), which is why the battery is asked first.
    const std::string chassis = dmi("chassis_type");
    const std::string role    = readLine("/sys/firmware/acpi/pm_profile");
    if (!(!chassis.empty() && formFromChassis(std::atoi(chassis.c_str()), s)) &&
        !(!role.empty() && formFromPlatformRole(std::atoi(role.c_str()), s)))
    {
        if (readPowerSupplies().battery)
            setForm(s, FormFactor::Laptop, FormFactorEvidence::Battery);
        else if (deviceTree && !hasDmi)
        {
            setForm(s, FormFactor::Desktop, FormFactorEvidence::DeviceTree);
            setDetail(s, FormFactorDetail::Embedded);
        }
    }
#endif

    // The SoC's name where the OS gives one, else the CPU's brand: Apple's
    // brand string is the SoC's ("Apple M1 Pro"), and a PC's SoC is its CPU.
    if (s.soc.empty()) s.soc = cpuBrand();
    if (s.formFactor.empty()) setForm(s, FormFactor::Desktop, FormFactorEvidence::Default);
    return s;
}

RunConditions probeConditions()
{
    RunConditions c;
#if defined(__APPLE__)
#if TARGET_OS_OSX
    c.powerSource = xpupeak::apple::providingPowerSource();
#endif
    // iOS's power source is UIKit's (UIDevice.batteryState), which the app
    // reads and passes in (setHostFacts).
    c.lowPowerMode = xpupeak::apple::lowPowerMode();

#elif defined(_WIN32)
    SYSTEM_POWER_STATUS ps{};
    if (GetSystemPowerStatus(&ps))
    {
        if (ps.ACLineStatus == 1) c.powerSource = toString(xpupeak::PowerSource::Ac);
        else if (ps.ACLineStatus == 0) c.powerSource = toString(xpupeak::PowerSource::Battery);
        // Energy saver (battery saver before Windows 11), Windows 10 and later.
        c.lowPowerMode = ps.SystemStatusFlag ? 1 : 0;
    }
    c.powerProfile = windowsPowerProfile();

#elif defined(__ANDROID__)
    // Battery saver is PowerManager's, which only the app reads
    // (setHostFacts); the supplies answer where SELinux lets an app see them.
    c.powerSource = powerSourceFrom(readPowerSupplies(), /*noBatteryMeansMains=*/false);
    c.cpuGovernor = cpuGovernor();

#elif defined(__linux__)
    c.powerSource  = powerSourceFrom(readPowerSupplies(),
                                     /*noBatteryMeansMains=*/!linuxVirtual(readFile("/proc/cpuinfo")));
    c.powerProfile = readLine("/sys/firmware/acpi/platform_profile");
    c.cpuGovernor  = cpuGovernor();
#endif

    c.thermalStart = thermalStateNow();

    std::lock_guard<std::mutex> lock(g_appFactsMutex);
    if (!g_appPowerSource.empty()) c.powerSource = g_appPowerSource;
    if (g_appLowPowerMode >= 0) c.lowPowerMode = g_appLowPowerMode;
    return c;
}

void finishConditions(RunConditions &c)
{
    c.thermalEnd = thermalStateNow();
}

void setHostFacts(const std::string &json)
{
    std::string power;
    int lowPower = -1;
    std::string error;
    const JsonValue v = json.empty() ? JsonValue() : jsonParse(json, &error);
    if (error.empty() && v.isObject())
    {
        xpupeak::PowerSource source;
        if (const JsonValue *p = v.find("power_source");
            p && p->type() == JsonValue::Type::String && xpupeak::fromString(p->asString(), source))
            power = toString(source);
        if (const JsonValue *l = v.find("low_power_mode"); l && l->type() == JsonValue::Type::Bool)
            lowPower = l->asBool() ? 1 : 0;
    }
    std::lock_guard<std::mutex> lock(g_appFactsMutex);
    g_appPowerSource  = power;
    g_appLowPowerMode = lowPower;
}
