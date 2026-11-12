#include <common/hardware_link.h>
#include <common/common.h>      // XPUPEAK_VLOG
#include <common/host_info.h>   // probeHost
#include <common/options.h>     // backendFromName, backendInfo
#include <common/run_checkpoint.h>  // the run's evidence key

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>

namespace xpupeak {

// ── Identifiers ────────────────────────────────────────────────────────────

namespace {

const char kHex[] = "0123456789abcdef";

std::string hexOf(const unsigned char *bytes, size_t n)
{
    std::string out;
    out.reserve(n * 2);
    for (size_t i = 0; i < n; i++)
    {
        out += kHex[bytes[i] >> 4];
        out += kHex[bytes[i] & 0xf];
    }
    return out;
}

bool allZero(const unsigned char *bytes, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (bytes[i]) return false;
    return true;
}

int hexDigit(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// The hex digits of `s`, lower-case, with every separator dropped; "" when
// anything else is in it.
std::string hexDigitsOf(const std::string &s)
{
    std::string out;
    for (char c : s)
    {
        if (hexDigit(c) >= 0)
            out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        else if (c != '-' && c != ':' && c != ' ' && c != '{' && c != '}')
            return "";
    }
    return out;
}

bool isZeroText(const std::string &hex)
{
    return hex.find_first_not_of('0') == std::string::npos;
}

// The canonical form of one raw identifier of `kind`, or "" for none.
std::string canonicalIdentifier(const std::string &kind, const std::string &raw)
{
    if (kind == "pci_bus") return pciBusFromText(raw);
    const bool prefixed = raw.size() > 2 && raw[0] == '0' && (raw[1] == 'x' || raw[1] == 'X');
    std::string hex = hexDigitsOf(prefixed ? raw.substr(2) : raw);
    if (hex.empty() || isZeroText(hex)) return "";
    // A number written short ("1f2") is the same number written long.
    if (kind == "luid" || kind == "registry")
    {
        if (hex.size() > 16) return "";
        hex.insert(0, 16 - hex.size(), '0');
    }
    else if (hex.size() != 32)
    {
        return "";   // a UUID is 16 bytes
    }
    return hex;
}

} // namespace

std::string uuidText(const unsigned char bytes[16])
{
    if (allZero(bytes, 16)) return "";
    // ROCm's runtime answers "GPU-XX" for a GPU with no unique id, the same
    // for every such card, and two cards with one UUID would merge.
    static const unsigned char kNoUniqueId[] = {'G', 'P', 'U', '-', 'X', 'X'};
    if (std::memcmp(bytes, kNoUniqueId, sizeof(kNoUniqueId)) == 0 &&
        allZero(bytes + sizeof(kNoUniqueId), 16 - sizeof(kNoUniqueId)))
        return "";
    return hexOf(bytes, 16);
}

std::string luidText(const unsigned char bytes[8])
{
    std::uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | bytes[i];
    return v ? registryText(v) : std::string();
}

std::string luidText(std::uint32_t lowPart, std::int32_t highPart)
{
    const std::uint64_t v = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(highPart)) << 32) |
                            lowPart;
    return v ? registryText(v) : std::string();
}

std::string pciBusText(unsigned domain, unsigned bus, unsigned device, unsigned function)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%x", domain & 0xffffu, bus & 0xffu,
                  device & 0x1fu, function & 0x7u);
    return buf;
}

std::string pciBusFromText(const std::string &text)
{
    // [domain:]bus:device[.function], in hex, as lspci and the APIs write it.
    std::vector<unsigned long> parts;
    std::string cur;
    char lastSep = 0, funcSep = 0;
    for (size_t i = 0; i <= text.size(); i++)
    {
        const char c = i < text.size() ? text[i] : '\0';
        if (hexDigit(c) >= 0)
        {
            cur += c;
            continue;
        }
        if (c == ':' || c == '.' || c == '\0')
        {
            if (cur.empty() || cur.size() > 8) return "";
            parts.push_back(std::strtoul(cur.c_str(), nullptr, 16));
            cur.clear();
            if (c == '.') funcSep = '.';
            lastSep = c;
            if (c == '\0') break;
            continue;
        }
        if (c == ' ' && cur.empty() && parts.empty()) continue;   // leading blanks
        return "";
    }
    (void)lastSep;
    unsigned long domain = 0, bus = 0, device = 0, function = 0;
    if (funcSep)
    {
        if (parts.size() == 4)      { domain = parts[0]; bus = parts[1]; device = parts[2]; function = parts[3]; }
        else if (parts.size() == 3) { bus = parts[0]; device = parts[1]; function = parts[2]; }
        else return "";
    }
    else
    {
        if (parts.size() == 3)      { domain = parts[0]; bus = parts[1]; device = parts[2]; }
        else if (parts.size() == 2) { bus = parts[0]; device = parts[1]; }
        else return "";
    }
    if (domain > 0xffff || bus > 0xff || device > 0x1f || function > 7) return "";
    return pciBusText(static_cast<unsigned>(domain), static_cast<unsigned>(bus),
                      static_cast<unsigned>(device), static_cast<unsigned>(function));
}

std::string registryText(std::uint64_t id)
{
    if (!id) return "";
    char buf[24];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(id));
    return buf;
}

std::string pciIdText(std::uint32_t id)
{
    if (!id) return "";
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%x", id);
    return buf;
}

// ── Tokens: HMAC-SHA256 ────────────────────────────────────────────────────
//
// A self-contained SHA-256 (FIPS 180-4): the tokens are the only thing that
// needs one, and a crypto library for one hash is not worth a dependency on
// every platform the tool builds for.

namespace {

struct Sha256
{
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    unsigned char block[64] = {};
    size_t        used = 0;
    std::uint64_t bits = 0;

    static std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void compress(const unsigned char *p)
    {
        static const std::uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        std::uint32_t w[64];
        for (int i = 0; i < 16; i++)
            w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) |
                   (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
        for (int i = 16; i < 64; i++)
        {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++)
        {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = hh + s1 + ch + k[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const unsigned char *p, size_t n)
    {
        bits += static_cast<std::uint64_t>(n) * 8;
        while (n)
        {
            const size_t take = std::min(n, sizeof(block) - used);
            std::memcpy(block + used, p, take);
            used += take;
            p += take;
            n -= take;
            if (used == sizeof(block))
            {
                compress(block);
                used = 0;
            }
        }
    }

    void finish(unsigned char out[32])
    {
        const std::uint64_t total = bits;
        const unsigned char one = 0x80, zero = 0;
        update(&one, 1);
        while (used != 56) update(&zero, 1);
        unsigned char len[8];
        for (int i = 0; i < 8; i++) len[i] = static_cast<unsigned char>(total >> (56 - 8 * i));
        update(len, 8);
        for (int i = 0; i < 8; i++)
        {
            out[4 * i]     = static_cast<unsigned char>(h[i] >> 24);
            out[4 * i + 1] = static_cast<unsigned char>(h[i] >> 16);
            out[4 * i + 2] = static_cast<unsigned char>(h[i] >> 8);
            out[4 * i + 3] = static_cast<unsigned char>(h[i]);
        }
    }
};

void hmacSha256(const std::vector<unsigned char> &key, const std::string &message,
                unsigned char out[32])
{
    unsigned char k[64] = {};
    if (key.size() > sizeof(k))
    {
        Sha256 kh;
        kh.update(key.data(), key.size());
        kh.finish(k);
    }
    else
    {
        std::memcpy(k, key.data(), key.size());
    }
    unsigned char ipad[64], opad[64];
    for (int i = 0; i < 64; i++)
    {
        ipad[i] = static_cast<unsigned char>(k[i] ^ 0x36);
        opad[i] = static_cast<unsigned char>(k[i] ^ 0x5c);
    }
    unsigned char inner[32];
    Sha256 in;
    in.update(ipad, sizeof(ipad));
    in.update(reinterpret_cast<const unsigned char *>(message.data()), message.size());
    in.finish(inner);
    Sha256 outer;
    outer.update(opad, sizeof(opad));
    outer.update(inner, sizeof(inner));
    outer.finish(out);
}

// The key's bytes; empty for a key that is not even-length hex.
std::vector<unsigned char> keyBytes(const std::string &hex)
{
    std::vector<unsigned char> out;
    if (hex.empty() || hex.size() % 2) return out;
    for (size_t i = 0; i < hex.size(); i += 2)
    {
        const int a = hexDigit(hex[i]), b = hexDigit(hex[i + 1]);
        if (a < 0 || b < 0) return {};
        out.push_back(static_cast<unsigned char>(a * 16 + b));
    }
    return out;
}

} // namespace

const std::string &processEvidenceKey()
{
    static const std::string key = newEvidenceKey();
    return key;
}

std::string evidenceToken(const std::string &keyHex, const std::string &kind,
                          const std::string &value)
{
    std::vector<unsigned char> key = keyBytes(keyHex);
    if (key.empty()) key = keyBytes(processEvidenceKey());
    std::string message = kind;
    message += '\0';
    message += value;
    unsigned char mac[32];
    hmacSha256(key, message, mac);
    return hexOf(mac, 8);
}

DeviceEvidence evidenceFrom(const logger::Identity &id, const std::string &keyHex)
{
    DeviceEvidence e;
    const std::pair<const char *, const std::string *> raw[] = {
        {"uuid", &id.uuid}, {"luid", &id.luid}, {"pci_bus", &id.pciBus}, {"registry", &id.registryId}};
    for (const auto &r : raw)
    {
        const std::string value = canonicalIdentifier(r.first, *r.second);
        if (!value.empty()) e.tokens[r.first] = evidenceToken(keyHex, r.first, value);
    }
    e.host = id.host;
    e.soc  = id.socNpu         ? toString(ProcessorKind::Npu)
           : id.socGpu         ? toString(ProcessorKind::Gpu)
           : id.socAccelerator ? toString(ProcessorKind::Accelerator)
                               : "";
    e.defaultGpu = id.defaultGpu;
    return e;
}

std::string currentEvidenceKey()
{
    const RunCheckpoint *run = RunCheckpoint::current();
    return run && !run->evidenceKey().empty() ? run->evidenceKey() : processEvidenceKey();
}

// ── Scrubbing a library's message ───────────────────────────────────────────

namespace {

bool isHex(char c) { return hexDigit(c) >= 0; }
bool isWordChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

size_t hexRun(const std::string &s, size_t at)
{
    size_t n = 0;
    while (at + n < s.size() && isHex(s[at + n])) n++;
    return n;
}

// "0000:01:00.0", the domain 4 to 8 digits (nvidia-smi writes 8).
bool matchPci(const std::string &s, size_t at, size_t &len)
{
    const size_t d = hexRun(s, at);
    if (d < 4 || d > 8) return false;
    size_t i = at + d;
    if (i >= s.size() || s[i] != ':' || hexRun(s, i + 1) != 2) return false;
    i += 3;
    if (i >= s.size() || s[i] != ':' || hexRun(s, i + 1) != 2) return false;
    i += 3;
    if (i + 1 >= s.size() || s[i] != '.' || s[i + 1] < '0' || s[i + 1] > '7') return false;
    i += 2;
    if (i < s.size() && isWordChar(s[i])) return false;
    len = i - at;
    return true;
}

// 8-4-4-4-12 hex digits, or 32 together.
bool matchUuid(const std::string &s, size_t at, size_t &len, std::string &hex)
{
    static const size_t groups[] = {8, 4, 4, 4, 12};
    size_t i = at;
    hex.clear();
    bool dashed = true;
    for (size_t g = 0; g < 5 && dashed; g++)
    {
        if (hexRun(s, i) < groups[g]) { dashed = false; break; }
        hex += s.substr(i, groups[g]);
        i += groups[g];
        if (g < 4)
        {
            if (i >= s.size() || s[i] != '-') { dashed = false; break; }
            i++;
        }
    }
    if (dashed && (i >= s.size() || !isWordChar(s[i])))
    {
        len = i - at;
        return true;
    }
    if (hexRun(s, at) == 32 && (at + 32 >= s.size() || !isWordChar(s[at + 32])))
    {
        hex = s.substr(at, 32);
        len = 32;
        return true;
    }
    return false;
}

// "LUID=1234", "luid: 0x4d2", "registryID 4294969843": the number after a
// name, which is the identifier.
bool matchNamedNumber(const std::string &s, size_t at, const char *name, size_t &numAt,
                      size_t &numLen, std::uint64_t &value)
{
    const size_t n = std::strlen(name);
    if (at + n > s.size()) return false;
    for (size_t k = 0; k < n; k++)
        if (std::tolower(static_cast<unsigned char>(s[at + k])) != name[k]) return false;
    size_t i = at + n;
    if (i < s.size() && isWordChar(s[i]) && !(s[i] == '_' || std::isdigit((unsigned char)s[i])))
        return false;
    while (i < s.size() && (s[i] == '"' || s[i] == '=' || s[i] == ':' || s[i] == ' ' || s[i] == '_'))
        i++;
    const bool hex = i + 1 < s.size() && s[i] == '0' && (s[i + 1] == 'x' || s[i + 1] == 'X');
    const size_t from = hex ? i + 2 : i;
    size_t j = from;
    while (j < s.size() && (hex ? isHex(s[j]) : std::isdigit(static_cast<unsigned char>(s[j])) != 0))
        j++;
    if (j == from || j - from > (hex ? 16u : 20u) || (j < s.size() && isWordChar(s[j])))
        return false;
    value = std::strtoull(s.substr(from, j - from).c_str(), nullptr, hex ? 16 : 10);
    if (!value) return false;
    numAt  = i;
    numLen = j - i;
    return true;
}

} // namespace

std::string scrubIdentifiers(const std::string &message)
{
    // Most lines hold nothing of the kind: leave them as they are, uncopied.
    bool maybe = false;
    for (char c : message)
        if (c == ':' || c == '-' || c == '=')
        {
            maybe = true;
            break;
        }
    if (!maybe && message.size() < 32) return message;

    const std::string key = currentEvidenceKey();
    auto token = [&](const char *kind, const std::string &canonical) {
        return std::string("<") + kind + " " + evidenceToken(key, kind, canonical) + ">";
    };
    std::string out;
    out.reserve(message.size());
    size_t i = 0;
    while (i < message.size())
    {
        const bool boundary = i == 0 || !isWordChar(message[i - 1]);
        if (boundary)
        {
            size_t len = 0, numAt = 0, numLen = 0;
            std::uint64_t value = 0;
            std::string hex;
            if (matchPci(message, i, len))
            {
                const std::string canonical = pciBusFromText(message.substr(i, len));
                out += canonical.empty() ? std::string("<pci_bus>") : token("pci_bus", canonical);
                i += len;
                continue;
            }
            if (matchUuid(message, i, len, hex))
            {
                for (char &c : hex) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                out += token("uuid", hex);
                i += len;
                continue;
            }
            const char *const named[][2] = {{"luid", "luid"}, {"registryid", "registry"},
                                            {"registry id", "registry"}};
            bool done = false;
            for (const auto &nm : named)
                if (matchNamedNumber(message, i, nm[0], numAt, numLen, value))
                {
                    out += message.substr(i, numAt - i);
                    out += token(nm[1], registryText(value));
                    i = numAt + numLen;
                    done = true;
                    break;
                }
            if (done) continue;
        }
        out += message[i++];
    }
    return out;
}

// ── The linker ─────────────────────────────────────────────────────────────

namespace {

// Strong evidence in the order a device's `link` names it when it shares
// more than one kind: a UUID first, as the most specific.
const char *const kStrong[]    = {"uuid", "luid", "pci_bus", "registry"};
// The kinds every API writes in one scheme, so two different values say two
// different devices.  A UUID only ever adds a link: its schemes differ from
// one vendor's API to another's, so two different UUIDs say nothing.
const char *const kUniversal[] = {"luid", "pci_bus", "registry"};

const char *kindWords(const std::string &kind)
{
    if (kind == "luid")    return "LUIDs";
    if (kind == "pci_bus") return "PCI bus addresses";
    if (kind == "registry") return "registry IDs";
    return "UUIDs";
}

std::string lower(std::string s)
{
    for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// The vendor a vendor string names by its first word, in the vocabulary's
// own words: "NVIDIA Corporation", "Intel(R) Corporation", "Apple", "QUALCOMM".
bool vendorOfWord(const std::string &text, Vendor &out)
{
    std::string w;
    for (char c : text)
    {
        if (std::isalpha(static_cast<unsigned char>(c)))
            w += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        else if (!w.empty())
            break;
    }
    return !w.empty() && fromString(w, out);
}

bool pciIdOf(const std::string &hex, std::uint32_t &out)
{
    if (hex.empty() || hex.size() > 8) return false;
    for (char c : hex)
        if (hexDigit(c) < 0) return false;
    out = static_cast<std::uint32_t>(std::strtoul(hex.c_str(), nullptr, 16));
    return true;
}

} // namespace

bool vendorOf(const HardwareFacts &f, Vendor &out)
{
    std::uint32_t id = 0;
    if (pciIdOf(f.vendorId, id) && vendorFromPciId(id, out)) return true;
    return vendorOfWord(f.vendorName, out);
}

std::string computeUnitName(const HardwareFacts &facts, DeviceType type)
{
    if (type == DeviceType::Cpu) return "thread";
    Vendor v;
    if (type != DeviceType::Gpu || !vendorOf(facts, v)) return "";
    switch (v)
    {
    case Vendor::Nvidia: return "SM";
    case Vendor::Amd:    return "CU";
    case Vendor::Intel:  return "EU";
    case Vendor::Apple:
    case Vendor::Arm:    return "core";
    default:             return "";
    }
}

namespace {

ProcessorKind kindOf(DeviceType t)
{
    switch (t)
    {
    case DeviceType::Cpu: return ProcessorKind::Cpu;
    case DeviceType::Gpu: return ProcessorKind::Gpu;
    case DeviceType::Npu: return ProcessorKind::Npu;
    default:              return ProcessorKind::Accelerator;
    }
}

int backendOrder(const std::string &name)
{
    Backend b;
    return backendFromName(name, b) ? static_cast<int>(b) : static_cast<int>(Backend::COUNT);
}

// As --list-devices prints it and --devices takes it: "vulkan:0".
std::string tokenOf(const LinkDevice &d)
{
    Backend b;
    const std::string flag = backendFromName(d.backend, b) ? std::string(backendInfo(b).flag)
                                                           : lower(d.backend);
    return d.deviceIndex >= 0 ? flag + ":" + std::to_string(d.deviceIndex) : flag;
}

struct Node
{
    const LinkDevice *d = nullptr;
    size_t        input = 0;     // index into the caller's devices
    ProcessorKind kind = ProcessorKind::Accelerator;
    bool          host = false;
    std::string   soc;           // "npu" | "gpu" | "accelerator" | ""
    bool          vendorKnown = false;
    Vendor        vendor = Vendor::COUNT;
    std::string   link;          // decided so far
};

// One set of devices: what the rules ask of it.
struct Group
{
    std::vector<size_t> members;           // node positions, in run order
    ProcessorKind kind = ProcessorKind::Accelerator;
    bool          host = false;
    bool          soc = false;             // holds a device on the SoC by evidence
    bool          socGpuOnly = true;       // every member is an SoC GPU delegate
    bool          vendorKnown = false;
    Vendor        vendor = Vendor::COUNT;
    std::string   deviceId;                // with a known vendor: a model id
    int           integrated = -1;
    std::map<std::string, std::set<std::string>> universal;
    std::map<std::string, int> backends;   // devices per backend
};

class Linker
{
public:
    Linker(const std::vector<LinkDevice> &devices, const HostCpu &host, const std::string &socName)
        : in(devices), hostCpu(host), soc(socName)
    {
        std::vector<size_t> order(devices.size());
        for (size_t i = 0; i < order.size(); i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            const int oa = backendOrder(devices[a].backend), ob = backendOrder(devices[b].backend);
            if (oa != ob) return oa < ob;
            return devices[a].deviceIndex < devices[b].deviceIndex;
        });
        for (size_t i : order)
        {
            Node n;
            n.d     = &devices[i];
            n.input = i;
            n.kind  = kindOf(devices[i].type);
            n.host  = devices[i].evidence.host || devices[i].type == DeviceType::Cpu;
            n.soc   = n.host ? std::string() : devices[i].evidence.soc;
            n.vendorKnown = vendorOf(devices[i].facts, n.vendor);
            nodes.push_back(n);
        }
        parent.resize(nodes.size());
        for (size_t i = 0; i < parent.size(); i++) parent[i] = i;
    }

    HardwareLink run();

private:
    const std::vector<LinkDevice> &in;
    HostCpu      hostCpu;
    std::string  soc;
    std::vector<Node>   nodes;      // run order
    std::vector<size_t> parent;
    std::vector<std::string> notes; // the --verbose lines for pairs left apart

    size_t find(size_t x)
    {
        while (parent[x] != x) x = parent[x] = parent[parent[x]];
        return x;
    }
    // The set's root is its first device in run order.
    void unite(size_t a, size_t b)
    {
        a = find(a);
        b = find(b);
        if (a == b) return;
        if (b < a) std::swap(a, b);
        parent[b] = a;
    }

    std::map<size_t, Group> groups();
    static bool distinct(const Group &a, const Group &b, std::string &why);
    void markUnlinked(const Group &g, const char *link);
    std::string vendorWord(const Group &g) const
    {
        return g.vendorKnown ? toString(g.vendor) : "unknown-vendor";
    }
    std::string tokens(const Group &g) const
    {
        std::string s;
        for (size_t m : g.members) s += (s.empty() ? "" : ", ") + tokenOf(*nodes[m].d);
        return s;
    }

    void ruleHost();
    void ruleStrong();
    void ruleSocNpu();
    void ruleSole();
    void ruleSocGpu();
    void ruleWildcard();
};

std::map<size_t, Group> Linker::groups()
{
    std::map<size_t, Group> out;
    for (size_t i = 0; i < nodes.size(); i++)
    {
        const Node &n = nodes[i];
        Group &g = out[find(i)];
        if (g.members.empty())
            g.kind = n.kind;
        else if (n.kind == ProcessorKind::Gpu || (n.kind == ProcessorKind::Npu && g.kind != ProcessorKind::Gpu))
            g.kind = n.kind;      // a typed view outranks an untyped one
        g.members.push_back(i);
        g.host = g.host || n.host;
        g.soc  = g.soc || !n.soc.empty();
        g.socGpuOnly = g.socGpuOnly && n.soc == "gpu";
        if (n.vendorKnown && !g.vendorKnown)
        {
            g.vendorKnown = true;
            g.vendor      = n.vendor;
        }
        if (g.integrated < 0) g.integrated = n.d->facts.integrated;
        if (g.deviceId.empty() && n.vendorKnown && !n.d->facts.deviceId.empty() &&
            !n.d->facts.vendorId.empty())
            g.deviceId = n.d->facts.vendorId + ":" + n.d->facts.deviceId;
        for (const char *k : kUniversal)
        {
            auto t = n.d->evidence.tokens.find(k);
            if (t != n.d->evidence.tokens.end()) g.universal[k].insert(t->second);
        }
        g.backends[n.d->backend]++;
    }
    return out;
}

bool Linker::distinct(const Group &a, const Group &b, std::string &why)
{
    if (a.vendorKnown && b.vendorKnown && a.vendor != b.vendor)
    {
        why = "their vendors differ";
        return true;
    }
    if (!a.deviceId.empty() && !b.deviceId.empty() && a.deviceId != b.deviceId)
    {
        why = "their PCI device ids differ";
        return true;
    }
    if (a.integrated >= 0 && b.integrated >= 0 && a.integrated != b.integrated)
    {
        why = "one is integrated and the other is not";
        return true;
    }
    for (const char *k : kUniversal)
    {
        auto ta = a.universal.find(k), tb = b.universal.find(k);
        if (ta == a.universal.end() || tb == b.universal.end()) continue;
        bool shared = false;
        for (const std::string &t : ta->second) shared = shared || tb->second.count(t);
        if (!shared)
        {
            why = std::string("their ") + kindWords(k) + " differ";
            return true;
        }
    }
    for (const auto &kv : a.backends)
        if (b.backends.count(kv.first))
        {
            why = kv.first + " lists both";
            return true;
        }
    return false;
}

void Linker::markUnlinked(const Group &g, const char *link)
{
    for (size_t m : g.members)
        if (nodes[m].link.empty()) nodes[m].link = link;
}

// Host: every device that executes on the CPU is the host CPU.
void Linker::ruleHost()
{
    size_t first = nodes.size();
    for (size_t i = 0; i < nodes.size(); i++)
        if (nodes[i].host)
        {
            nodes[i].link = toString(LinkEvidence::Host);
            if (first == nodes.size())
                first = i;
            else
                unite(first, i);
        }
}

// Strong: an equal token of one kind merges.  A device's `link` names the
// first kind it shares with another device of its set.
void Linker::ruleStrong()
{
    for (const char *k : kStrong)
    {
        std::map<std::string, size_t> seen;
        for (size_t i = 0; i < nodes.size(); i++)
        {
            if (nodes[i].host) continue;
            auto t = nodes[i].d->evidence.tokens.find(k);
            if (t == nodes[i].d->evidence.tokens.end()) continue;
            auto it = seen.find(t->second);
            if (it == seen.end())
                seen.emplace(t->second, i);
            else
                unite(it->second, i);
        }
    }
    for (size_t i = 0; i < nodes.size(); i++)
    {
        if (nodes[i].host) continue;
        for (const char *k : kStrong)
        {
            auto t = nodes[i].d->evidence.tokens.find(k);
            if (t == nodes[i].d->evidence.tokens.end()) continue;
            bool shared = false;
            for (size_t j = 0; j < nodes.size() && !shared; j++)
            {
                if (j == i || nodes[j].host || find(j) != find(i)) continue;
                auto u = nodes[j].d->evidence.tokens.find(k);
                shared = u != nodes[j].d->evidence.tokens.end() && u->second == t->second;
            }
            if (shared)
            {
                nodes[i].link = k;
                break;
            }
        }
    }
}

// SoC: an ML runtime's NPU is the SoC's NPU -- every one of them, unless
// their vendors say otherwise -- and a device on the SoC with no one unit
// (Android NNAPI) is the SoC's, as a processor of its own.
void Linker::ruleSocNpu()
{
    size_t first = nodes.size();
    for (size_t i = 0; i < nodes.size(); i++)
    {
        if (nodes[i].soc == "accelerator" && nodes[i].link.empty())
            nodes[i].link = toString(LinkEvidence::Soc);
        if (nodes[i].soc != "npu") continue;
        if (nodes[i].link.empty()) nodes[i].link = toString(LinkEvidence::Soc);
        if (first == nodes.size())
        {
            first = i;
            continue;
        }
        std::map<size_t, Group> gs = groups();
        std::string why;
        if (distinct(gs[find(first)], gs[find(i)], why) && why.find("lists both") == std::string::npos)
        {
            notes.push_back(tokenOf(*nodes[i].d) + " is an NPU on the SoC, and not " +
                            tokenOf(*nodes[first].d) + "'s: " + why);
            continue;
        }
        unite(first, i);
    }
}

// Sole: the views of one vendor's one GPU (NPU, accelerator) merge, when the
// run shows there is only one: no backend lists two, and nothing says two of
// them are different devices.
void Linker::ruleSole()
{
    const ProcessorKind kinds[] = {ProcessorKind::Gpu, ProcessorKind::Npu, ProcessorKind::Accelerator};
    for (ProcessorKind kind : kinds)
    {
        std::map<size_t, Group> gs = groups();
        std::map<Vendor, std::vector<size_t>> byVendor;   // roots
        for (const auto &kv : gs)
            if (!kv.second.host && kv.second.kind == kind && kv.second.vendorKnown &&
                !kv.second.socGpuOnly)
                byVendor[kv.second.vendor].push_back(kv.first);
        for (const auto &bv : byVendor)
        {
            const std::vector<size_t> &roots = bv.second;
            if (roots.size() < 2) continue;
            std::map<std::string, int> perBackend;
            for (size_t r : roots)
                for (const auto &b : gs[r].backends) perBackend[b.first] += b.second;
            std::string why;
            for (const auto &b : perBackend)
                if (b.second > 1)
                    why = b.first + " lists " + std::to_string(b.second) + " " +
                          toString(bv.first) + " " + toString(kind) + "s";
            for (size_t a = 0; a < roots.size() && why.empty(); a++)
                for (size_t b = a + 1; b < roots.size() && why.empty(); b++)
                {
                    std::string w;
                    if (distinct(gs[roots[a]], gs[roots[b]], w))
                        why = tokens(gs[roots[a]]) + " and " + tokens(gs[roots[b]]) + ": " + w;
                }
            if (!why.empty())
            {
                // Said for the sets nothing universal identifies: the ones a
                // better rule might have joined to another.
                for (size_t r : roots)
                    if (gs[r].universal.empty())
                        notes.push_back(tokens(gs[r]) + " not linked by vendor: " + why);
                continue;
            }
            for (size_t r : roots)
            {
                markUnlinked(gs[r], toString(LinkEvidence::Sole));
                unite(roots.front(), r);
            }
        }
    }
}

// SoC: an Android GPU delegate is the SoC's GPU -- the one integrated GPU
// another backend sees, or else a processor of its own on the SoC.
void Linker::ruleSocGpu()
{
    std::vector<size_t> seekers;
    for (size_t i = 0; i < nodes.size(); i++)
        if (nodes[i].soc == "gpu") seekers.push_back(i);
    if (seekers.empty()) return;
    for (size_t i : seekers)
        if (nodes[i].link.empty()) nodes[i].link = toString(LinkEvidence::Soc);

    std::map<size_t, Group> gs = groups();
    std::vector<size_t> candidates;
    for (const auto &kv : gs)
        if (!kv.second.host && kv.second.kind == ProcessorKind::Gpu && !kv.second.socGpuOnly &&
            kv.second.integrated == 1)
            candidates.push_back(kv.first);
    const size_t into = candidates.size() == 1 ? candidates.front() : seekers.front();
    if (candidates.size() == 1)
        markUnlinked(gs[into], toString(LinkEvidence::Soc));
    else if (candidates.size() > 1)
    {
        std::string all;
        for (size_t r : candidates) all += (all.empty() ? "" : "; ") + tokens(gs[r]);
        for (size_t i : seekers)
            notes.push_back(tokenOf(*nodes[i].d) + " is the SoC's GPU, which is one of " +
                            std::to_string(candidates.size()) + " integrated GPUs (" + all + ")");
    }
    for (size_t i : seekers) unite(into, i);
}

// Sole, for a GPU nothing identifies and whose vendor is unknown -- the
// platform's default GPU (LiteRT's WebGPU and Metal accelerators), a runtime
// that names no adapter: it joins the run's one GPU, when the run shows only
// one, and never one of several.
void Linker::ruleWildcard()
{
    std::map<size_t, Group> gs = groups();
    std::vector<size_t> wild, candidates;
    for (const auto &kv : gs)
    {
        const Group &g = kv.second;
        if (g.host || g.kind != ProcessorKind::Gpu) continue;
        const bool isWild = !g.vendorKnown && g.universal.empty() && !g.soc;
        (isWild ? wild : candidates).push_back(kv.first);
    }
    if (wild.empty()) return;

    std::map<std::string, int> gpusPerBackend;
    for (const auto &kv : gs)
        if (!kv.second.host && kv.second.kind == ProcessorKind::Gpu)
            for (const auto &b : kv.second.backends) gpusPerBackend[b.first] += b.second;
    std::string why;
    for (const auto &b : gpusPerBackend)
        if (b.second > 1) why = b.first + " lists " + std::to_string(b.second) + " GPUs";
    if (why.empty() && candidates.size() != 1)
        why = candidates.empty() ? "no other backend identifies a GPU"
                                 : std::to_string(candidates.size()) + " GPUs are in this run";
    for (size_t w : wild)
    {
        std::string reason = why;
        if (reason.empty() && distinct(gs[w], gs[candidates.front()], reason))
            reason = "it and " + tokens(gs[candidates.front()]) + " differ: " + reason;
        if (!reason.empty())
        {
            notes.push_back(tokens(gs[w]) + " not linked: its GPU is not named (" + reason + ")");
            continue;
        }
        markUnlinked(gs[w], toString(LinkEvidence::Sole));
        markUnlinked(gs[candidates.front()], toString(LinkEvidence::Sole));
        unite(candidates.front(), w);
    }
}

int kindRank(ProcessorKind k)
{
    switch (k)
    {
    case ProcessorKind::Cpu: return 0;
    case ProcessorKind::Gpu: return 1;
    case ProcessorKind::Npu: return 2;
    default:                 return 3;
    }
}

// Facts merged from the views of one processor, most native first (run
// order is the registry's): each field from the first view that gives it,
// the vendor and device ids together from the first view whose vendor id is
// a known one, and a count together with what it counts.
HardwareFacts mergedFacts(const std::vector<const LinkDevice *> &views)
{
    HardwareFacts m;
    const LinkDevice *ids = nullptr;
    for (int pass = 0; pass < 3 && !ids; pass++)
        for (const LinkDevice *d : views)
        {
            std::uint32_t id = 0;
            Vendor v;
            const bool known = pciIdOf(d->facts.vendorId, id) && vendorFromPciId(id, v);
            if ((pass == 0 && known && !d->facts.deviceId.empty()) || (pass == 1 && known) ||
                (pass == 2 && !d->facts.vendorId.empty()))
            {
                ids = d;
                break;
            }
        }
    if (ids)
    {
        m.vendorId = ids->facts.vendorId;
        m.deviceId = ids->facts.deviceId;
    }
    for (const LinkDevice *d : views)
    {
        const HardwareFacts &f = d->facts;
        if (m.vendorName.empty()) m.vendorName = f.vendorName;
        if (m.arch.empty()) m.arch = f.arch;
        if (!m.computeUnits && f.computeUnits)
        {
            m.computeUnits = f.computeUnits;
            m.computeUnit  = f.computeUnit;
        }
        if (!m.clockMhz) m.clockMhz = f.clockMhz;
        if (!m.memoryBytes) m.memoryBytes = f.memoryBytes;
        if (m.integrated < 0) m.integrated = f.integrated;
    }
    return m;
}

// A CPU's vendor: CPUID's vendor string, an ARM MIDR's implementer, else the
// first word of what it is called.
bool cpuVendor(const HardwareFacts &f, const std::string &name, Vendor &out)
{
    if (!f.cpu.cpuidVendor.empty() && vendorFromCpuid(f.cpu.cpuidVendor, out)) return true;
    for (const CpuCluster &c : f.cpu.clusters)
    {
        std::string hex = c.midr;
        if (hex.size() > 2 && (hex.compare(0, 2, "0x") == 0 || hex.compare(0, 2, "0X") == 0))
            hex.erase(0, 2);
        std::uint32_t midr = 0;
        if (pciIdOf(hex, midr) && vendorFromMidrImplementer((midr >> 24) & 0xffu, out)) return true;
    }
    return vendorOfWord(f.vendorName, out) || vendorOfWord(name, out);
}

HardwareLink Linker::run()
{
    ruleHost();
    ruleStrong();
    ruleSocNpu();
    ruleSole();
    ruleSocGpu();
    ruleWildcard();
    for (Node &n : nodes)
        if (n.link.empty()) n.link = toString(LinkEvidence::Own);

    // One processor per set, the host CPU's whether or not a device is in it.
    struct Proc
    {
        Group group;
        bool  inHost = false;
        size_t first = 0;      // run position of its first device
        ProcessorInfo info;
    };
    std::vector<Proc> procs;
    bool haveHost = false;
    for (auto &kv : groups())
    {
        Proc p;
        p.group = kv.second;
        p.first = kv.first + 1;          // the host CPU without devices sorts first
        if (p.group.host)
        {
            p.group.kind = ProcessorKind::Cpu;
            haveHost = true;
        }
        p.inHost = p.group.host || p.group.soc || p.group.integrated == 1;
        procs.push_back(std::move(p));
    }
    if (!haveHost)
    {
        Proc p;
        p.group.kind = ProcessorKind::Cpu;
        p.group.host = true;
        p.inHost = true;
        p.first = 0;
        procs.push_back(std::move(p));
    }
    std::stable_sort(procs.begin(), procs.end(), [](const Proc &a, const Proc &b) {
        if (a.inHost != b.inHost) return a.inHost;
        if (kindRank(a.group.kind) != kindRank(b.group.kind))
            return kindRank(a.group.kind) < kindRank(b.group.kind);
        return a.first < b.first;
    });

    HardwareLink out;
    out.processorOf.assign(in.size(), std::string());
    out.linkOf.assign(in.size(), std::string());

    PackageInfo hostPkg;
    hostPkg.id   = "pkg0";
    hostPkg.kind = toString(PackageKind::Cpu);
    out.packages.push_back(hostPkg);

    std::map<ProcessorKind, unsigned> counts;
    for (Proc &p : procs)
    {
        ProcessorInfo &info = p.info;
        info.kind = toString(p.group.kind);
        info.id   = processorIdFor(p.group.kind, counts[p.group.kind]++);
        std::vector<const LinkDevice *> views;
        for (size_t m : p.group.members) views.push_back(nodes[m].d);

        if (p.group.host)
        {
            // The CPU backend's facts, or the host's own start: never another
            // backend's, whose CPU driver reports its own vendor (Intel's
            // OpenCL runtime on an AMD CPU) and its own idea of a count.
            const LinkDevice *cpu = nullptr;
            for (const LinkDevice *d : views)
                if (backendOrder(d->backend) == static_cast<int>(Backend::Cpu))
                {
                    cpu = d;
                    break;
                }
            if (cpu)
            {
                info.facts = cpu->facts;
                info.name  = cpu->name;
            }
            else
            {
                info.name            = hostCpu.name.empty() ? soc : hostCpu.name;
                info.facts.cpu.threads = hostCpu.threads;
                info.facts.cpu.isa     = hostCpu.isa;
            }
            Vendor v;
            if (cpuVendor(info.facts, info.name, v)) info.vendor = toString(v);
            info.vendorName = !info.facts.vendorName.empty() ? info.facts.vendorName
                                                             : info.facts.cpu.cpuidVendor;
        }
        else
        {
            info.facts = mergedFacts(views);
            info.name  = views.empty() ? std::string() : views.front()->name;
            Vendor v;
            if (vendorOf(info.facts, v)) info.vendor = toString(v);
            info.vendorName = info.facts.vendorName;
        }
        info.facts.vendorName.clear();   // the processor's own `vendor_name`

        if (p.inHost)
        {
            info.package = out.packages.front().id;
            if (!p.group.host) out.packages.front().kind = toString(PackageKind::Soc);
        }
        else
        {
            PackageInfo pkg;
            pkg.id     = "pkg" + std::to_string(out.packages.size());
            pkg.kind   = p.group.integrated == 0 ? toString(PackageKind::Discrete) : "";
            pkg.vendor = info.vendor;
            pkg.name   = info.name;
            info.package = pkg.id;
            out.packages.push_back(pkg);
        }
        for (size_t m : p.group.members)
        {
            out.processorOf[nodes[m].input] = info.id;
            out.linkOf[nodes[m].input]      = nodes[m].link;
        }
        out.processors.push_back(info);
    }

    // The host package is named for the SoC where the system names one, and
    // otherwise for its CPU.
    PackageInfo &host = out.packages.front();
    for (const ProcessorInfo &p : out.processors)
        if (p.package == host.id && p.kind == toString(ProcessorKind::Cpu))
        {
            Vendor v;
            host.name   = soc.empty() ? p.name : soc;
            host.vendor = vendorOfWord(soc, v) ? toString(v) : p.vendor;
            break;
        }

    for (size_t i = 0; i < nodes.size(); i++)
    {
        const Node &n = nodes[i];
        const std::string &id = out.processorOf[n.input];
        XPUPEAK_VLOG("hardware: %s (%s) -> %s by %s\n", tokenOf(*n.d).c_str(), n.d->name.c_str(),
                     id.c_str(), n.link.c_str());
    }
    for (const std::string &note : notes)
        XPUPEAK_VLOG("hardware: %s\n", note.c_str());
    return out;
}

} // namespace

const ProcessorInfo *HardwareLink::processor(const std::string &id) const
{
    for (const ProcessorInfo &p : processors)
        if (p.id == id) return &p;
    return nullptr;
}

std::vector<size_t> HardwareLink::devicesOf(const std::string &processorId) const
{
    std::vector<size_t> out;
    for (size_t i = 0; i < processorOf.size(); i++)
        if (processorOf[i] == processorId) out.push_back(i);
    return out;
}

HardwareLink linkDevices(const std::vector<LinkDevice> &devices, const HostCpu &host,
                         const std::string &socName)
{
    return Linker(devices, host, socName).run();
}

void linkHardware(RunDocument &doc, LinkHost how)
{
    std::vector<LinkDevice> devices;
    devices.reserve(doc.devices.size());
    for (const DeviceResult &d : doc.devices)
        devices.push_back({d.backend, d.deviceIndex, d.name, d.type, d.evidence, d.facts});

    HostCpu host;
    if (how == LinkHost::Probe)
    {
        const HostInfo h = probeHost();
        host.name    = h.cpu;
        host.threads = h.logicalCores;
        host.isa     = h.arch;
    }
    else
    {
        // The file's own host CPU, as it was linked when it was written.
        for (const ProcessorInfo &p : doc.processors)
            if (p.kind == toString(ProcessorKind::Cpu) && p.package == "pkg0")
            {
                host.name    = p.name;
                host.threads = p.facts.cpu.threads;
                host.isa     = p.facts.cpu.isa;
                break;
            }
    }
    if (host.name.empty()) host.name = doc.meta.system.soc;
    if (host.isa.empty()) host.isa = doc.meta.system.arch;

    HardwareLink link = linkDevices(devices, host, doc.meta.system.soc);
    doc.packages   = std::move(link.packages);
    doc.processors = std::move(link.processors);
    for (size_t i = 0; i < doc.devices.size(); i++)
    {
        doc.devices[i].processor = link.processorOf[i];
        doc.devices[i].link      = link.linkOf[i];
    }
}

// ── Reading a linked document ──────────────────────────────────────────────

std::string processorIdFor(ProcessorKind kind, unsigned ordinal)
{
    return std::string(toString(kind)) + std::to_string(ordinal);
}

bool parseProcessorId(const std::string &id, ProcessorKind &kind, unsigned &ordinal)
{
    size_t digits = id.size();
    while (digits > 0 && std::isdigit(static_cast<unsigned char>(id[digits - 1]))) digits--;
    if (digits == 0 || digits == id.size()) return false;
    const std::string number = id.substr(digits);
    if (number.size() > 1 && number[0] == '0') return false;
    if (number.size() > 9) return false;
    if (!fromString(id.substr(0, digits), kind)) return false;
    ordinal = static_cast<unsigned>(std::stoul(number));
    return true;
}

const ProcessorInfo *processorOf(const RunDocument &doc, const DeviceResult &device)
{
    if (device.processor.empty()) return nullptr;
    for (const ProcessorInfo &p : doc.processors)
        if (p.id == device.processor) return &p;
    return nullptr;
}

std::vector<const DeviceResult *> devicesOf(const RunDocument &doc, const std::string &processorId)
{
    std::vector<const DeviceResult *> out;
    for (const DeviceResult &d : doc.devices)
        if (d.processor == processorId) out.push_back(&d);
    return out;
}

const PackageInfo *packageOf(const RunDocument &doc, const ProcessorInfo &processor)
{
    for (const PackageInfo &p : doc.packages)
        if (p.id == processor.package) return &p;
    return nullptr;
}

std::vector<const ProcessorInfo *> processorsIn(const RunDocument &doc, const std::string &packageId)
{
    std::vector<const ProcessorInfo *> out;
    for (const ProcessorInfo &p : doc.processors)
        if (p.package == packageId) out.push_back(&p);
    return out;
}

} // namespace xpupeak
