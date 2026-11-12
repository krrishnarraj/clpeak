#include <common/run_document.h>
#include <common/common.h>
#include <common/json.h>
#include <common/json_writer.h>
#include <common/options.h>
#include <version.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <locale>
#include <mutex>
#include <random>
#include <set>
#include <sstream>

// ── Enum <-> string ────────────────────────────────────────────────────────

const char *categoryString(Category c)
{
    switch (c)
    {
    case Category::Compute:  return "compute";
    case Category::Crypto:     return "crypto";
    case Category::String:     return "string";
    case Category::Bandwidth:  return "bandwidth";
    case Category::Latency:    return "latency";
    case Category::Ai:         return "ai";
    // Spelled out rather than left empty: the GUI's own sentinel has always
    // been "unknown", and an empty tag meant the two vocabularies disagreed
    // on the one value that most needed to round-trip.
    case Category::Unknown:    return "unknown";
    }
    return "unknown";
}

Category categoryFromString(const std::string &s)
{
    if (s == "compute")     return Category::Compute;
    if (s == "crypto")      return Category::Crypto;
    if (s == "string")      return Category::String;
    if (s == "bandwidth")   return Category::Bandwidth;
    if (s == "latency")     return Category::Latency;
    if (s == "ai")          return Category::Ai;
    return Category::Unknown;
}

const char *statusString(ResultStatus s)
{
    switch (s)
    {
    case ResultStatus::Ok:          return "ok";
    case ResultStatus::Unsupported: return "unsupported";
    case ResultStatus::Skipped:     return "skipped";
    case ResultStatus::Error:       return "error";
    }
    return "ok";
}

ResultStatus statusFromString(const std::string &s)
{
    if (s == "unsupported") return ResultStatus::Unsupported;
    if (s == "skipped")     return ResultStatus::Skipped;
    if (s == "error")       return ResultStatus::Error;
    return ResultStatus::Ok;
}

const char *shapeString(TestShape s)
{
    return (s == TestShape::Homogeneous) ? "homogeneous" : "heterogeneous";
}

TestShape shapeFromString(const std::string &s)
{
    return (s == "homogeneous") ? TestShape::Homogeneous
                                : TestShape::Heterogeneous;
}

const char *deviceTypeString(DeviceType t)
{
    switch (t)
    {
    case DeviceType::Cpu:         return "cpu";
    case DeviceType::Gpu:         return "gpu";
    case DeviceType::Accelerator: return "accelerator";
    case DeviceType::Npu:         return "npu";
    case DeviceType::Unknown:     break;
    }
    return "unknown";
}

DeviceType deviceTypeFromString(const std::string &s)
{
    if (s == "cpu")         return DeviceType::Cpu;
    if (s == "gpu")         return DeviceType::Gpu;
    if (s == "accelerator") return DeviceType::Accelerator;
    if (s == "npu")         return DeviceType::Npu;
    return DeviceType::Unknown;
}

// ── Dims ───────────────────────────────────────────────────────────────────

DimValue DimValue::ofText(std::string s)
{
    DimValue v;
    v.text = std::move(s);
    return v;
}

DimValue DimValue::ofNumber(double n)
{
    DimValue v;
    v.isNumber = true;
    v.number   = n;
    return v;
}

namespace {

bool isWholeNumber(double n)
{
    return std::isfinite(n) && std::floor(n) == n && std::fabs(n) < 9.007199254740992e15;
}

std::string numberText(double n)
{
    if (isWholeNumber(n)) return std::to_string(static_cast<long long>(n));
    std::ostringstream ss;
    ss.imbue(std::locale::classic());
    ss.precision(17);
    ss << n;
    return ss.str();
}

// A producer's value of the wrong kind for its key: a data type for
// `level`, a word of another key's list.  Debug builds stop there, since
// every such pairing is a slip at a call site that no reader could repair.
void checkDim(bool ok, const char *what)
{
    (void)what;
    assert(ok && what);
}

} // namespace

std::string DimValue::str() const
{
    return isNumber ? numberText(number) : text;
}

bool DimValue::operator==(const DimValue &o) const
{
    if (isNumber != o.isNumber) return false;
    return isNumber ? number == o.number : text == o.text;
}

DimEntry::DimEntry(xpupeak::Dim k, xpupeak::DType v)
    : key(xpupeak::toString(k)), value(DimValue::ofText(xpupeak::toString(v)))
{
    checkDim(xpupeak::dimInfo(k).kind == xpupeak::DimKind::DataType,
             "a data type for a key that takes none");
}

DimEntry::DimEntry(xpupeak::Dim k, const char *text)
    : key(xpupeak::toString(k)), value(DimValue::ofText(text ? text : ""))
{
    // Text keys take any string; a Word key one of its words, and a data
    // type key a data type's name, for code that walks a table of strings.
    switch (xpupeak::dimInfo(k).kind)
    {
    case xpupeak::DimKind::Text:
        break;
    case xpupeak::DimKind::Word:
    {
        bool known = false;
        for (unsigned i = 0; i < xpupeak::dimWordCount(k); i++)
            if (value.text == xpupeak::dimWord(k, i)) known = true;
        checkDim(known, "a word that is not in the key's list");
        break;
    }
    case xpupeak::DimKind::DataType:
    {
        xpupeak::DType t;
        checkDim(xpupeak::fromString(value.text, t), "a data type the vocabulary does not have");
        break;
    }
    case xpupeak::DimKind::Integer:
        checkDim(false, "text for a key that takes a number");
        break;
    }
}

DimEntry::DimEntry(xpupeak::Dim k, const std::string &text) : DimEntry(k, text.c_str()) {}

DimEntry DimEntry::integer(xpupeak::Dim k, double v)
{
    checkDim(xpupeak::dimInfo(k).kind == xpupeak::DimKind::Integer,
             "a number for a key that takes none");
    return DimEntry(xpupeak::toString(k), DimValue::ofNumber(v));
}

DimEntry DimEntry::word(xpupeak::Dim k, xpupeak::Dim owner, unsigned index)
{
    checkDim(k == owner, "a word of another key's list");
    const char *w = xpupeak::dimWord(owner, index);
    return DimEntry(xpupeak::toString(k), DimValue::ofText(w ? w : ""));
}

Dims::Dims(std::initializer_list<DimEntry> list)
{
    for (const DimEntry &e : list) set(e);
}

void Dims::set(DimEntry e)
{
    for (DimEntry &have : entries)
        if (have.key == e.key)
        {
            have.value = std::move(e.value);
            return;
        }
    entries.push_back(std::move(e));
}

const DimValue *Dims::find(const std::string &key) const
{
    for (const DimEntry &e : entries)
        if (e.key == key) return &e.value;
    return nullptr;
}

bool Dims::operator==(const Dims &o) const
{
    if (entries.size() != o.entries.size()) return false;
    for (const DimEntry &e : entries)
    {
        const DimValue *v = o.find(e.key);
        if (!v || *v != e.value) return false;
    }
    return true;
}

Dims effectiveDims(const Dims &test, const Dims &reading)
{
    Dims out = test;
    for (const DimEntry &e : reading) out.set(e);
    return out;
}

// ── Hardware facts ─────────────────────────────────────────────────────────

bool CpuFacts::empty() const
{
    return isa.empty() && !cores && !threads && !sockets && clusters.empty() &&
           caches.empty() && cpuidVendor.empty() && cpuidFamily < 0 && cpuidModel < 0 &&
           cpuidStepping < 0 && engines.empty() && features.empty();
}

bool HardwareFacts::empty() const
{
    return vendorId.empty() && deviceId.empty() && arch.empty() && !computeUnits &&
           computeUnit.empty() && !clockMhz && !memoryBytes && integrated < 0 && cpu.empty();
}

BuildBackend buildBackendFor(Backend b)
{
    BuildBackend out;
    out.name     = backendInfo(b).name;
    out.kind     = xpupeak::toString(xpupeak::backendKind(b));
    out.portable = xpupeak::backendPortable(b);
    return out;
}

std::string extFromJson(const std::string &json)
{
    if (json.empty()) return "";
    const JsonValue v = jsonParse(json);
    return v.isObject() ? jsonSerialize(v) : std::string();
}

// ── Lookup / merge ─────────────────────────────────────────────────────────

TestResult *DeviceResult::findTest(const std::string &testKey)
{
    for (TestResult &t : tests)
        if (t.key() == testKey) return &t;
    return nullptr;
}

DeviceResult *RunDocument::findDevice(const std::string &deviceKey)
{
    for (DeviceResult &d : devices)
        if (d.key() == deviceKey) return &d;
    return nullptr;
}

void RunDocument::append(const RunDocument &other)
{
    for (const DeviceResult &d : other.devices)
    {
        DeviceResult *existing = findDevice(d.key());
        if (!existing)
        {
            devices.push_back(d);
            continue;
        }
        // Same device seen twice (two loggers, or a backend that enumerates
        // it more than once): fold the tests in rather than emitting a second
        // device block the GUI would show as a separate run.  A reading this
        // document already has stays: a resumed run measures again a test the
        // part before was held in, and that part's readings are the ones kept
        // (RunCheckpoint::held).
        for (const TestResult &t : d.tests)
        {
            TestResult *known = existing->findTest(t.key());
            if (!known) { existing->tests.push_back(t); continue; }
            for (const MetricResult &m : t.metrics)
            {
                const bool had = std::any_of(known->metrics.begin(), known->metrics.end(),
                                             [&](const MetricResult &k) { return k.id == m.id; });
                if (!had) known->metrics.push_back(m);
            }
        }
        if (existing->properties.empty()) existing->properties = d.properties;
        if (existing->facts.empty())      existing->facts      = d.facts;
        if (existing->evidence.empty())   existing->evidence   = d.evidence;
        if (existing->runtime.empty())    existing->runtime    = d.runtime;
        if (existing->ext.empty())        existing->ext        = d.ext;
    }
}

RunProgress::Device &RunProgress::device(const std::string &backend, int deviceIndex)
{
    for (Device &d : devices)
        if (d.backend == backend && d.deviceIndex == deviceIndex) return d;
    devices.push_back(Device{backend, deviceIndex, false, {}, {}});
    return devices.back();
}

bool RunProgress::backendDone(const std::string &backend) const
{
    for (const std::string &b : backendsDone)
        if (b == backend) return true;
    return false;
}

// ── Baselines ──────────────────────────────────────────────────────────────

std::string deviceKey(const std::string &name, int deviceIndex)
{
    if (deviceIndex < 0) return name;
    return name + "#" + std::to_string(deviceIndex);
}

std::string baselineKey(const std::string &backend, const std::string &platform,
                        const std::string &device, const std::string &testKey,
                        const std::string &metricId)
{
    return backend + "/" + platform + "/" + device + "/" + testKey + "/" +
           metricId;
}

BaselineMap buildBaselineMap(const RunDocument &doc)
{
    BaselineMap m;
    for (const DeviceResult &d : doc.devices)
        for (const TestResult &t : d.tests)
            for (const MetricResult &r : t.metrics)
                if (r.status == ResultStatus::Ok)
                    m[baselineKey(d.backend, d.platform,
                                  deviceKey(d.name, d.deviceIndex),
                                  t.key(), r.id)] = r.value;
    return m;
}

// ── Time ───────────────────────────────────────────────────────────────────

std::string isoTimestampUtc()
{
    const std::time_t now = std::time(nullptr);
    std::tm           tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}

namespace {

// Bytes for an id or a key.  random_device is mixed with the clock through a
// generator, so a runtime whose random_device is deterministic (some MinGW
// builds) still gives each run its own.  Nothing here is a secret that has to
// withstand an attacker: the run id identifies a run, and the evidence key
// only keeps one file's tokens from matching another's.
std::vector<unsigned char> randomBytes(size_t n)
{
    std::random_device rd;
    const auto now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq seed{rd(), rd(), rd(), rd(), static_cast<unsigned>(now),
                       static_cast<unsigned>(static_cast<unsigned long long>(now) >> 32)};
    std::mt19937_64 gen(seed);
    std::vector<unsigned char> out(n);
    for (unsigned char &b : out)
        b = static_cast<unsigned char>((gen() >> 56) ^ (rd() & 0xffu));
    return out;
}

std::string hexOf(const std::vector<unsigned char> &bytes, size_t from, size_t count)
{
    static const char digits[] = "0123456789abcdef";
    std::string out;
    for (size_t i = from; i < from + count && i < bytes.size(); i++)
    {
        out += digits[bytes[i] >> 4];
        out += digits[bytes[i] & 0xf];
    }
    return out;
}

} // namespace

std::string newRunId()
{
    std::vector<unsigned char> b = randomBytes(16);
    b[6] = static_cast<unsigned char>((b[6] & 0x0f) | 0x40);   // version 4
    b[8] = static_cast<unsigned char>((b[8] & 0x3f) | 0x80);   // RFC 4122 variant
    return hexOf(b, 0, 4) + "-" + hexOf(b, 4, 2) + "-" + hexOf(b, 6, 2) + "-" +
           hexOf(b, 8, 2) + "-" + hexOf(b, 10, 6);
}

std::string newEvidenceKey()
{
    const std::vector<unsigned char> b = randomBytes(32);
    return hexOf(b, 0, b.size());
}

bool parseIsoTimestampUtc(const std::string &iso, long long &secondsSinceEpoch)
{
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, s = 0;
    char z = 0;
    if (std::sscanf(iso.c_str(), "%4d-%2d-%2dT%2d:%2d:%2d%c", &y, &mo, &d, &h, &mi, &s, &z) != 7 ||
        z != 'Z' || mo < 1 || mo > 12 || d < 1 || d > 31 || h > 23 || mi > 59 || s > 60)
        return false;
    // Days since 1970-01-01 in the proleptic Gregorian calendar, without
    // timegm (not on Windows) or mktime (local time).
    const int       yy  = y - (mo <= 2 ? 1 : 0);
    const long long era = (yy >= 0 ? yy : yy - 399) / 400;
    const long long yoe = yy - era * 400;
    const long long doy = (153 * (mo + (mo > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const long long doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    const long long days = era * 146097 + doe - 719468;
    secondsSinceEpoch = days * 86400 + h * 3600 + mi * 60 + s;
    return true;
}

// ── JSON writer ────────────────────────────────────────────────────────────

namespace {

void writeExt(JsonWriter &w, const std::string &ext)
{
    if (!ext.empty()) w.raw("ext", ext);
}

// The machine.  Every field is optional; what the platform cannot answer is
// absent rather than empty.
void writeSystem(JsonWriter &w, const SystemInfo &s)
{
    w.beginObject("system");
    w.strIf("form_factor",        s.formFactor);
    w.strIf("form_factor_detail", s.formFactorDetail);
    w.strIf("form_factor_from",   s.formFactorFrom);
    if (s.isVirtual)  w.boolean("virtual", true);
    if (s.translated) w.boolean("translated", true);
    w.strIf("manufacturer", s.manufacturer);
    w.strIf("brand",        s.brand);
    w.strIf("model",        s.model);
    w.strIf("codename",     s.codename);
    w.strIf("name",         s.name);
    w.strIf("family",       s.family);
    w.strIf("board",        s.board);
    w.strIf("soc",          s.soc);
    const OsInfo &os = s.os;
    if (!os.name.empty() || !os.version.empty() || !os.build.empty() || !os.kernel.empty() ||
        !os.distribution.empty())
    {
        w.beginObject("os");
        w.strIf("name",         os.name);
        w.strIf("version",      os.version);
        w.strIf("build",        os.build);
        w.strIf("kernel",       os.kernel);
        w.strIf("distribution", os.distribution);
        w.endObject();
    }
    w.strIf("arch", s.arch);
    if (s.memoryBytes) w.uint("memory_bytes", s.memoryBytes);
    writeExt(w, s.ext);
    w.endObject();
}

void writeConditions(JsonWriter &w, const RunConditions &c)
{
    if (c.empty()) return;
    w.beginObject("conditions");
    w.strIf("power_source", c.powerSource);
    if (c.lowPowerMode >= 0) w.boolean("low_power_mode", c.lowPowerMode != 0);
    w.strIf("power_profile", c.powerProfile);
    w.strIf("cpu_governor",  c.cpuGovernor);
    w.strIf("thermal_start", c.thermalStart);
    w.strIf("thermal_end",   c.thermalEnd);
    writeExt(w, c.ext);
    w.endObject();
}

void writeAnnotations(JsonWriter &w, const std::map<std::string, std::string> &a)
{
    if (a.empty()) return;
    w.beginObject("annotations");
    for (const auto &kv : a) w.str(kv.first.c_str(), kv.second);
    w.endObject();
}

void writeInvocation(JsonWriter &w, const Invocation &inv)
{
    w.beginObject("invocation");
    if (!inv.argv.empty())
    {
        w.beginArray("argv");
        for (const std::string &a : inv.argv) w.rawString(a);
        w.endArray();
    }
    if (inv.targetTimeUs)      w.uint("target_time_us",        inv.targetTimeUs);
    if (inv.targetTimeUsCpu)   w.uint("target_time_us_cpu",    inv.targetTimeUsCpu);
    if (inv.targetTimeUsMlGpu) w.uint("target_time_us_ml_gpu", inv.targetTimeUsMlGpu);
    if (inv.warmup)            w.uint("warmup",                inv.warmup);
    // Absent means "calibrated per test", which is the normal mode; a number
    // means the run was pinned to it with -i and is not comparable to one that
    // was not.
    if (inv.iters)             w.uint("iters",                 inv.iters);
    if (!inv.categories.empty())
    {
        w.beginArray("categories");
        for (const std::string &c : inv.categories) w.rawString(c);
        w.endArray();
    }
    if (!inv.tests.empty())
    {
        w.beginArray("tests");
        for (const std::string &t : inv.tests) w.rawString(t);
        w.endArray();
    }
    if (inv.verbose) w.boolean("verbose", true);
    if (inv.standard) w.boolean("standard", true);
    w.endObject();
}

void writeBuild(JsonWriter &w, const BuildInfo &b)
{
    if (b.backends.empty()) return;
    w.beginObject("build");
    w.beginArray("backends");
    for (const BuildBackend &be : b.backends)
    {
        w.beginObjectInline();
        w.str("name", be.name);
        w.strIf("kind", be.kind);
        if (be.portable) w.boolean("portable", true);
        w.endObject();
    }
    w.endArray();
    w.strIf("config", b.config);
    w.endObject();
}

// One entry per line: the log reads as a transcript and greps like one,
// while the rest of the document keeps its one-key-per-line layout.
void writeLogEntry(JsonWriter &w, const LogEntry &e)
{
    w.beginObjectInline();
    w.num("elapsed_s", e.elapsedS);
    w.str("level", xpupeak::logLevelString(e.level));
    w.strIf("source",  e.source);
    w.strIf("backend", e.backend);
    w.strIf("device",  e.device);
    if (e.deviceIndex >= 0) w.integer("device_index", e.deviceIndex);
    w.strIf("test", e.test);
    w.str("message", e.message);
    w.endObject();
}

// A dims object's entries.  A number is a whole number by the format's rule,
// and written as one.
void writeDimEntries(JsonWriter &w, const Dims &dims)
{
    for (const DimEntry &e : dims)
    {
        if (!e.value.isNumber)
            w.str(e.key.c_str(), e.value.text);
        else if (isWholeNumber(e.value.number))
            w.integer(e.key.c_str(), static_cast<long long>(e.value.number));
        else
            w.raw(e.key.c_str(), numberText(e.value.number));
    }
}

// Dims on one line: a reading's coordinates read as one record.
void writeDimsKey(JsonWriter &w, const Dims &dims)
{
    if (dims.empty()) return;
    w.beginObjectInline("dims");
    writeDimEntries(w, dims);
    w.endObject();
}

void writeMetric(JsonWriter &w, const MetricResult &m)
{
    w.beginObject();
    w.str("id", m.id);
    w.strIf("label", m.label);
    w.strIf("benchmark", m.benchmark);
    writeDimsKey(w, m.dims);
    if (m.status == ResultStatus::Ok)
    {
        // An omitted "status" means ok.  Every reading in a healthy file is
        // ok, so spelling it out on each would be noise.
        w.num("value", m.value);
    }
    else
    {
        w.str("status", statusString(m.status));
        w.strIf("reason", m.reason);
    }
    if (m.hasUnit)
    {
        w.str("unit", m.unit);
        w.str("quantity", quantityString(m.quantity));
    }
    if (m.direction != Direction::FromUnit)
        w.str("direction", directionString(m.direction));
    w.strIf("description", m.description);
    writeExt(w, m.ext);
    w.endObject();
}

void writeTest(JsonWriter &w, const TestResult &t)
{
    w.beginObject();
    w.str("id", t.id);
    w.str("title", t.title);
    w.strIf("variant", t.variant);
    w.strIf("benchmark", t.benchmark);
    // Absent means 1, the first.
    if (t.revision > 1) w.integer("revision", t.revision);
    writeDimsKey(w, t.dims);
    w.str("category", categoryString(t.category));
    w.str("shape", shapeString(t.shape));
    w.strIf("axis", t.axis);
    w.str("direction", directionString(t.direction));
    w.str("quantity", quantityString(t.quantity));
    w.str("unit", t.unit);
    w.strIf("description", t.description);
    if (t.durationS > 0.0) w.num("duration_s", t.durationS);
    writeExt(w, t.ext);
    w.beginArray("metrics");
    for (const MetricResult &m : t.metrics) writeMetric(w, m);
    w.endArray();
    w.endObject();
}

void writeCaches(JsonWriter &w, const std::vector<CacheInfo> &caches)
{
    if (caches.empty()) return;
    w.beginArray("caches");
    for (const CacheInfo &c : caches)
    {
        w.beginObjectInline();
        w.str("level", c.level);
        if (c.sizeBytes) w.uint("size_bytes", c.sizeBytes);
        if (c.instances) w.uint("instances", c.instances);
        if (c.lineBytes) w.uint("line_bytes", c.lineBytes);
        w.endObject();
    }
    w.endArray();
}

// A CPU's facts, as keys of the object open: the processor's, or a device's
// `facts`.
void writeCpuFacts(JsonWriter &w, const CpuFacts &c)
{
    w.strIf("isa", c.isa);
    if (c.cores)   w.uint("cores", c.cores);
    if (c.threads) w.uint("threads", c.threads);
    if (c.sockets) w.uint("sockets", c.sockets);
    if (!c.cpuidVendor.empty() || c.cpuidFamily >= 0 || c.cpuidModel >= 0 || c.cpuidStepping >= 0)
    {
        w.beginObjectInline("cpuid");
        w.strIf("vendor", c.cpuidVendor);
        if (c.cpuidFamily >= 0)   w.integer("family", c.cpuidFamily);
        if (c.cpuidModel >= 0)    w.integer("model", c.cpuidModel);
        if (c.cpuidStepping >= 0) w.integer("stepping", c.cpuidStepping);
        w.endObject();
    }
    if (!c.clusters.empty())
    {
        w.beginArray("clusters");
        for (const CpuCluster &k : c.clusters)
        {
            w.beginObject();
            w.strIf("kind", k.kind);
            if (k.cores)          w.uint("cores", k.cores);
            if (k.threadsPerCore) w.uint("threads_per_core", k.threadsPerCore);
            if (k.maxMhz)         w.uint("max_mhz", k.maxMhz);
            w.strIf("microarch", k.microarch);
            w.strIf("midr", k.midr);
            writeCaches(w, k.caches);
            w.endObject();
        }
        w.endArray();
    }
    writeCaches(w, c.caches);
    if (!c.engines.empty())
    {
        w.beginArray("engines");
        for (const CpuEngine &e : c.engines)
        {
            w.beginObjectInline();
            w.strIf("kind", e.kind);
            w.strIf("name", e.name);
            w.strIf("scope", e.scope);
            w.strIf("reached_by", e.reachedBy);
            w.endObject();
        }
        w.endArray();
    }
    if (!c.features.empty())
    {
        w.beginArray("features");
        for (const std::string &f : c.features) w.rawString(f);
        w.endArray();
    }
}

// Typed facts, as keys of the object open.
void writeFacts(JsonWriter &w, const HardwareFacts &f)
{
    w.strIf("vendor_id", f.vendorId);
    w.strIf("device_id", f.deviceId);
    w.strIf("arch", f.arch);
    if (f.integrated >= 0) w.boolean("integrated", f.integrated != 0);
    if (f.computeUnits) w.uint("compute_units", f.computeUnits);
    w.strIf("compute_unit", f.computeUnit);
    if (f.clockMhz) w.uint("clock_mhz", f.clockMhz);
    if (f.memoryBytes) w.uint("memory_bytes", f.memoryBytes);
    writeCpuFacts(w, f.cpu);
}

void writeEvidence(JsonWriter &w, const DeviceEvidence &e)
{
    if (e.empty()) return;
    w.beginObjectInline("evidence");
    for (const auto &kv : e.tokens) w.str(kv.first.c_str(), kv.second);
    if (e.host) w.boolean("host", true);
    w.strIf("soc", e.soc);
    if (e.defaultGpu) w.boolean("default_gpu", true);
    w.endObject();
}

void writeRuntime(JsonWriter &w, const RuntimeInfo &r)
{
    if (r.empty()) return;
    w.beginObjectInline("runtime");
    w.strIf("provider", r.provider);
    w.strIf("accelerator", r.accelerator);
    w.strIf("version", r.version);
    w.endObject();
}

void writePackages(JsonWriter &w, const std::vector<PackageInfo> &packages)
{
    if (packages.empty()) return;
    w.beginArray("packages");
    for (const PackageInfo &p : packages)
    {
        w.beginObject();
        w.str("id", p.id);
        w.strIf("kind", p.kind);
        w.strIf("vendor", p.vendor);
        w.strIf("name", p.name);
        writeExt(w, p.ext);
        w.endObject();
    }
    w.endArray();
}

void writeProcessors(JsonWriter &w, const std::vector<ProcessorInfo> &processors)
{
    if (processors.empty()) return;
    w.beginArray("processors");
    for (const ProcessorInfo &p : processors)
    {
        w.beginObject();
        w.str("id", p.id);
        w.strIf("kind", p.kind);
        w.strIf("package", p.package);
        w.strIf("vendor", p.vendor);
        w.strIf("vendor_name", p.vendorName);
        w.strIf("name", p.name);
        writeFacts(w, p.facts);
        writeExt(w, p.ext);
        w.endObject();
    }
    w.endArray();
}

void writeDevice(JsonWriter &w, const DeviceResult &d)
{
    w.beginObject();
    w.str("backend", d.backend);
    w.str("platform", d.platform);
    w.str("name", d.name);
    w.strIf("driver", d.driver);
    w.str("type", deviceTypeString(d.type));
    if (d.platformIndex >= 0) w.integer("platform_index", d.platformIndex);
    if (d.deviceIndex   >= 0) w.integer("device_index",   d.deviceIndex);
    w.strIf("processor", d.processor);
    w.strIf("link", d.link);
    writeRuntime(w, d.runtime);
    if (!d.facts.empty())
    {
        w.beginObject("facts");
        writeFacts(w, d.facts);
        w.endObject();
    }
    writeEvidence(w, d.evidence);
    writeExt(w, d.ext);
    if (!d.properties.empty())
    {
        w.beginArray("properties");
        for (const DeviceProp &p : d.properties)
        {
            w.beginObject();
            w.str("key", p.key);
            w.str("value", p.value);
            w.endObject();
        }
        w.endArray();
    }
    w.beginArray("tests");
    for (const TestResult &t : d.tests) writeTest(w, t);
    w.endArray();
    w.endObject();
}

void writePlace(JsonWriter &w, const char *key, const RunPlace &p)
{
    if (p.empty()) return;
    w.beginObject(key);
    w.str("backend", p.backend);
    w.strIf("device", p.device);
    if (p.deviceIndex >= 0) w.integer("device_index", p.deviceIndex);
    w.strIf("test", p.test);
    w.endObject();
}

void writeResumed(JsonWriter &w, const std::vector<ResumeRecord> &parts)
{
    if (parts.empty()) return;
    w.beginArray("resumed");
    for (const ResumeRecord &r : parts)
    {
        w.beginObject();
        w.str("at", r.at);
        w.num("elapsed_s", r.elapsedS);
        w.strIf("after", r.after);
        writePlace(w, "from", r.from);
        writePlace(w, "skipped", r.skipped);
        w.endObject();
    }
    w.endArray();
}

void writeProgress(JsonWriter &w, const RunProgress &p)
{
    w.beginObject("progress");
    w.beginArray("backends_done");
    for (const std::string &b : p.backendsDone) w.rawString(b);
    w.endArray();
    w.beginArray("devices");
    for (const RunProgress::Device &d : p.devices)
    {
        w.beginObject();
        w.str("backend", d.backend);
        w.integer("device_index", d.deviceIndex);
        if (d.done)
        {
            w.boolean("done", true);
        }
        else
        {
            w.beginArray("benchmarks_done");
            for (const std::string &b : d.benchmarks) w.rawString(b);
            w.endArray();
            if (!d.held.empty())
            {
                w.beginArray("held_tests");
                for (const std::string &t : d.held) w.rawString(t);
                w.endArray();
            }
        }
        w.endObject();
    }
    w.endArray();
    w.endObject();
}

const char kRunSchema[]     = "xpupeak/run";
const char kRunLogSchema[]  = "xpupeak/run-log";
const char kPartialSchema[] = "xpupeak/run-partial";

// Every document says what wrote it; one with no generator of its own is
// this build's.
void writeGenerator(JsonWriter &w, const GeneratorInfo &g)
{
    w.beginObject("generator");
    w.str("name", g.name.empty() ? std::string(RESULT_GENERATOR_NAME) : g.name);
    w.str("version", g.version.empty() ? std::string(XPUPEAK_VERSION_STR) : g.version);
    w.endObject();
}

// The result document, or with `progress` a run's checkpoint: the same meta
// and devices on one line, and no log or inventory -- the sidecar holds the
// log, and the checkpoint is rewritten whole at every boundary that adds to
// it, so it carries nothing it does not need to.
void writeDocument(const RunDocument &doc, std::ostream &out,
                   const RunProgress *progress = nullptr)
{
    JsonWriter w(out, /*compact=*/progress != nullptr);
    w.beginObject();
    const RunMeta &meta = doc.meta;
    w.str("schema", progress ? kPartialSchema : kRunSchema);
    w.integer("format_version", RESULT_FORMAT_VERSION);
    w.integer("format_minor", RESULT_FORMAT_MINOR);
    w.strIf("run_id", meta.runId);
    writeGenerator(w, meta.generator);
    w.strIf("generated_at", meta.generatedAt);
    if (meta.durationS > 0.0) w.num("duration_s", meta.durationS);
    if (meta.cancelled)       w.boolean("cancelled", true);

    writeBuild(w, meta.build);
    writeInvocation(w, meta.invocation);
    writeSystem(w, meta.system);
    writeConditions(w, meta.conditions);
    writeAnnotations(w, meta.annotations);
    writeResumed(w, meta.resumed);
    writeExt(w, meta.ext);
    if (progress)
    {
        writeProgress(w, *progress);
        // The checkpoint's alone: a resumed run's tokens must match the part
        // before's, and the result must never carry what made them.
        w.strIf("evidence_key", meta.evidenceKey);
    }

    writePackages(w, doc.packages);
    writeProcessors(w, doc.processors);

    w.beginArray("devices");
    for (const DeviceResult &d : doc.devices) writeDevice(w, d);
    w.endArray();

    // Results first; the two blocks a reader consults when they look wrong
    // come after, and the log is the one that can run long.
    if (!progress && !doc.inventory.empty())
    {
        w.beginArray("inventory");
        writeInventoryBackends(w, doc.inventory);
        w.endArray();
    }
    if (!progress && !doc.log.empty())
    {
        w.beginArray("log");
        for (const LogEntry &e : doc.log) writeLogEntry(w, e);
        w.endArray();
    }

    w.endObject();
    out << "\n";
}

} // namespace

std::string dimsToJson(const Dims &dims)
{
    std::ostringstream ss;
    JsonWriter w(ss, /*compact=*/true);
    w.beginObject();
    writeDimEntries(w, dims);
    w.endObject();
    return ss.str();
}

std::string logEntryToJson(const LogEntry &entry)
{
    std::ostringstream ss;
    JsonWriter w(ss);
    writeLogEntry(w, entry);
    return ss.str();
}

std::string runLogHeaderJson(const RunMeta &meta)
{
    std::ostringstream ss;
    JsonWriter w(ss);
    w.beginObjectInline();
    w.str("schema", kRunLogSchema);
    w.integer("format_version", RESULT_FORMAT_VERSION);
    w.integer("format_minor", RESULT_FORMAT_MINOR);
    w.strIf("run_id", meta.runId);
    writeGenerator(w, meta.generator);
    w.strIf("generated_at", meta.generatedAt);
    writeBuild(w, meta.build);
    writeInvocation(w, meta.invocation);
    writeSystem(w, meta.system);
    w.endObject();
    return ss.str();
}

bool saveRunJson(const RunDocument &doc, const std::string &filename)
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << programName() << ": cannot open output file: " << filename << "\n";
        return false;
    }
    writeDocument(doc, f);
    return f.good();
}

std::string runDocumentToJson(const RunDocument &doc)
{
    std::ostringstream ss;
    writeDocument(doc, ss);
    return ss.str();
}

std::string partialRunToJson(const RunDocument &doc, const RunProgress &progress)
{
    std::ostringstream ss;
    writeDocument(doc, ss, &progress);
    return ss.str();
}

// ── JSON loader ────────────────────────────────────────────────────────────

namespace {

// Everything below reads what it knows and passes over the rest: a key,
// a value or a dim a newer build wrote is never an error (docs/format-v4.md,
// Compatibility).  Vocabulary values are kept as written.

std::string readExt(const JsonValue &v)
{
    const JsonValue *e = v.find("ext");
    return (e && e->isObject()) ? jsonSerialize(*e) : std::string();
}

unsigned readUint(const JsonValue &v, const char *key)
{
    const double n = v.num(key);
    return n > 0.0 ? static_cast<unsigned>(n) : 0u;
}

std::uint64_t readBytes(const JsonValue &v, const char *key)
{
    const double n = v.num(key);
    return n > 0.0 ? static_cast<std::uint64_t>(n) : 0u;
}

int readTri(const JsonValue &v, const char *key)
{
    const JsonValue *b = v.find(key);
    return (b && b->type() == JsonValue::Type::Bool) ? (b->asBool() ? 1 : 0) : -1;
}

// Strings and whole numbers; a value of another type is not a dim.
Dims readDims(const JsonValue &v)
{
    Dims out;
    const JsonValue *d = v.find("dims");
    if (!d || !d->isObject()) return out;
    for (const auto &kv : d->obj)
    {
        if (kv.second.type() == JsonValue::Type::String)
            out.entries.emplace_back(kv.first, DimValue::ofText(kv.second.asString()));
        else if (kv.second.type() == JsonValue::Type::Number)
            out.entries.emplace_back(kv.first, DimValue::ofNumber(kv.second.asNumber()));
    }
    return out;
}

void readSystem(const JsonValue &v, SystemInfo &s)
{
    if (!v.isObject()) return;
    s.formFactor       = v.str("form_factor");
    s.formFactorDetail = v.str("form_factor_detail");
    s.formFactorFrom   = v.str("form_factor_from");
    s.isVirtual        = v.flag("virtual");
    s.translated       = v.flag("translated");
    s.manufacturer     = v.str("manufacturer");
    s.brand            = v.str("brand");
    s.model            = v.str("model");
    s.codename         = v.str("codename");
    s.name             = v.str("name");
    s.family           = v.str("family");
    s.board            = v.str("board");
    s.soc              = v.str("soc");
    if (const JsonValue *os = v.find("os"); os && os->isObject())
    {
        s.os.name         = os->str("name");
        s.os.version      = os->str("version");
        s.os.build        = os->str("build");
        s.os.kernel       = os->str("kernel");
        s.os.distribution = os->str("distribution");
    }
    s.arch        = v.str("arch");
    s.memoryBytes = readBytes(v, "memory_bytes");
    s.ext         = readExt(v);
}

void readConditions(const JsonValue &v, RunConditions &c)
{
    if (!v.isObject()) return;
    c.powerSource  = v.str("power_source");
    c.lowPowerMode = readTri(v, "low_power_mode");
    c.powerProfile = v.str("power_profile");
    c.cpuGovernor  = v.str("cpu_governor");
    c.thermalStart = v.str("thermal_start");
    c.thermalEnd   = v.str("thermal_end");
    c.ext          = readExt(v);
}

std::vector<CacheInfo> readCaches(const JsonValue &v)
{
    std::vector<CacheInfo> out;
    if (const JsonValue *cs = v.find("caches"))
        for (const JsonValue &c : cs->items())
        {
            CacheInfo info;
            info.level     = c.str("level");
            info.sizeBytes = readBytes(c, "size_bytes");
            info.instances = readUint(c, "instances");
            info.lineBytes = readUint(c, "line_bytes");
            out.push_back(std::move(info));
        }
    return out;
}

// Facts from the keys of `v`: a processor's own, or a device's `facts`.
void readFacts(const JsonValue &v, HardwareFacts &f)
{
    if (!v.isObject()) return;
    f.vendorId     = v.str("vendor_id");
    f.deviceId     = v.str("device_id");
    f.arch         = v.str("arch");
    f.integrated   = readTri(v, "integrated");
    f.computeUnits = readUint(v, "compute_units");
    f.computeUnit  = v.str("compute_unit");
    f.clockMhz     = readUint(v, "clock_mhz");
    f.memoryBytes  = readBytes(v, "memory_bytes");

    CpuFacts &c = f.cpu;
    c.isa     = v.str("isa");
    c.cores   = readUint(v, "cores");
    c.threads = readUint(v, "threads");
    c.sockets = readUint(v, "sockets");
    if (const JsonValue *id = v.find("cpuid"); id && id->isObject())
    {
        c.cpuidVendor   = id->str("vendor");
        c.cpuidFamily   = static_cast<int>(id->num("family", -1));
        c.cpuidModel    = static_cast<int>(id->num("model", -1));
        c.cpuidStepping = static_cast<int>(id->num("stepping", -1));
    }
    if (const JsonValue *ks = v.find("clusters"))
        for (const JsonValue &k : ks->items())
        {
            CpuCluster cl;
            cl.kind           = k.str("kind");
            cl.cores          = readUint(k, "cores");
            cl.threadsPerCore = readUint(k, "threads_per_core");
            cl.maxMhz         = readUint(k, "max_mhz");
            cl.microarch      = k.str("microarch");
            cl.midr           = k.str("midr");
            cl.caches         = readCaches(k);
            c.clusters.push_back(std::move(cl));
        }
    c.caches = readCaches(v);
    if (const JsonValue *es = v.find("engines"))
        for (const JsonValue &e : es->items())
            c.engines.push_back({e.str("kind"), e.str("name"), e.str("scope"), e.str("reached_by")});
    if (const JsonValue *fs = v.find("features"))
        for (const JsonValue &x : fs->items())
            if (x.type() == JsonValue::Type::String) c.features.push_back(x.asString());
}

// The weak evidence by name; every other string is a token, a kind a newer
// build added included.
void readEvidence(const JsonValue &v, DeviceEvidence &e)
{
    if (!v.isObject()) return;
    for (const auto &kv : v.obj)
    {
        if (kv.first == "host")
            e.host = kv.second.asBool();
        else if (kv.first == "default_gpu")
            e.defaultGpu = kv.second.asBool();
        else if (kv.first == "soc")
            e.soc = kv.second.asString();
        else if (kv.second.type() == JsonValue::Type::String)
            e.tokens[kv.first] = kv.second.asString();
    }
}

void readRuntime(const JsonValue &v, RuntimeInfo &r)
{
    if (!v.isObject()) return;
    r.provider    = v.str("provider");
    r.accelerator = v.str("accelerator");
    r.version     = v.str("version");
}

void readBuild(const JsonValue &v, BuildInfo &b)
{
    if (!v.isObject()) return;
    if (const JsonValue *names = v.find("backends"))
        for (const JsonValue &x : names->items())
        {
            BuildBackend be;
            if (x.type() == JsonValue::Type::String)
            {
                be.name = x.asString();
            }
            else
            {
                be.name     = x.str("name");
                be.kind     = x.str("kind");
                be.portable = x.flag("portable");
            }
            if (!be.name.empty()) b.backends.push_back(std::move(be));
        }
    b.config = v.str("config");
}

void readInvocation(const JsonValue &v, Invocation &inv)
{
    if (!v.isObject()) return;
    if (const JsonValue *a = v.find("argv"))
        for (const JsonValue &x : a->items()) inv.argv.push_back(x.asString());
    inv.targetTimeUs      = static_cast<unsigned>(v.num("target_time_us"));
    inv.targetTimeUsCpu   = static_cast<unsigned>(v.num("target_time_us_cpu"));
    inv.targetTimeUsMlGpu = static_cast<unsigned>(v.num("target_time_us_ml_gpu"));
    inv.warmup            = static_cast<unsigned>(v.num("warmup"));
    inv.iters             = static_cast<unsigned>(v.num("iters"));
    if (const JsonValue *c = v.find("categories"))
        for (const JsonValue &x : c->items()) inv.categories.push_back(x.asString());
    if (const JsonValue *t = v.find("tests"))
        for (const JsonValue &x : t->items()) inv.tests.push_back(x.asString());
    inv.verbose  = v.flag("verbose");
    inv.standard = v.flag("standard");
}

LogEntry readLogEntry(const JsonValue &v)
{
    LogEntry e;
    e.elapsedS    = v.num("elapsed_s");
    e.level       = xpupeak::logLevelFromString(v.str("level"));
    e.source      = v.str("source");
    e.backend     = v.str("backend");
    e.device      = v.str("device");
    e.deviceIndex = static_cast<int>(v.num("device_index", -1));
    e.test        = v.str("test");
    e.message     = v.str("message");
    return e;
}

MetricResult readMetric(const JsonValue &v)
{
    MetricResult m;
    m.id          = v.str("id");
    m.label       = v.str("label");
    m.description = v.str("description");
    m.reason      = v.str("reason");
    // A reading with a "value" is ok; one without carries its status.  Reading
    // it this way keeps the writer's omit-the-obvious convention honest.
    if (v.has("value"))
    {
        m.status = ResultStatus::Ok;
        m.value  = v.num("value");
    }
    else
    {
        m.status = statusFromString(v.str("status", "ok"));
    }
    if (v.has("unit"))
    {
        m.hasUnit  = true;
        m.unit     = v.str("unit");
        m.quantity = quantityFromString(v.str("quantity"));
    }
    if (v.has("direction"))
        m.direction = directionFromString(v.str("direction"));
    m.dims      = readDims(v);
    m.benchmark = v.str("benchmark");
    m.ext       = readExt(v);
    return m;
}

TestResult readTest(const JsonValue &v)
{
    TestResult t;
    t.id          = v.str("id");
    t.title       = v.str("title", t.id);
    t.variant     = v.str("variant");
    t.axis        = v.str("axis");
    t.description = v.str("description");
    t.category    = categoryFromString(v.str("category"));
    t.shape       = shapeFromString(v.str("shape"));
    t.direction   = directionFromString(v.str("direction"));
    t.quantity    = quantityFromString(v.str("quantity"));
    t.unit        = v.str("unit");
    t.durationS   = v.num("duration_s");
    t.benchmark   = v.str("benchmark");
    t.dims        = readDims(v);
    t.revision    = std::max(1, static_cast<int>(v.num("revision", 1)));
    t.ext         = readExt(v);
    if (const JsonValue *ms = v.find("metrics"))
        for (const JsonValue &m : ms->items()) t.metrics.push_back(readMetric(m));
    return t;
}

DeviceResult readDevice(const JsonValue &v)
{
    DeviceResult d;
    d.backend       = v.str("backend");
    d.platform      = v.str("platform", d.backend);
    d.name          = v.str("name");
    d.driver        = v.str("driver");
    d.type          = deviceTypeFromString(v.str("type"));
    d.platformIndex = static_cast<int>(v.num("platform_index", -1));
    d.deviceIndex   = static_cast<int>(v.num("device_index", -1));
    if (const JsonValue *ps = v.find("properties"))
        for (const JsonValue &p : ps->items())
            d.properties.push_back({p.str("key"), p.str("value")});
    d.processor = v.str("processor");
    d.link      = v.str("link");
    if (const JsonValue *f = v.find("facts"))    readFacts(*f, d.facts);
    if (const JsonValue *e = v.find("evidence")) readEvidence(*e, d.evidence);
    if (const JsonValue *r = v.find("runtime"))  readRuntime(*r, d.runtime);
    d.ext = readExt(v);
    if (const JsonValue *ts = v.find("tests"))
        for (const JsonValue &t : ts->items()) d.tests.push_back(readTest(t));
    return d;
}

RunPlace readPlace(const JsonValue *v)
{
    RunPlace p;
    if (!v || !v->isObject()) return p;
    p.backend     = v->str("backend");
    p.device      = v->str("device");
    p.deviceIndex = static_cast<int>(v->num("device_index", -1));
    p.test        = v->str("test");
    return p;
}

void readProgress(const JsonValue &v, RunProgress &p)
{
    if (const JsonValue *bs = v.find("backends_done"))
        for (const JsonValue &b : bs->items()) p.backendsDone.push_back(b.asString());
    if (const JsonValue *ds = v.find("devices"))
        for (const JsonValue &d : ds->items())
        {
            RunProgress::Device dev;
            dev.backend     = d.str("backend");
            dev.deviceIndex = static_cast<int>(d.num("device_index", -1));
            dev.done        = d.flag("done");
            if (const JsonValue *bs = d.find("benchmarks_done"))
                for (const JsonValue &b : bs->items()) dev.benchmarks.push_back(b.asString());
            if (const JsonValue *hs = d.find("held_tests"))
                for (const JsonValue &h : hs->items()) dev.held.push_back(h.asString());
            p.devices.push_back(std::move(dev));
        }
}

// Everything but the schema check, shared by the result loader and the
// checkpoint's.
void readDocument(const JsonValue &root, RunDocument &out)
{
    out = RunDocument();
    if (const JsonValue *g = root.find("generator"); g && g->isObject())
    {
        out.meta.generator.name    = g->str("name");
        out.meta.generator.version = g->str("version");
    }
    out.meta.generatedAt   = root.str("generated_at");
    out.meta.durationS     = root.num("duration_s");
    out.meta.cancelled     = root.flag("cancelled");
    out.meta.runId         = root.str("run_id");
    out.meta.formatMinor   = std::max(0, static_cast<int>(root.num("format_minor", 0)));
    if (const JsonValue *b = root.find("build"))      readBuild(*b, out.meta.build);
    if (const JsonValue *s = root.find("system"))     readSystem(*s, out.meta.system);
    if (const JsonValue *c = root.find("conditions")) readConditions(*c, out.meta.conditions);
    if (const JsonValue *i = root.find("invocation")) readInvocation(*i, out.meta.invocation);
    if (const JsonValue *a = root.find("annotations"); a && a->isObject())
        for (const auto &kv : a->obj)
            if (kv.second.type() == JsonValue::Type::String)
                out.meta.annotations[kv.first] = kv.second.asString();
    out.meta.ext = readExt(root);
    if (const JsonValue *rs = root.find("resumed"))
        for (const JsonValue &r : rs->items())
        {
            ResumeRecord rec;
            rec.at       = r.str("at");
            rec.elapsedS = r.num("elapsed_s");
            rec.after    = r.str("after");
            rec.from     = readPlace(r.find("from"));
            rec.skipped  = readPlace(r.find("skipped"));
            out.meta.resumed.push_back(std::move(rec));
        }
    if (const JsonValue *ps = root.find("packages"))
        for (const JsonValue &p : ps->items())
        {
            PackageInfo pkg;
            pkg.id     = p.str("id");
            pkg.kind   = p.str("kind");
            pkg.vendor = p.str("vendor");
            pkg.name   = p.str("name");
            pkg.ext    = readExt(p);
            out.packages.push_back(std::move(pkg));
        }
    if (const JsonValue *ps = root.find("processors"))
        for (const JsonValue &p : ps->items())
        {
            ProcessorInfo proc;
            proc.id         = p.str("id");
            proc.kind       = p.str("kind");
            proc.package    = p.str("package");
            proc.vendor     = p.str("vendor");
            proc.vendorName = p.str("vendor_name");
            proc.name       = p.str("name");
            readFacts(p, proc.facts);
            proc.ext        = readExt(p);
            out.processors.push_back(std::move(proc));
        }
    if (const JsonValue *ds = root.find("devices"))
        for (const JsonValue &d : ds->items()) out.devices.push_back(readDevice(d));
    // `inventory` is not read back: this loader serves --compare, which needs
    // readings, and the GUI reads the file with its own parser.
    if (const JsonValue *l = root.find("log"))
        for (const JsonValue &x : l->items()) out.log.push_back(readLogEntry(x));
}

// Every schema clpeak 3.x wrote starts clpeak/run.  4.0 reads none of
// them -- no import, no converter -- and says which program wrote the file
// rather than only that its version is not this one's.
bool writtenBy3x(const JsonValue &root)
{
    return root.str("schema").rfind("clpeak/run", 0) == 0;
}

std::string refusal3x(const std::string &filename)
{
    return filename + " was written by clpeak 3.x, which this version does not read";
}

bool readFile(const std::string &filename, std::string &text)
{
    std::ifstream f(filename, std::ios::binary);
    if (!f.is_open()) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    text = ss.str();
    return true;
}

} // namespace

bool parseRunJson(const std::string &text, const std::string &name, RunDocument &out,
                  std::string &error, bool &newerMinor)
{
    newerMinor = false;
    std::string why;
    JsonValue   root = jsonParse(text, &why);
    if (!root.isObject())
    {
        error = name + " is not an XpuPeak result file" + (why.empty() ? "" : " (" + why + ")");
        return false;
    }
    if (writtenBy3x(root))
    {
        error = refusal3x(name);
        return false;
    }
    if (root.str("schema") == kPartialSchema)
    {
        error = name + " is the checkpoint of a run that did not finish, not a result; "
                "continue the run with --resume";
        return false;
    }

    // The major is the promise: every minor of it is read, and what a newer
    // one added is passed over.
    const int version = static_cast<int>(root.num("format_version", 0));
    if (version != RESULT_FORMAT_VERSION)
    {
        error = name + " is format_version=" + std::to_string(version) + "; this build expects v" +
                std::to_string(RESULT_FORMAT_VERSION) +
                ". Regenerate the file with this version of XpuPeak.";
        return false;
    }

    readDocument(root, out);
    newerMinor = out.meta.formatMinor > RESULT_FORMAT_MINOR;
    return true;
}

bool loadRunJson(const std::string &filename, RunDocument &out)
{
    std::string text;
    if (!readFile(filename, text))
    {
        std::cerr << programName() << ": cannot open result file: " << filename << "\n";
        return false;
    }
    std::string error;
    bool        newer = false;
    if (!parseRunJson(text, filename, out, error, newer))
    {
        std::cerr << programName() << ": " << error << "\n";
        return false;
    }
    // Said once per file: each backend's logger reads a --compare baseline.
    static std::mutex            saidLock;
    static std::set<std::string> said;
    if (newer)
    {
        std::lock_guard<std::mutex> lock(saidLock);
        if (said.insert(filename).second)
            std::cerr << programName() << ": " << filename << " was written by a newer XpuPeak "
                      << "(format " << RESULT_FORMAT_VERSION << "." << out.meta.formatMinor
                      << "); some details aren't shown\n";
    }
    return true;
}

bool logEntryFromJson(const std::string &line, LogEntry &out)
{
    JsonValue v = jsonParse(line);
    if (!v.isObject() || !v.has("message") || !v.has("level")) return false;
    out = readLogEntry(v);
    return true;
}

bool loadPartialJson(const std::string &filename, RunDocument &out,
                     RunProgress &progress, std::string &error)
{
    std::string text;
    if (!readFile(filename, text))
    {
        error = "cannot open " + filename;
        return false;
    }
    std::string why;
    JsonValue   root = jsonParse(text, &why);
    if (root.isObject() && writtenBy3x(root))
    {
        error = refusal3x(filename);
        return false;
    }
    if (!root.isObject() || root.str("schema") != kPartialSchema)
    {
        error = filename + " is not an XpuPeak run checkpoint" +
                (why.empty() ? "" : " (" + why + ")");
        return false;
    }
    const int version = static_cast<int>(root.num("format_version", 0));
    if (version != RESULT_FORMAT_VERSION)
    {
        error = filename + " is format_version=" + std::to_string(version) +
                "; this build reads v" + std::to_string(RESULT_FORMAT_VERSION);
        return false;
    }
    readDocument(root, out);
    out.meta.evidenceKey = root.str("evidence_key");
    progress = RunProgress();
    if (const JsonValue *p = root.find("progress")) readProgress(*p, progress);
    return true;
}
