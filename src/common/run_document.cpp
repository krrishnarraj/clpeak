#include <common/run_document.h>
#include <common/common.h>
#include <common/json.h>
#include <version.h>

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>

// ── Enum <-> string ────────────────────────────────────────────────────────

const char *categoryString(Category c)
{
    switch (c)
    {
    case Category::FpCompute:  return "fp_compute";
    case Category::IntCompute: return "int_compute";
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
    if (s == "fp_compute")  return Category::FpCompute;
    if (s == "int_compute") return Category::IntCompute;
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
    case DeviceType::Unknown:     break;
    }
    return "unknown";
}

DeviceType deviceTypeFromString(const std::string &s)
{
    if (s == "cpu")         return DeviceType::Cpu;
    if (s == "gpu")         return DeviceType::Gpu;
    if (s == "accelerator") return DeviceType::Accelerator;
    return DeviceType::Unknown;
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
        // device block the GUI would show as a separate run.
        for (const TestResult &t : d.tests)
        {
            TestResult *known = existing->findTest(t.key());
            if (!known) { existing->tests.push_back(t); continue; }
            known->metrics.insert(known->metrics.end(),
                                  t.metrics.begin(), t.metrics.end());
        }
        if (existing->properties.empty()) existing->properties = d.properties;
    }
    notes.insert(notes.end(), other.notes.begin(), other.notes.end());
}

// ── Baselines ──────────────────────────────────────────────────────────────

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
                    m[baselineKey(d.backend, d.platform, d.name, t.key(), r.id)] =
                        r.value;
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

// ── JSON writer ────────────────────────────────────────────────────────────

namespace {

// Every number written here goes through the classic locale.  The GUI hosts
// this writer inside toolkits that call setlocale(LC_ALL, "") -- GTK's
// gtk_init does -- and a comma decimal separator produces a file that is not
// JSON at all.  The dump is machine interchange, never user-facing text.
//
// Seven significant digits: enough to round-trip a float's worth of precision
// (measurements are floats), enough to keep a six-digit GFLOPS reading whole,
// and -- unlike the fixed four decimals this replaces -- it does not flatten
// the ONNX numeric-error readings, which are parts-per-million and used to be
// written as 0.0000.
std::string fmtNum(double v)
{
    std::ostringstream ss;
    ss.imbue(std::locale::classic());
    ss << std::setprecision(7) << v;
    return ss.str();
}

std::string fmtUint(std::uint64_t v)
{
    std::ostringstream ss;
    ss.imbue(std::locale::classic());
    ss << v;
    return ss.str();
}

// Indent-tracking JSON emitter.  The document is pretty-printed because it is
// now the only format clpeak writes: people read these files, diff them, and
// paste them into bug reports.
class Writer
{
public:
    explicit Writer(std::ostream &o) : out(o) { out.imbue(std::locale::classic()); }

    void beginObject()          { punctuate(); out << "{";  depth++; fresh = true; }
    void beginObject(const char *k) { key(k);  out << "{";  depth++; fresh = true; }
    void endObject()            { depth--; newline(); out << "}"; fresh = false; }
    void beginArray(const char *k)  { key(k);  out << "[";  depth++; fresh = true; }
    void endArray()             { depth--; newline(); out << "]"; fresh = false; }

    void num(const char *k, double v)        { key(k); out << fmtNum(v); }
    void uint(const char *k, std::uint64_t v){ key(k); out << fmtUint(v); }
    void integer(const char *k, long long v) { key(k); out << v; }
    void boolean(const char *k, bool v)      { key(k); out << (v ? "true" : "false"); }
    void str(const char *k, const std::string &v)
    {
        key(k);
        out << "\"" << jsonEscape(v) << "\"";
    }
    // Optional string: absent rather than empty, so a file carries only facts.
    void strIf(const char *k, const std::string &v) { if (!v.empty()) str(k, v); }

    void rawString(const std::string &v)
    {
        punctuate();
        out << "\"" << jsonEscape(v) << "\"";
    }

private:
    void newline()
    {
        out << "\n" << std::string(static_cast<size_t>(depth) * 2, ' ');
    }
    void punctuate()
    {
        if (!fresh) out << ",";
        newline();
        fresh = false;
    }
    void key(const char *k)
    {
        punctuate();
        out << "\"" << k << "\": ";
    }

    std::ostream &out;
    int  depth = 0;
    bool fresh = true;   // nothing written at this depth yet -> no leading comma
};

void writeHost(Writer &w, const HostInfo &h)
{
    w.beginObject("host");
    w.strIf("os",         h.os);
    w.strIf("os_version", h.osVersion);
    w.strIf("arch",       h.arch);
    w.strIf("cpu",        h.cpu);
    if (h.logicalCores) w.uint("logical_cores", h.logicalCores);
    if (h.memoryBytes)  w.uint("memory_bytes",  h.memoryBytes);
    w.endObject();
}

void writeInvocation(Writer &w, const Invocation &inv)
{
    w.beginObject("invocation");
    if (!inv.argv.empty())
    {
        w.beginArray("argv");
        for (const std::string &a : inv.argv) w.rawString(a);
        w.endArray();
    }
    if (inv.targetTimeUs)    w.uint("target_time_us",     inv.targetTimeUs);
    if (inv.targetTimeUsCpu) w.uint("target_time_us_cpu", inv.targetTimeUsCpu);
    if (inv.warmup)          w.uint("warmup",             inv.warmup);
    // Absent means "calibrated per test", which is the normal mode; a number
    // means the run was pinned to it with -i and is not comparable to one that
    // was not.
    if (inv.iters)           w.uint("iters",              inv.iters);
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
    w.endObject();
}

void writeMetric(Writer &w, const MetricResult &m)
{
    w.beginObject();
    w.str("id", m.id);
    w.strIf("label", m.label);
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
        w.num("scale", m.scale);
    }
    if (m.direction != Direction::FromUnit)
        w.str("direction", directionString(m.direction));
    w.strIf("description", m.description);
    w.endObject();
}

void writeTest(Writer &w, const TestResult &t)
{
    w.beginObject();
    w.str("id", t.id);
    w.str("title", t.title);
    w.strIf("variant", t.variant);
    w.str("category", categoryString(t.category));
    w.str("shape", shapeString(t.shape));
    w.strIf("axis", t.axis);
    w.str("direction", directionString(t.direction));
    w.str("quantity", quantityString(t.quantity));
    w.str("unit", t.unit);
    w.num("scale", t.scale);
    w.strIf("description", t.description);
    w.beginArray("metrics");
    for (const MetricResult &m : t.metrics) writeMetric(w, m);
    w.endArray();
    w.endObject();
}

void writeDevice(Writer &w, const DeviceResult &d)
{
    w.beginObject();
    w.str("backend", d.backend);
    w.str("platform", d.platform);
    w.str("name", d.name);
    w.strIf("driver", d.driver);
    w.str("type", deviceTypeString(d.type));
    if (d.platformIndex >= 0) w.integer("platform_index", d.platformIndex);
    if (d.deviceIndex   >= 0) w.integer("device_index",   d.deviceIndex);
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

void writeDocument(const RunDocument &doc, std::ostream &out)
{
    Writer w(out);
    w.beginObject();
    w.str("schema", "clpeak/run");
    w.integer("format_version", RESULT_FORMAT_VERSION);
    w.str("clpeak_version", doc.meta.clpeakVersion.empty()
                                ? std::string(CLPEAK_VERSION_STR)
                                : doc.meta.clpeakVersion);
    w.strIf("generated_at", doc.meta.generatedAt);
    if (doc.meta.durationS > 0.0) w.num("duration_s", doc.meta.durationS);
    if (doc.meta.cancelled)       w.boolean("cancelled", true);

    writeHost(w, doc.meta.host);
    writeInvocation(w, doc.meta.invocation);

    if (!doc.notes.empty())
    {
        w.beginArray("notes");
        for (const RunNote &n : doc.notes)
        {
            w.beginObject();
            w.strIf("backend", n.backend);
            w.strIf("device",  n.device);
            w.str("message",   n.message);
            w.endObject();
        }
        w.endArray();
    }

    w.beginArray("devices");
    for (const DeviceResult &d : doc.devices) writeDevice(w, d);
    w.endArray();

    w.endObject();
    out << "\n";
}

} // namespace

bool saveRunJson(const RunDocument &doc, const std::string &filename)
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "clpeak: cannot open output file: " << filename << "\n";
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

// ── JSON loader ────────────────────────────────────────────────────────────

namespace {

void readHost(const JsonValue &v, HostInfo &h)
{
    if (!v.isObject()) return;
    h.os           = v.str("os");
    h.osVersion    = v.str("os_version");
    h.arch         = v.str("arch");
    h.cpu          = v.str("cpu");
    h.logicalCores = static_cast<unsigned>(v.num("logical_cores"));
    h.memoryBytes  = static_cast<std::uint64_t>(v.num("memory_bytes"));
}

void readInvocation(const JsonValue &v, Invocation &inv)
{
    if (!v.isObject()) return;
    if (const JsonValue *a = v.find("argv"))
        for (const JsonValue &x : a->items()) inv.argv.push_back(x.asString());
    inv.targetTimeUs    = static_cast<unsigned>(v.num("target_time_us"));
    inv.targetTimeUsCpu = static_cast<unsigned>(v.num("target_time_us_cpu"));
    inv.warmup          = static_cast<unsigned>(v.num("warmup"));
    inv.iters           = static_cast<unsigned>(v.num("iters"));
    if (const JsonValue *c = v.find("categories"))
        for (const JsonValue &x : c->items()) inv.categories.push_back(x.asString());
    if (const JsonValue *t = v.find("tests"))
        for (const JsonValue &x : t->items()) inv.tests.push_back(x.asString());
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
        m.scale    = v.num("scale", 1.0);
    }
    if (v.has("direction"))
        m.direction = directionFromString(v.str("direction"));
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
    t.scale       = v.num("scale", 1.0);
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
    if (const JsonValue *ts = v.find("tests"))
        for (const JsonValue &t : ts->items()) d.tests.push_back(readTest(t));
    return d;
}

} // namespace

bool loadRunJson(const std::string &filename, RunDocument &out)
{
    std::ifstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "clpeak: cannot open result file: " << filename << "\n";
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();

    std::string error;
    JsonValue   root = jsonParse(ss.str(), &error);
    if (!root.isObject())
    {
        std::cerr << "clpeak: " << filename << " is not a clpeak result file"
                  << (error.empty() ? "" : " (" + error + ")") << "\n";
        return false;
    }

    const int version = static_cast<int>(root.num("format_version", 0));
    if (version != RESULT_FORMAT_VERSION)
    {
        std::cerr << "clpeak: " << filename << " is format_version=" << version
                  << "; this build expects v" << RESULT_FORMAT_VERSION
                  << ". Regenerate the file with this version of clpeak.\n";
        return false;
    }

    out = RunDocument();
    out.meta.clpeakVersion = root.str("clpeak_version");
    out.meta.generatedAt   = root.str("generated_at");
    out.meta.durationS     = root.num("duration_s");
    out.meta.cancelled     = root.flag("cancelled");
    if (const JsonValue *h = root.find("host"))       readHost(*h, out.meta.host);
    if (const JsonValue *i = root.find("invocation")) readInvocation(*i, out.meta.invocation);
    if (const JsonValue *n = root.find("notes"))
        for (const JsonValue &x : n->items())
            out.notes.push_back({x.str("backend"), x.str("device"), x.str("message")});
    if (const JsonValue *ds = root.find("devices"))
        for (const JsonValue &d : ds->items()) out.devices.push_back(readDevice(d));

    return true;
}
