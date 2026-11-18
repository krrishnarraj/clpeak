#include <common/measures.h>

#include <common/json.h>
#include <common/options.h>      // benchmarkFromFlag
#include <common/vocabulary.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// The engine of docs/format-v4.md, Comparing runs.  The app's engine
// (app/lib/src/model/measures.dart, headlines.dart) is the other half of a
// pair: each step below says which of its lines it mirrors, and the two must
// stay in step -- a rule changed here is changed there in the same commit.

namespace xpupeak {

extern const unsigned char kBundledCatalogJson[];
extern const std::size_t   kBundledCatalogJsonSize;

namespace {

std::string trimmed(const std::string &s)
{
    const char *ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

// The shortest text that reads back as `n`, in the form Dart's
// double.toString() gives: fixed between 1e-6 and 1e21, an exponent outside.
// Only a dim the format does not allow -- a fraction -- ever needs it.
std::string shortestDouble(double n)
{
    if (std::isnan(n)) return "NaN";
    if (std::isinf(n)) return n < 0 ? "-Infinity" : "Infinity";
    char buf[64];
    int precision = 1;
    for (; precision <= 17; precision++)
    {
        std::snprintf(buf, sizeof(buf), "%.*e", precision - 1, n);
        if (std::strtod(buf, nullptr) == n) break;
    }
    // buf is d.ddde±XX with `precision` digits.
    std::string text = buf;
    const size_t e = text.find('e');
    std::string mantissa = text.substr(0, e);
    const int exponent = std::atoi(text.c_str() + e + 1);
    const bool negative = !mantissa.empty() && mantissa[0] == '-';
    if (negative) mantissa.erase(0, 1);
    std::string digits;
    for (char c : mantissa)
        if (c != '.') digits += c;
    const double a = std::fabs(n);
    std::string out;
    if (a >= 1e-6 && a < 1e21)
    {
        // Fixed notation with exactly those digits.
        const int point = exponent + 1;   // digits before the decimal point
        if (point <= 0)
            out = "0." + std::string(static_cast<size_t>(-point), '0') + digits;
        else if (point >= static_cast<int>(digits.size()))
            out = digits + std::string(static_cast<size_t>(point) - digits.size(), '0') + ".0";
        else
            out = digits.substr(0, static_cast<size_t>(point)) + "." +
                  digits.substr(static_cast<size_t>(point));
    }
    else
    {
        out = digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        out += std::string("e") + (exponent < 0 ? "-" : "+") + std::to_string(std::abs(exponent));
    }
    return negative ? "-" + out : out;
}

// dimDefaultFor() (vocabulary.h) with "no default" told apart from a
// default: measures.dart's dimDefaultFor() returns null for the first, and a
// value only ever equals the second.
bool defaultOf(const std::string &key, const std::string &family, const std::string &dtype,
               std::string &out)
{
    Dim d;
    if (!fromString(key, d)) return false;
    const DimKeyInfo &info = dimInfo(d);
    Benchmark b = Benchmark::COUNT;
    const FamilyInfo *fam = familyFromString(family, b) ? &familyInfo(b) : nullptr;
    std::string type = dtype;
    if (type.empty() && fam && fam->hasDtype) type = toString(fam->dtype);
    switch (info.def)
    {
    case DimDefault::None:
        return false;
    case DimDefault::Value:
        out = info.defaultValue;
        return true;
    case DimDefault::FamilyDtype:
        if (!fam || !fam->hasDtype) return false;
        out = toString(fam->dtype);
        return true;
    case DimDefault::SameAsDtype:
        if (type.empty()) return false;
        out = type;
        return true;
    case DimDefault::FamilyAcc:
    {
        if (fam && fam->acc == AccRule::Fixed)
        {
            out = toString(fam->accType);
            return true;
        }
        DType t;
        if (!fromString(type, t)) return false;
        out = (fam && fam->acc == AccRule::SameAsDtype) ? toString(t) : toString(usualAccumulator(t));
        return true;
    }
    }
    return false;
}

const FamilyInfo *familyOf(const std::string &family)
{
    Benchmark b;
    return familyFromString(family, b) ? &familyInfo(b) : nullptr;
}

std::string quantityText(Quantity q, const std::string &word)
{
    return q == Quantity::Unknown && !word.empty() ? word : std::string(quantityString(q));
}

// A dims object from the catalog's JSON: every value that is not null, a
// string or a number kept as its kind.
Dims dimsFromJson(const JsonValue &v)
{
    Dims out;
    for (const auto &kv : v.obj)
    {
        if (kv.second.type() == JsonValue::Type::String)
            out.entries.emplace_back(kv.first, DimValue::ofText(kv.second.asString()));
        else if (kv.second.type() == JsonValue::Type::Number)
            out.entries.emplace_back(kv.first, DimValue::ofNumber(kv.second.asNumber()));
        else if (kv.second.type() == JsonValue::Type::Bool)
            out.entries.emplace_back(kv.first, DimValue::ofText(kv.second.asBool() ? "true" : "false"));
    }
    return out;
}

std::vector<std::string> stringsOf(const JsonValue &v)
{
    std::vector<std::string> out;
    for (const JsonValue &x : v.items())
        if (x.type() == JsonValue::Type::String) out.push_back(x.asString());
    return out;
}

std::string bytesWords(const std::string &v)
{
    char *end = nullptr;
    const long long n = std::strtoll(v.c_str(), &end, 10);
    if (v.empty() || !end || *end) return v;
    const char *units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double size = static_cast<double>(n);
    int i = 0;
    while (size >= 1024 && i < 4)
    {
        size /= 1024;
        i++;
    }
    char buf[64];
    if (size == std::trunc(size))
        std::snprintf(buf, sizeof(buf), "%lld %s", static_cast<long long>(size), units[i]);
    else
        std::snprintf(buf, sizeof(buf), "%.1f %s", size, units[i]);
    return buf;
}

std::string underscoresAsSpaces(std::string s)
{
    std::replace(s.begin(), s.end(), '_', ' ');
    return s;
}

// measures.dart's _dimWords: one identity dim in a person's words.
std::string dimWords(const std::string &key, const std::string &v)
{
    if (key == "dtype") return "";
    if (key == "weights") return dataTypeLabel(v) + " weights";
    if (key == "kv") return dataTypeLabel(v) + " KV cache";
    if (key == "acc") return dataTypeLabel(v) + " accumulate";
    if (key == "quant")
    {
        if (v == "qdq") return "QDQ";
        if (v == "lut") return "lookup-table weights";
        if (v == "block") return "block-scaled";
        if (v == "channel") return "per-channel scales";
        return v;
    }
    if (key == "sparsity") return v == "2:4" ? "2:4 structured sparsity" : v + " sparsity";
    if (key == "scope") return v == "core" ? "one core" : v;
    if (key == "level")
    {
        std::string up = v;
        for (char &c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        return up;
    }
    if (key == "direction")
    {
        if (v == "h2d") return "host to device";
        if (v == "d2h") return "device to host";
        if (v == "roundtrip") return "round trip";
        return v;
    }
    if (key == "memory") return v + " memory";
    if (key == "context") return v + " tokens";
    if (key == "in_flight") return v + " in flight";
    if (key == "threads_per_core") return v + " threads per core";
    if (key == "contention") return v == "all" ? "contended" : v;
    if (key == "working_set") return bytesWords(v);
    if (key == "op" || key == "phase" || key == "layer" || key == "pattern" ||
        key == "pixel_format" || key == "workload")
        return underscoresAsSpaces(v);
    return underscoresAsSpaces(key) + " " + v;
}

} // namespace

// ── Measure keys ─────────────────────────────────────────────────────────────

std::string dimValueText(const DimValue &v)
{
    if (!v.isNumber) return v.text;
    if (std::isfinite(v.number) && std::floor(v.number) == v.number)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.0f", v.number);
        return buf;
    }
    return shortestDouble(v.number);
}

bool isIdentityDim(const std::string &key, const std::map<std::string, std::string> &declared)
{
    Dim d;
    if (fromString(key, d)) return dimInfo(d).cls == DimClass::Identity;
    const auto it = declared.find(key);
    return it == declared.end() || it->second != "variant";
}

IdentityDims identityDims(const Dims &dims, const std::string &family,
                          const std::map<std::string, std::string> &declared)
{
    const DimValue *dtypeValue = dims.find("dtype");
    const std::string dtype = dtypeValue ? dimValueText(*dtypeValue) : "";
    IdentityDims out;
    for (const DimEntry &e : dims)
    {
        if (!isIdentityDim(e.key, declared)) continue;
        const std::string text = dimValueText(e.value);
        std::string def;
        if (defaultOf(e.key, family, dtype, def) && def == text) continue;
        out.emplace_back(e.key, text);
    }
    std::sort(out.begin(), out.end(),
              [](const std::pair<std::string, std::string> &a,
                 const std::pair<std::string, std::string> &b) { return a.first < b.first; });
    return out;
}

MeasureKey::MeasureKey(std::string family_, std::string quantity_, IdentityDims identity_)
    : family(std::move(family_)), quantity(std::move(quantity_)), identity(std::move(identity_))
{
    id = family + "|" + quantity;
    for (const auto &kv : identity) id += "|" + kv.first + "=" + kv.second;
}

const std::string *MeasureKey::find(const std::string &key) const
{
    for (const auto &kv : identity)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

std::string readingQuantity(const TestResult &test, const MetricResult &metric)
{
    return metric.hasUnit ? quantityText(metric.quantity, metric.quantityWord)
                          : quantityText(test.quantity, test.quantityWord);
}

UnitInfo MeasuredReading::units() const
{
    UnitInfo u;
    u.symbol    = test->unitOf(*metric);
    u.quantity  = metric->hasUnit ? metric->quantity : test->quantity;
    u.direction = direction;
    return u;
}

bool measureReading(const DeviceResult &device, const TestResult &test,
                    const MetricResult &metric,
                    const std::map<std::string, std::string> &declared, MeasuredReading &out)
{
    if (metric.status != ResultStatus::Ok || !std::isfinite(metric.value)) return false;
    const std::string &family = !metric.benchmark.empty() ? metric.benchmark : test.benchmark;
    if (family.empty()) return false;
    out.device = &device;
    out.test   = &test;
    out.metric = &metric;
    out.dims   = effectiveDims(test.dims, metric.dims);
    if (const DimValue *dtype = out.dims.find("dtype"))
        out.dtype = dimValueText(*dtype);
    else if (const FamilyInfo *fam = familyOf(family); fam && fam->hasDtype)
        out.dtype = toString(fam->dtype);
    else
        out.dtype.clear();
    out.key       = MeasureKey(family, readingQuantity(test, metric),
                               identityDims(out.dims, family, declared));
    out.direction = test.directionOf(metric);
    return true;
}

int readingOrder(const MeasuredReading &a, const MeasuredReading &b)
{
    int c = a.backend().compare(b.backend());
    if (c != 0) return c < 0 ? -1 : 1;
    if (a.device->deviceIndex != b.device->deviceIndex)
        return a.device->deviceIndex < b.device->deviceIndex ? -1 : 1;
    c = a.test->key().compare(b.test->key());
    if (c != 0) return c < 0 ? -1 : 1;
    c = a.metric->id.compare(b.metric->id);
    return c == 0 ? 0 : (c < 0 ? -1 : 1);
}

bool isBetter(const MeasuredReading &a, const MeasuredReading &b)
{
    const int order = readingOrder(a, b);
    if (a.value() != b.value())
    {
        const Direction direction = order <= 0 ? a.direction : b.direction;
        return direction == Direction::LowerIsBetter ? a.value() < b.value()
                                                     : a.value() > b.value();
    }
    return order < 0;
}

MeasureBest::MeasureBest(const MeasuredReading &first) : key(first.key), best(first)
{
    byBackend.push_back(first);
}

void MeasureBest::add(const MeasuredReading &r)
{
    if (isBetter(r, best)) best = r;
    for (MeasuredReading &mine : byBackend)
        if (mine.backend() == r.backend())
        {
            if (isBetter(r, mine)) mine = r;
            return;
        }
    byBackend.push_back(r);
}

const MeasuredReading *MeasureBest::ofBackend(const std::string &backend) const
{
    for (const MeasuredReading &r : byBackend)
        if (r.backend() == backend) return &r;
    return nullptr;
}

namespace {

std::vector<MeasuredReading> bestFirst(std::vector<MeasuredReading> v)
{
    std::sort(v.begin(), v.end(),
              [](const MeasuredReading &a, const MeasuredReading &b) { return isBetter(a, b); });
    return v;
}

} // namespace

std::vector<MeasuredReading> MeasureBest::ranked() const { return bestFirst(byBackend); }
std::vector<MeasuredReading> HeadlineValue::ranked() const { return bestFirst(byBackend); }

// ── The catalog ──────────────────────────────────────────────────────────────

bool Headline::appliesTo(const std::string &processorKind) const
{
    return anyProcessor ||
           std::find(processors.begin(), processors.end(), processorKind) != processors.end();
}

bool Headline::selects(const MeasureKey &key) const
{
    if (key.quantity != quantity) return false;
    if (!anyFamily && std::find(families.begin(), families.end(), key.family) == families.end())
        return false;
    if (anyDims) return true;
    for (const Dims &d : dims)
        if (identityDims(d, key.family) == key.identity) return true;
    return false;
}

HeadlineCatalog HeadlineCatalog::fromJson(const std::string &text)
{
    HeadlineCatalog out;
    const JsonValue root = jsonParse(text);
    if (!root.isObject() || root.str("schema") != kSchema) return out;
    out.version = static_cast<int>(root.num("catalog_version", 0));
    if (const JsonValue *hs = root.find("headlines"))
        for (const JsonValue &h : hs->items())
        {
            if (!h.isObject()) continue;
            Headline line;
            line.id       = h.str("id");
            line.title    = h.has("title") ? h.str("title") : line.id;
            line.quantity = h.str("quantity");
            if (const JsonValue *p = h.find("processors"); p && p->isArray())
            {
                line.anyProcessor = false;
                line.processors   = stringsOf(*p);
            }
            if (const JsonValue *f = h.find("families"); f && f->isArray())
            {
                line.anyFamily = false;
                line.families  = stringsOf(*f);
            }
            if (const JsonValue *d = h.find("dims"); d && d->isArray())
            {
                line.anyDims = false;
                for (const JsonValue &x : d->items())
                    if (x.isObject()) line.dims.push_back(dimsFromJson(x));
            }
            out.headlines.push_back(std::move(line));
        }
    return out;
}

const HeadlineCatalog &HeadlineCatalog::bundled()
{
    static const HeadlineCatalog catalog = fromJson(
        std::string(reinterpret_cast<const char *>(kBundledCatalogJson), kBundledCatalogJsonSize));
    return catalog;
}

// ── Per processor ────────────────────────────────────────────────────────────

std::string ProcessorMeasures::kind() const
{
    if (processor) return trimmed(processor->kind);
    return devices.empty() ? "" : deviceTypeString(devices.front()->type);
}

std::string ProcessorMeasures::name() const
{
    if (processor) return trimmed(processor->name);
    return devices.empty() ? "" : devices.front()->name;
}

const MeasureBest *ProcessorMeasures::find(const std::string &keyId) const
{
    const auto it = index_.find(keyId);
    return it == index_.end() ? nullptr : &byKey[it->second];
}

void ProcessorMeasures::add(const MeasuredReading &r)
{
    const auto it = index_.find(r.key.id);
    if (it == index_.end())
    {
        index_[r.key.id] = byKey.size();
        byKey.emplace_back(r);
    }
    else
    {
        byKey[it->second].add(r);
    }
}

bool ProcessorMeasures::headline(const Headline &h, HeadlineValue &out) const
{
    if (!h.appliesTo(kind())) return false;
    bool found = false;
    HeadlineValue v;
    v.headline = &h;
    for (const MeasureBest &m : byKey)
    {
        if (!h.selects(m.key)) continue;
        if (!found || isBetter(m.best, v.best)) v.best = m.best;
        found = true;
        for (const MeasuredReading &r : m.byBackend)
        {
            bool had = false;
            for (MeasuredReading &mine : v.byBackend)
                if (mine.backend() == r.backend())
                {
                    if (isBetter(r, mine)) mine = r;
                    had = true;
                    break;
                }
            if (!had) v.byBackend.push_back(r);
        }
    }
    if (found) out = std::move(v);
    return found;
}

RunMeasures RunMeasures::of(const RunDocument &doc)
{
    RunMeasures out;
    std::map<std::string, size_t> byProcessor;
    for (const ProcessorInfo &p : doc.processors)
    {
        const std::string id = trimmed(p.id);
        if (byProcessor.count(id)) continue;
        byProcessor[id] = out.groups.size();
        ProcessorMeasures g;
        g.id        = id;
        g.processor = &p;
        out.groups.push_back(std::move(g));
    }
    for (const DeviceResult &d : doc.devices)
    {
        size_t group;
        const auto it = byProcessor.find(d.processor);
        if (it != byProcessor.end())
        {
            group = it->second;
        }
        else
        {
            // Not linked, or linked to a processor the file does not list.
            ProcessorMeasures g;
            g.id = d.backend + "|" + d.platform + "|" + d.name + "|#" + std::to_string(d.deviceIndex);
            group = out.groups.size();
            out.groups.push_back(std::move(g));
        }
        ProcessorMeasures &g = out.groups[group];
        g.devices.push_back(&d);
        for (const TestResult *t : testsInCategoryOrder(d))
            for (const MetricResult &m : t->metrics)
            {
                MeasuredReading r;
                if (measureReading(d, *t, m, doc.meta.dimKeyClasses, r)) g.add(r);
            }
    }
    return out;
}

const ProcessorMeasures *RunMeasures::forProcessor(const std::string &id) const
{
    for (const ProcessorMeasures &g : groups)
        if (g.linked() && g.id == id) return &g;
    return nullptr;
}

// measures.dart's DeviceRun.tests: a device's tests grouped by category, in
// the order each category first appears, each group in file order.
std::vector<const TestResult *> testsInCategoryOrder(const DeviceResult &d)
{
    std::vector<Category> order;
    for (const TestResult &t : d.tests)
        if (std::find(order.begin(), order.end(), t.category) == order.end())
            order.push_back(t.category);
    std::vector<const TestResult *> out;
    for (Category c : order)
        for (const TestResult &t : d.tests)
            if (t.category == c) out.push_back(&t);
    return out;
}

bool asSaved(const RunDocument &doc, RunDocument &out, std::string &error)
{
    RunDocument lean;
    lean.meta       = doc.meta;
    lean.packages   = doc.packages;
    lean.processors = doc.processors;
    lean.devices    = doc.devices;
    bool newer = false;
    return parseRunJson(runDocumentToJson(lean), "this run", out, error, newer);
}

// ── Words ────────────────────────────────────────────────────────────────────

std::string dataTypeLabel(const std::string &dtype)
{
    std::string out = dtype;
    for (char &c : out)
        c = c == '_' ? ' ' : static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

std::vector<std::string> describeReading(const MeasuredReading &r)
{
    std::vector<std::string> parts;
    if (!r.dtype.empty()) parts.push_back(dataTypeLabel(r.dtype));
    static const char *const lead[] = {"weights", "kv", "acc", "quant", "sparsity", "scope"};
    std::vector<std::string> keys;
    for (const char *k : lead)
        if (r.key.find(k)) keys.push_back(k);
    for (const auto &kv : r.key.identity)
        if (std::find(std::begin(lead), std::end(lead), kv.first) == std::end(lead))
            keys.push_back(kv.first);
    for (const std::string &k : keys)
    {
        const std::string word = dimWords(k, *r.key.find(k));
        if (!word.empty()) parts.push_back(word);
    }
    const FamilyInfo *fam = familyOf(r.key.family);
    if (r.dims.empty() && !(fam && fam->hasDtype)) return {r.metric->displayLabel()};
    return parts;
}

} // namespace xpupeak
