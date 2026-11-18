#include <common/run_compare.h>

#include <common/options.h>      // benchmarkFromFlag
#include <common/vocabulary.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <mutex>
#include <sstream>

// The compare of docs/format-v4.md, Two runs side by side.  The app's
// (app/lib/src/model/compare.dart) is the other half of a pair: each piece
// below names the one of its it mirrors, and a rule changed in one is
// changed in the other in the same commit.

namespace xpupeak {

namespace {

std::string trimmed(const std::string &s)
{
    const char *ws = " \t\r\n";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

int sign(int c) { return c < 0 ? -1 : (c > 0 ? 1 : 0); }

// compare.dart's _valueOrder: the measure engine's tie order over two runs'
// values -- backend, device index, test key, reading id.
int valueOrder(const CompareValue &a, const CompareValue &b)
{
    int c = a.backend().compare(b.backend());
    if (c != 0) return sign(c);
    if (a.device->deviceIndex != b.device->deviceIndex)
        return a.device->deviceIndex < b.device->deviceIndex ? -1 : 1;
    c = a.test->key().compare(b.test->key());
    if (c != 0) return sign(c);
    return sign(a.metric->id.compare(b.metric->id));
}

// Readings of one backend's test -- one backend, one test id, whatever the
// variant -- at two revisions (docs/format-v4.md, Revisions).
bool methodChanged(const CompareValue &a, const CompareValue &b)
{
    return a.backend() == b.backend() && a.test->id == b.test->id &&
           a.revision() != b.revision();
}

bool measured(const MetricResult &m)
{
    return m.status == ResultStatus::Ok && std::isfinite(m.value);
}

const MetricResult *metricOf(const TestResult &t, const std::string &id)
{
    for (const MetricResult &m : t.metrics)
        if (m.id == id) return &m;
    return nullptr;
}

// DeviceRun.findTest: the first of the device's tests, in the app's walk
// order, with that key.
const TestResult *findTest(const DeviceResult &d, const std::string &key)
{
    for (const TestResult *t : testsInCategoryOrder(d))
        if (t->key() == key) return t;
    return nullptr;
}

// A device's name across two runs: backend, name, occurrence.
std::string deviceName(const CompareRun &run, const DeviceResult &d)
{
    return d.backend + '\0' + d.name + '\0' + std::to_string(run.occurrenceOf(d));
}

int rowOrder(Category ca, const MeasureKey &ka, Category cb, const MeasureKey &kb)
{
    if (ca != cb) return static_cast<int>(ca) < static_cast<int>(cb) ? -1 : 1;
    return measureKeyOrder(ka, kb);
}

bool parseInteger(const std::string &s, long long &out)
{
    if (s.empty()) return false;
    size_t i = (s[0] == '-' || s[0] == '+') ? 1 : 0;
    if (i == s.size()) return false;
    for (size_t k = i; k < s.size(); k++)
        if (!std::isdigit(static_cast<unsigned char>(s[k]))) return false;
    errno = 0;
    char *end = nullptr;
    out = std::strtoll(s.c_str(), &end, 10);
    return errno == 0 && end && *end == '\0';
}

// compare.dart's _dimValueOrder: a data type's or a word's place in its
// list (one not in it after those that are, by text), else by number, else
// by text.
int dimValueOrder(const std::string &key, const std::string &a, const std::string &b)
{
    auto inList = [&](int ia, int ib) {
        if (ia >= 0 && ib >= 0) return ia < ib ? -1 : (ia > ib ? 1 : 0);
        if (ia >= 0) return -1;
        if (ib >= 0) return 1;
        return sign(a.compare(b));
    };
    Dim d;
    if (fromString(key, d))
    {
        const DimKeyInfo &info = dimInfo(d);
        if (info.kind == DimKind::DataType)
        {
            DType ta, tb;
            return inList(fromString(a, ta) ? static_cast<int>(ta) : -1,
                          fromString(b, tb) ? static_cast<int>(tb) : -1);
        }
        if (info.kind == DimKind::Word)
        {
            int ia = -1, ib = -1;
            for (unsigned i = 0; i < dimWordCount(d); i++)
            {
                if (a == dimWord(d, i)) ia = static_cast<int>(i);
                if (b == dimWord(d, i)) ib = static_cast<int>(i);
            }
            return inList(ia, ib);
        }
    }
    long long na = 0, nb = 0;
    if (parseInteger(a, na) && parseInteger(b, nb)) return na < nb ? -1 : (na > nb ? 1 : 0);
    return sign(a.compare(b));
}

const std::string *identityValue(const IdentityDims &dims, const std::string &key)
{
    for (const auto &kv : dims)
        if (kv.first == key) return &kv.second;
    return nullptr;
}

int identityOrder(const IdentityDims &a, const IdentityDims &b)
{
    std::vector<std::string> keys;
    for (const auto &kv : a) keys.push_back(kv.first);
    for (const auto &kv : b)
        if (!identityValue(a, kv.first)) keys.push_back(kv.first);
    std::sort(keys.begin(), keys.end(), [](const std::string &x, const std::string &y) {
        const int dx = x == "dtype" ? 0 : 1, dy = y == "dtype" ? 0 : 1;
        return dx != dy ? dx < dy : x < y;
    });
    for (const std::string &k : keys)
    {
        const std::string *va = identityValue(a, k);
        const std::string *vb = identityValue(b, k);
        if (va && vb && *va == *vb) continue;
        if (!va) return -1;
        if (!vb) return 1;
        const int c = dimValueOrder(k, *va, *vb);
        if (c != 0) return c;
    }
    return 0;
}

int familyRank(const std::string &family)
{
    Benchmark b;
    return familyFromString(family, b) ? static_cast<int>(b) : static_cast<int>(Benchmark::COUNT);
}

// compare.dart's _quantityRanks: the quantities in the vocabulary's order.
int quantityRank(const std::string &q)
{
    static const char *const order[] = {"flops", "ops", "bytes_per_second", "seconds",
                                        "items_per_second", "ratio", "count"};
    for (int i = 0; i < 7; i++)
        if (q == order[i]) return i;
    return 7;
}

std::vector<CompareValue> bestFirst(const CompareRun &run, std::vector<MeasuredReading> readings)
{
    std::sort(readings.begin(), readings.end(),
              [](const MeasuredReading &x, const MeasuredReading &y) { return isBetter(x, y); });
    std::vector<CompareValue> out;
    for (const MeasuredReading &r : readings) out.push_back(CompareValue::of(run, r));
    return out;
}

DeviceCompare compareDevices(const CompareRun &runA, const DeviceResult &a,
                             const CompareRun &runB, const DeviceResult &b)
{
    DeviceCompare out;
    out.a = &a;
    out.b = &b;
    out.occurrence = runA.occurrenceOf(a);
    for (const TestResult *ta : testsInCategoryOrder(a))
    {
        const TestResult *tb = findTest(b, ta->key());
        for (const MetricResult &ma : ta->metrics)
        {
            if (!measured(ma)) continue;
            CompareValue va = CompareValue::reading(runA, a, *ta, ma);
            const MetricResult *mb = tb ? metricOf(*tb, ma.id) : nullptr;
            if (mb && measured(*mb))
            {
                CompareValue vb = CompareValue::reading(runB, b, *tb, *mb);
                ReadingCompare r;
                r.verdict = judge(va, vb);
                r.a = std::move(va);
                r.b = std::move(vb);
                out.readings.push_back(std::move(r));
            }
            else
            {
                out.onlyA.push_back(std::move(va));
            }
        }
    }
    for (const TestResult *tb : testsInCategoryOrder(b))
    {
        const TestResult *ta = findTest(a, tb->key());
        for (const MetricResult &mb : tb->metrics)
        {
            if (!measured(mb)) continue;
            const MetricResult *ma = ta ? metricOf(*ta, mb.id) : nullptr;
            if (!ma || !measured(*ma)) out.onlyB.push_back(CompareValue::reading(runB, b, *tb, mb));
        }
    }
    return out;
}

} // namespace

const char *toString(PairedBy by)
{
    switch (by)
    {
    case PairedBy::Model:  return "model";
    case PairedBy::Kind:   return "kind";
    case PairedBy::Person: return "person";
    }
    return "person";
}

// ── The two sides ────────────────────────────────────────────────────────────

CompareRun::CompareRun(const RunDocument &doc) : document(doc), measures(RunMeasures::of(doc))
{
    std::map<std::string, int> seen;
    for (const DeviceResult &d : doc.devices)
    {
        const std::string name = d.backend + '\0' + d.name;
        occurrence_[&d] = seen[name]++;
    }
}

int CompareRun::occurrenceOf(const DeviceResult &device) const
{
    const auto it = occurrence_.find(&device);
    return it == occurrence_.end() ? 0 : it->second;
}

CompareValue CompareValue::of(const CompareRun &run, const MeasuredReading &r)
{
    CompareValue v;
    v.run      = &run;
    v.device   = r.device;
    v.test     = r.test;
    v.metric   = r.metric;
    v.keyed    = true;
    v.measured = r;
    return v;
}

CompareValue CompareValue::reading(const CompareRun &run, const DeviceResult &device,
                                   const TestResult &test, const MetricResult &metric)
{
    CompareValue v;
    v.run    = &run;
    v.device = &device;
    v.test   = &test;
    v.metric = &metric;
    v.keyed  = measureReading(device, test, metric, run.document.meta.dimKeyClasses, v.measured);
    return v;
}

UnitInfo CompareValue::units() const
{
    UnitInfo u;
    u.symbol    = test->unitOf(*metric);
    u.quantity  = metric->hasUnit ? metric->quantity : test->quantity;
    u.direction = direction();
    return u;
}

std::vector<std::string> CompareValue::label() const
{
    return keyed ? describeReading(measured) : std::vector<std::string>{metric->displayLabel()};
}

// ── What two values say ──────────────────────────────────────────────────────

Verdict Verdict::of(double a, double b, Direction direction, bool methodChanged)
{
    Verdict v;
    if (methodChanged)
    {
        v.methodChanged = true;
        return v;
    }
    if (a == b)
    {
        v.hasRatio = v.hasFactor = true;
        v.ratio = v.factor = 1.0;
        return v;
    }
    const bool higher = direction != Direction::LowerIsBetter;
    v.hasWinner = true;
    v.winner    = (higher ? a > b : a < b) ? CompareSide::A : CompareSide::B;
    if (a <= 0 || b <= 0) return v;
    v.hasRatio  = v.hasFactor = true;
    v.ratio     = higher ? a / b : b / a;
    v.factor    = a > b ? a / b : b / a;
    return v;
}

Verdict judge(const CompareValue &a, const CompareValue &b)
{
    const Direction direction = a.direction() == b.direction() || valueOrder(a, b) <= 0
                                    ? a.direction()
                                    : b.direction();
    return Verdict::of(a.value(), b.value(), direction, methodChanged(a, b));
}

// ── Pairing ──────────────────────────────────────────────────────────────────

std::string processorModel(const CompareRun &run, const ProcessorMeasures &g)
{
    auto join = [](const std::vector<std::string> &parts) {
        std::string out;
        for (size_t i = 0; i < parts.size(); i++) out += (i ? "|" : "") + parts[i];
        return out;
    };
    const ProcessorInfo *p = g.processor;
    if (!p)
    {
        if (g.devices.empty()) return g.kind();
        const DeviceResult &d = *g.devices.front();
        const std::string vid = trimmed(d.facts.vendorId), did = trimmed(d.facts.deviceId);
        return join({g.kind(), !vid.empty() && !did.empty() ? "pci:" + vid + ":" + did : d.name});
    }
    const HardwareFacts &f = p->facts;
    std::string package;
    for (const PackageInfo &pkg : run.document.packages)
        if (trimmed(pkg.id) == trimmed(p->package))
        {
            package = trimmed(pkg.name);
            break;
        }
    const std::string kind = trimmed(p->kind), vendor = trimmed(p->vendor), name = trimmed(p->name);
    if (kind == "cpu")
    {
        const CpuFacts &c = f.cpu;
        std::string cores;
        if (!c.clusters.empty())
            for (size_t i = 0; i < c.clusters.size(); i++)
                cores += (i ? "+" : "") + std::to_string(c.clusters[i].cores);
        else if (c.cores > 0)
            cores = std::to_string(c.cores);
        std::vector<std::string> parts = {"cpu", vendor, name, cores};
        if (c.cpuidFamily >= 0)
            parts.push_back("cpuid:" + std::to_string(c.cpuidFamily) + ":" + std::to_string(c.cpuidModel));
        std::string midrs;
        for (const CpuCluster &k : c.clusters)
        {
            const std::string m = trimmed(k.midr);
            if (!m.empty()) midrs += (midrs.empty() ? "" : "+") + m;
        }
        if (!midrs.empty()) parts.push_back("midr:" + midrs);
        return join(parts);
    }
    if (kind == "gpu")
    {
        if (f.integrated == 1)
            return join({"gpu", "integrated", !package.empty() ? package : name,
                         std::to_string(f.computeUnits)});
        const std::string vid = trimmed(f.vendorId), did = trimmed(f.deviceId);
        if (!vid.empty() && !did.empty()) return "gpu|pci:" + vid + ":" + did;
        return join({"gpu", vendor, name});
    }
    return join({kind, vendor, name, package});
}

std::vector<ProcessorPair> pairProcessors(const CompareRun &a, const CompareRun &b)
{
    std::vector<const ProcessorMeasures *> free;
    std::map<const ProcessorMeasures *, std::string> modelOf;
    for (const ProcessorMeasures &g : b.measures.groups)
    {
        free.push_back(&g);
        modelOf[&g] = processorModel(b, g);
    }
    std::map<const ProcessorMeasures *, ProcessorPair> paired;
    for (const ProcessorMeasures &ga : a.measures.groups)
    {
        const std::string model = processorModel(a, ga);
        const std::string kind  = ga.kind();
        const auto it = std::find_if(free.begin(), free.end(), [&](const ProcessorMeasures *gb) {
            return gb->kind() == kind && modelOf[gb] == model;
        });
        if (it != free.end())
        {
            paired[&ga] = {&ga, *it, PairedBy::Model};
            free.erase(it);
        }
    }
    for (const ProcessorMeasures &ga : a.measures.groups)
    {
        if (paired.count(&ga)) continue;
        const std::string kind = ga.kind();
        const auto it = std::find_if(free.begin(), free.end(),
                                     [&](const ProcessorMeasures *gb) { return gb->kind() == kind; });
        if (it != free.end())
        {
            paired[&ga] = {&ga, *it, PairedBy::Kind};
            free.erase(it);
        }
    }
    std::vector<ProcessorPair> out;
    for (const ProcessorMeasures &ga : a.measures.groups)
        if (const auto it = paired.find(&ga); it != paired.end()) out.push_back(it->second);
    return out;
}

// ── Order ────────────────────────────────────────────────────────────────────

int measureKeyOrder(const MeasureKey &a, const MeasureKey &b)
{
    int ra = familyRank(a.family), rb = familyRank(b.family);
    if (ra != rb) return ra < rb ? -1 : 1;
    int c = a.family.compare(b.family);
    if (c != 0) return sign(c);
    ra = quantityRank(a.quantity);
    rb = quantityRank(b.quantity);
    if (ra != rb) return ra < rb ? -1 : 1;
    c = a.quantity.compare(b.quantity);
    if (c != 0) return sign(c);
    c = identityOrder(a.identity, b.identity);
    if (c != 0) return c;
    return sign(a.id.compare(b.id));
}

Category keyCategory(const MeasureKey &key, const std::vector<const CompareValue *> &values)
{
    Benchmark b;
    if (familyFromString(key.family, b)) return categoryOf(b);
    std::vector<const CompareValue *> sorted = values;
    std::sort(sorted.begin(), sorted.end(), [](const CompareValue *x, const CompareValue *y) {
        return valueOrder(*x, *y) < 0;
    });
    for (const CompareValue *v : sorted)
        if (v->test->category != Category::Unknown) return v->test->category;
    return Category::Unknown;
}

// ── Rows ─────────────────────────────────────────────────────────────────────

MeasureRow MeasureRow::of(const CompareRun &run, const MeasureBest &m)
{
    MeasureRow row;
    row.key      = m.key;
    row.best     = CompareValue::of(run, m.best);
    row.backends = bestFirst(run, m.byBackend);
    row.category = keyCategory(m.key, {&row.best});
    return row;
}

std::vector<std::string> KeyCompare::label() const
{
    return valueOrder(a.best, b.best) <= 0 ? a.best.label() : b.best.label();
}

// ── Two processors ───────────────────────────────────────────────────────────

PairCompare PairCompare::of(const CompareRun &runA, const ProcessorMeasures &a,
                            const CompareRun &runB, const ProcessorMeasures &b,
                            const HeadlineCatalog &catalog, PairedBy by)
{
    PairCompare out;
    out.runA     = &runA;
    out.a        = &a;
    out.runB     = &runB;
    out.b        = &b;
    out.pairedBy = by;

    for (const Headline &h : catalog.headlines)
    {
        HeadlineValue ha, hb;
        const bool hasA = a.headline(h, ha);
        const bool hasB = b.headline(h, hb);
        if (!hasA && !hasB) continue;
        HeadlineCompare hc;
        hc.headline = &h;
        hc.hasA = hasA;
        hc.hasB = hasB;
        if (hasA)
        {
            hc.a         = CompareValue::of(runA, ha.best);
            hc.aBackends = bestFirst(runA, ha.byBackend);
        }
        if (hasB)
        {
            hc.b         = CompareValue::of(runB, hb.best);
            hc.bBackends = bestFirst(runB, hb.byBackend);
        }
        if (hasA && hasB)
        {
            hc.hasVerdict = true;
            hc.verdict    = judge(hc.a, hc.b);
        }
        out.headlines.push_back(std::move(hc));
    }

    for (const MeasureBest &m : a.byKey)
    {
        MeasureRow rowA = MeasureRow::of(runA, m);
        const MeasureBest *mb = b.find(m.key.id);
        if (!mb)
        {
            out.onlyA.push_back(std::move(rowA));
            continue;
        }
        KeyCompare k;
        k.key      = m.key;
        k.b        = MeasureRow::of(runB, *mb);
        k.verdict  = judge(rowA.best, k.b.best);
        k.category = keyCategory(m.key, {&rowA.best, &k.b.best});
        k.a        = std::move(rowA);
        out.both.push_back(std::move(k));
    }
    for (const MeasureBest &m : b.byKey)
        if (!a.find(m.key.id)) out.onlyB.push_back(MeasureRow::of(runB, m));
    std::sort(out.both.begin(), out.both.end(), [](const KeyCompare &x, const KeyCompare &y) {
        return rowOrder(x.category, x.key, y.category, y.key) < 0;
    });
    for (std::vector<MeasureRow> *rows : {&out.onlyA, &out.onlyB})
        std::sort(rows->begin(), rows->end(), [](const MeasureRow &x, const MeasureRow &y) {
            return rowOrder(x.category, x.key, y.category, y.key) < 0;
        });

    // One model on both sides: the devices both runs saw on it compare
    // reading by reading too, named by backend, device name and occurrence.
    out.modelA = processorModel(runA, a);
    out.modelB = processorModel(runB, b);
    if (out.modelA == out.modelB)
    {
        std::vector<std::pair<std::string, const DeviceResult *>> rest;
        for (const DeviceResult *d : b.devices) rest.emplace_back(deviceName(runB, *d), d);
        for (const DeviceResult *d : a.devices)
        {
            const std::string name = deviceName(runA, *d);
            const auto it = std::find_if(rest.begin(), rest.end(),
                                         [&](const auto &r) { return r.first == name; });
            if (it == rest.end())
            {
                out.devicesOnlyA.push_back(d);
                continue;
            }
            out.devices.push_back(compareDevices(runA, *d, runB, *it->second));
            rest.erase(it);
        }
        for (const auto &r : rest) out.devicesOnlyB.push_back(r.second);
    }
    return out;
}

// ── Two runs ─────────────────────────────────────────────────────────────────

RunCompare RunCompare::of(const CompareRun &a, const CompareRun &b, const HeadlineCatalog &catalog)
{
    RunCompare out;
    out.a       = &a;
    out.b       = &b;
    out.catalog = &catalog;
    std::vector<const ProcessorMeasures *> usedA, usedB;
    for (const ProcessorPair &p : pairProcessors(a, b))
    {
        out.pairs.push_back(PairCompare::of(a, *p.a, b, *p.b, catalog, p.by));
        usedA.push_back(p.a);
        usedB.push_back(p.b);
    }
    for (const ProcessorMeasures &g : a.measures.groups)
        if (std::find(usedA.begin(), usedA.end(), &g) == usedA.end()) out.unpairedA.push_back(&g);
    for (const ProcessorMeasures &g : b.measures.groups)
        if (std::find(usedB.begin(), usedB.end(), &g) == usedB.end()) out.unpairedB.push_back(&g);
    return out;
}

// ── The baseline a run's rows meet ───────────────────────────────────────────

namespace {

std::string readingName(const std::string &backend, const std::string &device, int occurrence,
                        const std::string &testKey, const std::string &metricId)
{
    return backend + '\0' + device + '\0' + std::to_string(occurrence) + '\0' + testKey + '\0' +
           metricId;
}

} // namespace

ReadingBaseline::ReadingBaseline(const RunDocument &doc)
{
    std::map<std::string, int> seen;
    for (const DeviceResult &d : doc.devices)
    {
        const int occurrence = seen[d.backend + '\0' + d.name]++;
        for (const TestResult *t : testsInCategoryOrder(d))
            for (const MetricResult &m : t->metrics)
            {
                if (!measured(m)) continue;
                Reading r;
                r.value          = m.value;
                r.revision       = t->revision;
                r.direction      = t->directionOf(m);
                r.deviceIndex    = d.deviceIndex;
                r.units.symbol   = t->unitOf(m);
                r.units.quantity = m.hasUnit ? m.quantity : t->quantity;
                // The first of a name, as the app's findTest and _metric take it.
                readings_.emplace(readingName(d.backend, d.name, occurrence, t->key(), m.id), r);
            }
    }
}

const ReadingBaseline::Reading *ReadingBaseline::find(const std::string &backend,
                                                      const std::string &device, int occurrence,
                                                      const std::string &testKey,
                                                      const std::string &metricId) const
{
    const auto it = readings_.find(readingName(backend, device, occurrence, testKey, metricId));
    return it == readings_.end() ? nullptr : &it->second;
}

std::string ReadingBaseline::note(const std::string &backend, const std::string &device,
                                  int occurrence, int deviceIndex, const std::string &testKey,
                                  int revision, const std::string &metricId, double value,
                                  Direction direction, const UnitInfo &units) const
{
    const Reading *base = find(backend, device, occurrence, testKey, metricId);
    if (!base || !std::isfinite(value)) return "";

    // judge(), for a reading and its baseline: one backend, one test, so the
    // tie order comes down to the device index.
    const Direction d = direction == base->direction || deviceIndex <= base->deviceIndex
                            ? direction
                            : base->direction;
    const Verdict v = Verdict::of(value, base->value, d, revision != base->revision);

    // The baseline prints through the row's own scaling: "was 4476" beside a
    // row that says 4.48 TFLOPS reads as a collapse that did not happen.
    const ScaledValue sv = formatScaledValue(base->value, units);
    std::ostringstream out;
    out << "  (was " << sv.text;
    if (!sv.unit.empty()) out << " " << sv.unit;
    if (v.methodChanged)
    {
        // Measured another way: a kernel fix, a counting rule.  No ratio
        // between the two means anything (docs/format-v4.md, Revisions).
        out << ", method changed)";
        return out.str();
    }
    if (!v.hasRatio)
    {
        // A value not above zero: no percentage means anything, only which
        // is better.
        out << ", " << (v.winner == CompareSide::A ? "better" : "worse") << ")";
        return out.str();
    }
    const double delta    = 100.0 * (value - base->value) / base->value;
    const double absDelta = std::fabs(delta);
    out << ", " << (delta >= 0.0 ? '+' : '-') << std::fixed << std::setprecision(1) << absDelta
        << "%";
    // Which way a reading moved is not which way is better: on a latency or
    // numeric-error row, +3% is a regression.  Below this the figure prints
    // as 0.0%, and calling run-to-run noise better or worse would be
    // inventing a result.
    if (absDelta >= 0.05 && v.hasWinner)
        out << (v.winner == CompareSide::A ? " better" : " worse");
    out << ")";
    return out.str();
}

std::shared_ptr<const CompareBaseline> loadCompareBaseline(const std::string &file)
{
    static std::mutex lock;
    static std::map<std::string, std::shared_ptr<const CompareBaseline>> loaded;
    std::lock_guard<std::mutex> guard(lock);
    if (const auto it = loaded.find(file); it != loaded.end()) return it->second;
    auto base = std::make_shared<CompareBaseline>();
    base->file = file;
    std::shared_ptr<const CompareBaseline> out;
    if (loadRunJson(file, base->document))
    {
        base->readings = std::make_unique<ReadingBaseline>(base->document);
        out = base;
    }
    // A file that cannot be read is said once, by the loader, and stays so.
    loaded[file] = out;
    return out;
}

} // namespace xpupeak
