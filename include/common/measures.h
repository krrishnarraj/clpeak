#ifndef XPUPEAK_MEASURES_H
#define XPUPEAK_MEASURES_H

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <common/run_document.h>
#include <common/units.h>

// ── The measure engine ─────────────────────────────────────────────────────
//
// What makes two readings comparable, and each processor's best of every
// measure, over all its backends and per backend, with the reading it came
// from (docs/format-v4.md, Comparing runs):
//
//   for each ok reading r on a device linked to processor P:
//       k = measureKey(r)                  # family | quantity | identity dims
//       best[P][k]            = better(best[P][k], r)       # by direction
//       best[P][k][r.backend] = better(best[P][k][r.backend], r)
//   headline h = better of best[P][k] over the keys h selects
//
// Peaks are derived, never stored: a document holds readings, and this is
// the one algorithm that turns them into views.  The app runs the same
// algorithm in Dart (app/lib/src/model/measures.dart and headlines.dart),
// and the two answer alike -- every key, every best, every tie, every
// headline -- over a document as a file holds it.
//
// **Key a document as a file holds it.**  A result on disk is canonical: no
// dim at its default, `dim_keys` declaring each key's class, every value
// rounded as the writer writes it.  A run in memory is not -- its dims are as
// the backends authored them and its values are the floats they measured --
// so a host that summarises its own run keys the document as it reads back
// (asSaved()), which is the file the app would open.
//
// Everything here holds pointers into the document it was built from, which
// must outlive it.

namespace xpupeak {

// ---- Measure keys -------------------------------------------------------------

// A dim's value as a measure key spells it: a string as written, a whole
// number in decimal (`2048`, also where the file says `2048.0`).
std::string dimValueText(const DimValue &v);

// Identity dims, sorted by key, each as its text.
using IdentityDims = std::vector<std::pair<std::string, std::string>>;

// Whether two readings must agree on `key` to be comparable: the
// vocabulary's class, or for a key this build does not know the class its
// file declares in `dim_keys` (`declared`), identity where that is silent.
bool isIdentityDim(const std::string &key, const std::map<std::string, std::string> &declared);

// The identity dims of `dims` that differ from their defaults, for a reading
// of `family`, sorted by key: what a measure key carries.  `dims` are a
// reading's effective dims, or a headline's dims object, which is normalised
// the same way to be compared.
IdentityDims identityDims(const Dims &dims, const std::string &family,
                          const std::map<std::string, std::string> &declared = {});

// What a reading measured, in words every backend shares: its family, its
// quantity, and its identity dims away from their defaults.
struct MeasureKey {
    std::string  family;
    std::string  quantity;
    IdentityDims identity;    // sorted by key
    // `matrix-compute|flops|dtype=nvfp4|sparsity=2:4`
    std::string  id;

    MeasureKey() = default;
    MeasureKey(std::string family, std::string quantity, IdentityDims identity);

    bool operator==(const MeasureKey &o) const { return id == o.id; }
    bool operator!=(const MeasureKey &o) const { return id != o.id; }
    // The value of one identity dim, or nullptr.
    const std::string *find(const std::string &key) const;
};

// The quantity a reading counts, as its file spells it: its own unit's, else
// its test's.
std::string readingQuantity(const TestResult &test, const MetricResult &metric);

// One ok reading, where the engine found it.
struct MeasuredReading {
    const DeviceResult *device = nullptr;
    const TestResult   *test   = nullptr;
    const MetricResult *metric = nullptr;
    MeasureKey key;
    Dims       dims;          // effective: its test's, with its own over them
    // The data type it computed on -- its dims', else its family's -- or
    // empty where neither says.
    std::string dtype;
    Direction  direction = Direction::HigherIsBetter;

    double             value() const { return metric->value; }
    const std::string &backend() const { return device->backend; }
    int                revision() const { return test->revision; }
    // The unit to print it in: its own, else its test's.
    UnitInfo           units() const;
};

// The key and the reading `metric` measured on `device`, or false for one
// that has none: a reading that did not succeed, has no family, or no finite
// value.  `declared` is the file's `dim_keys`.
bool measureReading(const DeviceResult &device, const TestResult &test,
                    const MetricResult &metric,
                    const std::map<std::string, std::string> &declared, MeasuredReading &out);

// A fixed order over the readings on one processor -- by backend, the
// backend's device index, test key, then reading id, each by code unit --
// which settles what value cannot, so a best never depends on how the
// readings were walked.
int readingOrder(const MeasuredReading &a, const MeasuredReading &b);

// Whether `a` is a better reading of one measure than `b`: by direction --
// that of whichever comes first in readingOrder() -- and on a tie in value,
// whichever comes first.
bool isBetter(const MeasuredReading &a, const MeasuredReading &b);

// One measure on one processor: its best reading over every backend, and
// each backend's best, in the order the backends first measured it.
struct MeasureBest {
    MeasureKey                   key;
    MeasuredReading              best;
    std::vector<MeasuredReading> byBackend;

    explicit MeasureBest(const MeasuredReading &first);
    void add(const MeasuredReading &r);
    const MeasuredReading *ofBackend(const std::string &backend) const;
    // The backends' bests, best first.
    std::vector<MeasuredReading> ranked() const;
};

// ---- The headline catalog -------------------------------------------------------

// One headline: the measure keys it selects, of which its value on a
// processor is the best there (docs/format-v4.md, The headline catalog).
struct Headline {
    std::string id;
    std::string title;
    std::string quantity;                 // `flops`, `ops`, `bytes_per_second`
    bool anyProcessor = true;             // `processors` absent
    std::vector<std::string> processors;  // the processor kinds it applies to
    bool anyFamily = true;                // `families` absent
    std::vector<std::string> families;
    bool anyDims = true;                  // `dims` absent
    std::vector<Dims> dims;               // identity-dim objects, defaults filled in

    bool appliesTo(const std::string &processorKind) const;
    // Whether `key` is one of its measures.  A dims object is normalised as a
    // reading's dims are -- variant keys dropped, defaults dropped for the
    // key's family -- so `{"dtype": "fp16"}` and `{"dtype": "fp16", "acc":
    // "fp32"}` select the same matrix keys.
    bool selects(const MeasureKey &key) const;
};

struct HeadlineCatalog {
    static constexpr const char *kSchema = "xpupeak/measures";

    int version = 0;                      // `catalog_version`: its own, not the format's
    std::vector<Headline> headlines;

    // From the catalog's JSON text.  A document of another schema, or text
    // that is not JSON, reads as no headlines rather than as wrong ones.
    static HeadlineCatalog fromJson(const std::string &text);

    // The catalog this build embeds: app/assets/measures.json, the app's
    // own, as it was when the build was configured.
    static const HeadlineCatalog &bundled();
};

// A headline's value on one processor: the best reading of the keys it
// selects, and each backend's best of them.
struct HeadlineValue {
    const Headline              *headline = nullptr;
    MeasuredReading              best;
    std::vector<MeasuredReading> byBackend;   // first-measured order

    // The backends' bests, best first.
    std::vector<MeasuredReading> ranked() const;
};

// ---- Per processor ----------------------------------------------------------------

// Everything measured on one processor -- or, in a file whose devices are
// not linked, on one device, which is then all the engine can say is one
// piece of hardware.
struct ProcessorMeasures {
    // The processor's id, or for an unlinked device its key
    // (`backend|platform|name|#index`, as the app keys it).
    std::string id;
    const ProcessorInfo *processor = nullptr;     // nullptr: an unlinked device
    std::vector<const DeviceResult *> devices;    // run order
    std::vector<MeasureBest> byKey;               // first-measured order

    bool linked() const { return processor != nullptr; }
    // `cpu`, `gpu`, `npu`, `accelerator`: the processor's kind, or an
    // unlinked device's type.
    std::string kind() const;
    std::string name() const;

    const MeasureBest *find(const std::string &keyId) const;

    // `h`'s value here; false when it does not apply to this kind of
    // processor or nothing here measured a key it selects.
    bool headline(const Headline &h, HeadlineValue &out) const;

    void add(const MeasuredReading &r);

private:
    std::map<std::string, size_t> index_;
};

// A document's measures, per processor.
struct RunMeasures {
    // The document's processors in its order, each with the devices that
    // name it -- a processor nothing measured included -- then each device
    // linked to none, on its own.
    std::vector<ProcessorMeasures> groups;

    static RunMeasures of(const RunDocument &doc);
    const ProcessorMeasures *forProcessor(const std::string &id) const;
};

// A device's tests in the order the app walks them (DeviceRun.tests): by
// category, in the order each first appears, each category's in file order.
// The order keys are first measured in, and a compare's reading-by-reading
// rows are listed in.
std::vector<const TestResult *> testsInCategoryOrder(const DeviceResult &device);

// A run's document as its file holds it: written as saveRunJson() writes it
// and read back, so its dims are canonical and its values rounded as the
// file's are -- what the app keys when it opens that file.  Without its log
// and inventory, which nothing here reads.  False, with `error`, only if the
// round trip fails.
bool asSaved(const RunDocument &doc, RunDocument &out, std::string &error);

// ---- Words --------------------------------------------------------------------------

// A data type as a person reads it: `FP16`, `FP8 E4M3`, `NVFP4`, `INT8`.
std::string dataTypeLabel(const std::string &dtype);

// What a reading measured, in a person's words: its data type, then what
// sets it apart from its family's usual reading -- `[NVFP4, 2:4 structured
// sparsity]`, `[FP16, FP16 accumulate]`, `[triad]`.  A reading with no
// coordinates at all, from a backend that does not author them, is named by
// its own label instead.
std::vector<std::string> describeReading(const MeasuredReading &r);

} // namespace xpupeak

#endif // XPUPEAK_MEASURES_H
