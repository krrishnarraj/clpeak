#ifndef XPUPEAK_RUN_COMPARE_H
#define XPUPEAK_RUN_COMPARE_H

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <common/measures.h>
#include <common/run_document.h>

// ── Two runs side by side ──────────────────────────────────────────────────
//
// What --compare says (docs/format-v4.md, Two runs side by side and
// Revisions), as data: the two runs' processors paired by kind and then by
// model, and in each pair its headlines and the measure keys both have, each
// side's best with the backend that reached it, the ratio turned by
// direction, and "method changed" where the two values are one backend's test
// at two revisions; where the pair is one model, the devices both runs saw on
// it reading by reading, named by backend, device name and that name's
// occurrence, never by index.
//
// The app's compare (app/lib/src/model/compare.dart) is the other half of a
// pair: the same pairs, the same verdicts, the same order.  Both stand on
// the measure engine (include/common/measures.h).  A = the run being made,
// or the one opened first; B = the baseline.
//
// Everything holds pointers into the two documents, which must outlive it,
// and into the CompareRuns, which are built in place and never moved.

namespace xpupeak {

enum class CompareSide { A, B };

// Why two processors are side by side.
enum class PairedBy {
    Model,    // one kind, one model (processorModel)
    Kind,     // one kind, two models: the first of that kind on each side still free
    Person,   // a person paired them, whatever their kinds
};
const char *toString(PairedBy by);

// One side of a compare: a run's document and what the measure engine made
// of it.
struct CompareRun {
    explicit CompareRun(const RunDocument &doc);
    CompareRun(const CompareRun &) = delete;
    CompareRun &operator=(const CompareRun &) = delete;

    const RunDocument &document;
    RunMeasures        measures;

    // `device`'s place among this run's devices of its backend and name,
    // from 0 in file order.  With the backend and the name, it names a
    // device across two runs, where an index would not: a second card shifts
    // every index after it, and MoltenVK lists one GPU twice.
    int occurrenceOf(const DeviceResult &device) const;

private:
    std::map<const DeviceResult *, int> occurrence_;
};

// One side's value in a compare, with where it came from.
struct CompareValue {
    const CompareRun   *run    = nullptr;
    const DeviceResult *device = nullptr;
    const TestResult   *test   = nullptr;
    const MetricResult *metric = nullptr;
    // The reading as the measure engine keyed it; `keyed` false for one
    // with no family, which only reading by reading compares.
    bool            keyed = false;
    MeasuredReading measured;

    static CompareValue of(const CompareRun &run, const MeasuredReading &r);
    static CompareValue reading(const CompareRun &run, const DeviceResult &device,
                                const TestResult &test, const MetricResult &metric);

    double             value() const { return metric->value; }
    const std::string &backend() const { return device->backend; }
    int                revision() const { return test->revision; }
    Direction          direction() const { return test->directionOf(*metric); }
    UnitInfo           units() const;
    // What it measured, in a person's words (describeReading); a reading with
    // no family by its own label.
    std::vector<std::string> label() const;
};

// Two values of one measure, side by side.
struct Verdict {
    // A's value against B's, turned by direction so that above 1 means A is
    // better; exactly 1 for equal values.  None where the method changed,
    // or a value is not above zero.
    bool        hasRatio = false;
    double      ratio    = 0.0;
    // The better side; none for a tie, or where the method changed.
    bool        hasWinner = false;
    CompareSide winner    = CompareSide::A;
    // How many times better the winner is, at least 1; 1 for a tie.  None
    // where there is no ratio.
    bool        hasFactor = false;
    double      factor    = 0.0;
    // Readings of one backend's test at two revisions: not compared.
    bool        methodChanged = false;

    bool isTie() const { return !methodChanged && !hasWinner; }
    static Verdict of(double a, double b, Direction direction, bool methodChanged);
};

// A's value against B's: the first of the two in the tie order says which way
// is better where they disagree, and "method changed" where they are one
// backend's test at two revisions.
Verdict judge(const CompareValue &a, const CompareValue &b);

// One side's best of one measure key on one processor, with each backend's.
struct MeasureRow {
    MeasureKey                key;
    CompareValue              best;
    std::vector<CompareValue> backends;   // best first, so best's own leads
    Category                  category = Category::Unknown;

    static MeasureRow of(const CompareRun &run, const MeasureBest &m);
    std::vector<std::string> label() const { return best.label(); }
};

// One key both processors have.
struct KeyCompare {
    MeasureKey key;
    MeasureRow a, b;
    Verdict    verdict;
    Category   category = Category::Unknown;
    // The same words on both sides, but for a reading with no coordinates;
    // then the first in the tie order's.
    std::vector<std::string> label() const;
};

// A headline on both processors: each side's value, the best of the keys it
// selects there, which may be readings of different keys.
struct HeadlineCompare {
    const Headline *headline = nullptr;
    bool hasA = false, hasB = false;
    CompareValue a, b;
    std::vector<CompareValue> aBackends, bBackends;   // best first
    bool    hasVerdict = false;                       // both sides have one
    Verdict verdict;
};

// One reading both devices measured.
struct ReadingCompare {
    CompareValue a, b;
    Verdict      verdict;
};

// One device both runs saw on a processor of one model: one backend, one
// device name, the same occurrence of it.  Reading by reading.
struct DeviceCompare {
    const DeviceResult *a = nullptr;
    const DeviceResult *b = nullptr;
    int occurrence = 0;
    std::vector<ReadingCompare> readings;      // A's order
    std::vector<CompareValue> onlyA, onlyB;
};

// What two processors of two runs must share to be one part
// (docs/format-v4.md, Models), as one string: equality is what counts.
std::string processorModel(const CompareRun &run, const ProcessorMeasures &g);

// Two processors side by side.
struct PairCompare {
    const CompareRun        *runA = nullptr;
    const ProcessorMeasures *a    = nullptr;
    const CompareRun        *runB = nullptr;
    const ProcessorMeasures *b    = nullptr;
    PairedBy                 pairedBy = PairedBy::Person;
    std::string              modelA, modelB;

    std::vector<HeadlineCompare> headlines;     // catalog order
    std::vector<KeyCompare>      both;          // by category, then measureKeyOrder
    std::vector<MeasureRow>      onlyA, onlyB;  // the same order
    // Where the two are one model: each device both runs saw on it, in A's
    // order, reading by reading, and the devices one side alone saw.
    std::vector<DeviceCompare>       devices;
    std::vector<const DeviceResult *> devicesOnlyA, devicesOnlyB;

    bool sameModel() const { return modelA == modelB; }
    // Reading by reading where one model and a device both runs saw.
    bool byReadings() const { return !devices.empty(); }

    static PairCompare of(const CompareRun &runA, const ProcessorMeasures &a,
                          const CompareRun &runB, const ProcessorMeasures &b,
                          const HeadlineCatalog &catalog, PairedBy by = PairedBy::Person);
};

struct ProcessorPair {
    const ProcessorMeasures *a = nullptr;
    const ProcessorMeasures *b = nullptr;
    PairedBy                 by = PairedBy::Person;
};

// A's processors paired with B's, by kind, then by model: each of A's, in
// A's order, takes the first of B's of its kind and model still free; then
// the rest pair by kind alone, in order.  What is left is unpaired.  Every
// group the measure engine made takes part.  Pairs come in A's order.
std::vector<ProcessorPair> pairProcessors(const CompareRun &a, const CompareRun &b);

// The order a compare lists measure keys in, within a category: by family
// (the Benchmark enum's order; a family this build does not know after, by
// name), quantity (likewise), identity dims (the data type first, then the
// other keys by name; absent before present; values in their list's order or
// by number), then the key's text.  <0, 0, >0.
int measureKeyOrder(const MeasureKey &a, const MeasureKey &b);

// A key's category: its family's (categoryOf()), or for a family this build
// does not know, that of the first of `values`, in the tie order, whose test
// says.
Category keyCategory(const MeasureKey &key, const std::vector<const CompareValue *> &values);

// Two runs side by side.
struct RunCompare {
    const CompareRun      *a = nullptr;
    const CompareRun      *b = nullptr;
    const HeadlineCatalog *catalog = nullptr;
    std::vector<PairCompare>               pairs;       // A's processor order
    std::vector<const ProcessorMeasures *> unpairedA;   // A's order
    std::vector<const ProcessorMeasures *> unpairedB;   // B's order

    static RunCompare of(const CompareRun &a, const CompareRun &b, const HeadlineCatalog &catalog);
};

// ── The baseline a run's rows meet ─────────────────────────────────────────
//
// --compare's per-row note, reading by reading: every measured reading of
// the saved run by its device's backend, name and occurrence, its test key
// and its id, with what judge() needs of it.
class ReadingBaseline {
public:
    struct Reading {
        double      value = 0.0;
        int         revision = 1;
        Direction   direction = Direction::HigherIsBetter;
        int         deviceIndex = -1;
        UnitInfo    units;
    };

    explicit ReadingBaseline(const RunDocument &doc);

    const Reading *find(const std::string &backend, const std::string &device, int occurrence,
                        const std::string &testKey, const std::string &metricId) const;

    // A reading of this run against the baseline's of the same device, test
    // and id: "  (was 4.48 TFLOPS, +3.1% better)", "  (was 4.48 TFLOPS,
    // method changed)"; "" where the baseline has none.  `value` is this
    // run's, printed as `units` scales it.
    std::string note(const std::string &backend, const std::string &device, int occurrence,
                     int deviceIndex, const std::string &testKey, int revision,
                     const std::string &metricId, double value, Direction direction,
                     const UnitInfo &units) const;

private:
    std::map<std::string, Reading> readings_;
};

// The --compare baseline of this process, loaded once from `file` and kept:
// the document, and its readings as the rows meet them.  nullptr, after the
// loader's message on stderr, for a file that cannot be read as a result --
// clpeak 3.x's included, which is refused.  Every backend's logger and the
// CLI's compare share the one load.
struct CompareBaseline {
    std::string     file;
    RunDocument     document;
    std::unique_ptr<ReadingBaseline> readings;
};
std::shared_ptr<const CompareBaseline> loadCompareBaseline(const std::string &file);

} // namespace xpupeak

#endif // XPUPEAK_RUN_COMPARE_H
