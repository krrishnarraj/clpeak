#ifndef XPUPEAK_RUN_DOCUMENT_H
#define XPUPEAK_RUN_DOCUMENT_H

#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include <common/benchmark_enums.h>
#include <common/common.h>     // xpupeak::LogLevel
#include <common/host_info.h>  // SystemInfo, RunConditions
#include <common/inventory.h>
#include <common/units.h>
#include <common/vocabulary.h>

// ── The result document ────────────────────────────────────────────────────
//
// One run, as a tree: run -> devices -> tests -> metrics.  This is the shape
// the CLI prints, the shape the GUI renders, and the shape the file on disk
// holds, so nothing has to regroup a flat table into a hierarchy and guess at
// what belonged together.  Beside the tree, the hardware it ran on: the
// system, its packages and processors, which each device names
// (`processor`) -- one processor seen through several backends is one key.
//
// The single serialization is JSON (`saveRunJson` / `loadRunJson`).  XML and
// CSV are gone: XML's only advantage was nesting, which JSON has, and CSV
// could not carry device metadata or documentation at all.
//
// Schema and vocabulary are documented in docs/format-v4.md, with the rules
// that let the format change without breaking a file already shared
// (Compatibility): a minor only adds, a reader ignores what it does not
// know, and the model keeps vocabulary values as strings so a newer file's
// values survive a read.

// Dump-file schema version.  Loaders reject any other version outright.
// v4 is the format under the program's own names (schemas `xpupeak/...`,
// `generator`), and 4.0 reads no file an earlier version wrote: a file
// clpeak 3.x wrote is refused by its schema, with a message that says so.
// It stays 4 for the life of the compatibility promise.
constexpr int RESULT_FORMAT_VERSION = 4;

// The additive revision this build writes (`format_minor`; absent means 0).
// A minor may add optional keys, enum values, dim keys, families, quantities
// and sections, and nothing else; a reader accepts every minor of its major.
constexpr int RESULT_FORMAT_MINOR = 0;

// The writer every document names (`generator.name`).  The same under the
// second command name: that is this program too, and writes the same file.
constexpr const char RESULT_GENERATOR_NAME[] = "XpuPeak";

// A reading may carry no measurement when it could not be taken.  `value` is
// meaningful only for `Ok`; every other status carries a human-readable
// `reason` instead.
enum class ResultStatus {
    Ok,
    Unsupported,  // hardware / driver lacks the feature
    Skipped,      // deliberately not run
    Error         // attempted, failed at runtime (compile / OOM / timeout)
};

// Whether a test's metrics are comparable to one another.
//
//   Homogeneous   -- interchangeable variants of one measurement: float /
//                    float2 / float4, int8_dp chain depths, or a test with a
//                    single reading.  The best of them is the test's answer,
//                    so a presenter may collapse the test to that one number.
//
//   Heterogeneous -- each reading is its own measurement: cuBLASLt's nine
//                    datatypes, memory_latency's L1/L2/L3/DRAM, transfer's
//                    h2d vs d2h, a CPU test's ST vs MT.  There is no single
//                    answer, and picking the largest reading invents one.
//
// This cannot be derived from anything else.  `wmma_fp16` has one metric and
// is homogeneous; `mps-gemm-fp` has three and is not; both are flops.  The
// same tag even differs by backend -- a GPU's global_memory_bandwidth is a
// vector-width sweep, the CPU's is read/copy/triad.  So it is authored at the
// beginTest() call site, next to the description (include/common/AGENTS.md).
//
// Heterogeneous is the default: it never invents a headline number.  A test
// that has not been visited yet is therefore verbose, never wrong.  It is also
// deliberately the zero value, so a C-style descriptor a backend clears with
// `= {}` and forgets to classify lands on the conservative answer rather than
// on a headline nobody chose.
enum class TestShape {
    Heterogeneous,
    Homogeneous
};

// Canonical lower-snake names used in the dump format.
const char  *categoryString(Category c);
Category     categoryFromString(const std::string &s);
const char  *statusString(ResultStatus s);
ResultStatus statusFromString(const std::string &s);
const char  *shapeString(TestShape s);
TestShape    shapeFromString(const std::string &s);
const char  *deviceTypeString(DeviceType t);
DeviceType   deviceTypeFromString(const std::string &s);

// ---- Dims -----------------------------------------------------------------
//
// A reading's coordinates, from the vocabulary (include/common/vocabulary.h):
// `{dtype: fp16, acc: fp16}`, `{level: l1, scope: core}`.  A test's dims are
// what all its readings share; a reading's add to them and override them.
// Keys are strings, so a key a newer build wrote is kept -- a reader treats
// it as identity -- and values are a string or a whole number, and stay so.

struct DimValue {
    bool        isNumber = false;
    double      number   = 0.0;
    std::string text;

    static DimValue ofText(std::string s);
    static DimValue ofNumber(double n);

    // The value as one string -- "fp16", "4", "134217728" -- the form a
    // measure key and a default compare.
    std::string str() const;

    bool operator==(const DimValue &o) const;
    bool operator!=(const DimValue &o) const { return !(*this == o); }
};

// One key and its value.  The producer's forms check that the value suits
// the key -- a data type for `dtype`, a dim::Level for `level`, a number for
// `width` -- in a debug build.  The loader's takes any key.
struct DimEntry {
    std::string key;
    DimValue    value;

    DimEntry() = default;
    DimEntry(std::string k, DimValue v) : key(std::move(k)), value(std::move(v)) {}

    DimEntry(xpupeak::Dim k, xpupeak::DType v);
    DimEntry(xpupeak::Dim k, const char *text);
    DimEntry(xpupeak::Dim k, const std::string &text);

    template <class T, std::enable_if_t<std::is_integral<T>::value &&
                                        !std::is_same<T, bool>::value, int> = 0>
    DimEntry(xpupeak::Dim k, T v) : DimEntry(integer(k, static_cast<double>(v))) {}

    template <class E, std::enable_if_t<xpupeak::dim::IsWord<E>::value, int> = 0>
    DimEntry(xpupeak::Dim k, E v)
        : DimEntry(word(k, xpupeak::dim::WordOf<E>::key, static_cast<unsigned>(v))) {}

private:
    static DimEntry integer(xpupeak::Dim k, double v);
    static DimEntry word(xpupeak::Dim k, xpupeak::Dim owner, unsigned index);
};

struct Dims {
    std::vector<DimEntry> entries;   // in the order authored

    Dims() = default;
    Dims(std::initializer_list<DimEntry> list);

    bool   empty() const { return entries.empty(); }
    size_t size() const { return entries.size(); }

    // Add or replace one key.
    void set(DimEntry e);
    const DimValue *find(const std::string &key) const;

    std::vector<DimEntry>::const_iterator begin() const { return entries.begin(); }
    std::vector<DimEntry>::const_iterator end() const { return entries.end(); }

    bool operator==(const Dims &o) const;
    bool operator!=(const Dims &o) const { return !(*this == o); }
};

// A reading's dims as a consumer reads them: its test's, with its own added
// and winning.
Dims effectiveDims(const Dims &test, const Dims &reading);

// ---- Metric ---------------------------------------------------------------

// One reading.  Unit fields are overrides: they are set only when this
// reading is measured in something other than its test's unit, which is what
// lets a single heterogeneous test hold both flops and ops readings instead
// of being split into a `-fp` and a `-int` twin.
struct MetricResult {
    std::string  id;           // stable slug within the test: "fp8_e4m3"
    std::string  label;        // display form; empty means "same as id"
    ResultStatus status = ResultStatus::Ok;
    double       value  = 0.0;
    std::string  reason;       // populated only when status != Ok

    // What this one reading means, in plain language, authored at the emit()
    // or skip() call that produces it.  Travels with the row so a reopened
    // file explains itself.
    std::string  description;

    // Unit override.  `hasUnit` false means "inherit the test's".
    bool        hasUnit  = false;
    std::string unit;                             // resolved display symbol
    Quantity    quantity = Quantity::Unknown;

    // Direction override.  FromUnit means "inherit the test's".
    Direction   direction = Direction::FromUnit;

    // This reading's coordinates, over its test's (effectiveDims).
    Dims        dims;

    // The family this reading measures, when it is not its test's: a test
    // that holds readings of two Benchmarks -- the CPU's integer dot
    // product, int8 and int16 -- names the one it was opened under, and a
    // reading recorded while the other was open names that one.  A
    // Benchmark's flag name; empty means "its test's".
    std::string benchmark;

    std::string ext;           // experimental fields, raw JSON; "" for none

    const std::string &displayLabel() const { return label.empty() ? id : label; }
};

// ---- Test -----------------------------------------------------------------

struct TestResult {
    std::string id;            // canonical tag, ISA-free: "single_precision_compute"
    std::string title;         // human-readable: "Single-precision compute"

    // Runtime qualifier that is not part of the test's identity: the CPU
    // backend's detected ISA ("AVX2+FMA", "NEON + SME2 (SVL=512b)"), a GPU
    // arch, a library version.  Kept out of `id` on purpose -- baking it in is
    // what made CPU tags lossy and un-comparable across machines.
    std::string variant;

    // What varies from one metric to the next: "vector width", "data type",
    // "cache level", "direction", "threads", "pixel format".  Optional; a
    // presenter uses it as the column header over a heterogeneous test's
    // readings, and it is the plainest statement of why a test is the shape
    // it is.
    std::string axis;

    // One or two sentences on what this test measures, authored at the
    // beginTest() call site.
    std::string description;

    Category  category  = Category::Unknown;
    TestShape shape     = TestShape::Heterogeneous;
    Direction direction = Direction::HigherIsBetter;  // resolved, never FromUnit

    // Resolved from the authored unit token exactly once, at beginTest().
    std::string unit;                             // display symbol: "FLOPS"
    Quantity    quantity = Quantity::Unknown;

    // The family -- a Benchmark's flag name, "matrix-compute" -- that the
    // test measures: the one admit() had open when the test was opened.
    // What makes `cublas_gemm` and `coreml_gemm` the same measurement.
    std::string benchmark;

    // What every reading shares: a vector test's data type, a CPU test's
    // ISA.  A reading's own dims add to them.
    Dims        dims;

    // The test's methodology version: it goes up whenever a change moves the
    // numbers for a reason other than the hardware (a kernel fix, a counting
    // rule, a problem size), and readings of two revisions are not compared.
    // 1 is the first, and what an absent `revision` means.
    int         revision = 1;

    std::string ext;           // experimental fields, raw JSON; "" for none

    std::vector<MetricResult> metrics;

    // Wall-clock seconds the test was open, summed over reopens.  A test that
    // took forty seconds against a half-second budget is its own diagnosis --
    // a calibration that overran, a driver that stalled -- and nothing else in
    // the document says how long anything took.
    double durationS = 0.0;

    // Identity within a device.  `variant` participates: two ISAs of the same
    // CPU test are two tests, and comparing one against the other would be
    // meaningless.
    std::string key() const
    {
        return variant.empty() ? id : (id + "@" + variant);
    }

    // Unit context for one reading, honouring its overrides.
    const std::string &unitOf(const MetricResult &m) const
    {
        return m.hasUnit ? m.unit : unit;
    }
    Direction directionOf(const MetricResult &m) const
    {
        return (m.direction == Direction::FromUnit) ? direction : m.direction;
    }
};

// ---- Device ---------------------------------------------------------------

// One device's identity within a backend+platform: its name, plus the
// enumeration index when the backend reports one.  Shared by the document's
// own key and by the --compare baseline so the two never disagree.
std::string deviceKey(const std::string &name, int deviceIndex);


// Free-form per-device fact (compute units, VRAM, clocks, driver internals)
// captured once when the device is opened.  Not fields of a reading: they
// describe the device, not a measurement.  Display strings; the typed facts
// are HardwareFacts.
struct DeviceProp {
    std::string key;
    std::string value;
};

// ---- Hardware facts ---------------------------------------------------------
//
// What an API -- or for a CPU, the OS and CPUID -- says about the hardware,
// typed.  A device keeps its own API's (`facts`), and a processor the facts
// merged from the most native backend that ran, written on the processor
// itself; so a consumer with other rules can merge them again.  Vocabulary
// fields hold their strings.  None identifies a unit: a PCI device id is a
// model's, not a card's.

struct CacheInfo {
    std::string   level;           // CacheLevel: "l1d", "l2", "slc"
    std::uint64_t sizeBytes = 0;   // one instance
    unsigned      instances = 0;   // how many the processor, or the cluster, has
    unsigned      lineBytes = 0;
};

struct CpuCluster {
    std::string kind;              // ClusterKind; "" for a CPU of one kind of core
    unsigned    cores = 0;
    unsigned    threadsPerCore = 0;
    unsigned    maxMhz = 0;
    std::string microarch;         // as MIDR or CPUID decode it: "Firestorm", "Zen 2"
    std::string midr;              // ARM: MIDR_EL1 of its cores, hex
    std::vector<CacheInfo> caches; // the cluster's own
};

struct CpuEngine {
    std::string kind;              // EngineKind: "matrix"
    std::string name;              // "AMX", "SME"
    std::string scope;             // EngineScope: one per "core" | per "cluster"
    std::string reachedBy;         // EngineReach: "instructions" | "library"
};

struct CpuFacts {
    std::string isa;               // "arm64", "x86_64"
    unsigned    cores   = 0;
    unsigned    threads = 0;
    unsigned    sockets = 0;
    std::vector<CpuCluster> clusters;
    std::vector<CacheInfo>  caches;    // the ones the whole processor shares
    // x86: CPUID's vendor string and family / model / stepping; -1 unknown.
    std::string cpuidVendor;
    int         cpuidFamily   = -1;
    int         cpuidModel    = -1;
    int         cpuidStepping = -1;
    std::vector<CpuEngine>   engines;
    std::vector<std::string> features; // normalised: "neon", "dotprod", "avx512_vnni"

    bool empty() const;
};

struct HardwareFacts {
    std::string   vendorId;        // the API's vendor id, lower-case hex: "10de", "10005"
    std::string   deviceId;        // the PCI device id, lower-case hex: a model id
    std::string   arch;            // "sm_120", "gfx1201", "Apple7"
    unsigned      computeUnits = 0;
    std::string   computeUnit;     // what one is called: "SM", "CU", "Xe-core", "core"
    unsigned      clockMhz = 0;
    std::uint64_t memoryBytes = 0;
    int           integrated = -1; // -1 unknown, 0 discrete, 1 integrated
    CpuFacts      cpu;             // a CPU's

    bool empty() const;
};

// What links a device to its processor, as the document keeps it: tokens for
// the strong identifiers its API gave -- HMACs under a key made per run and
// never written to the result, so two devices in one file can be seen to
// share an identifier while nothing correlates across files -- and the
// weaker evidence, which identifies nothing.  The raw identifiers never
// leave the process (logger::DeviceSpec::Identity).
struct DeviceEvidence {
    std::map<std::string, std::string> tokens;   // "uuid" | "luid" | "pci_bus" | "registry"
    bool        host = false;      // executes on the host CPU
    std::string soc;               // ProcessorKind: the SoC's "npu" or "gpu"
    bool        defaultGpu = false;// the platform's default GPU

    bool empty() const { return tokens.empty() && !host && soc.empty() && !defaultGpu; }
};

// An ML runtime's view of where it ran.
struct RuntimeInfo {
    std::string provider;          // ONNX Runtime's execution provider: "CoreMLExecutionProvider"
    std::string accelerator;       // LiteRT's accelerator, Core ML's compute device: "NPU", "GPU"
    std::string version;           // the runtime's version: "1.31.0"

    bool empty() const { return provider.empty() && accelerator.empty() && version.empty(); }
};

struct DeviceResult {
    std::string backend;       // "OpenCL" | "Vulkan" | "CUDA" | "Metal" | …
    std::string platform;      // vendor platform; defaults to the backend name
    std::string name;
    std::string driver;
    DeviceType  type = DeviceType::Unknown;

    int platformIndex = -1;
    int deviceIndex   = -1;

    std::vector<DeviceProp> properties;

    // The processor this device is a view of (ProcessorInfo::id: "gpu0"),
    // and what linked it there (LinkEvidence: "uuid", "host", "sole").
    // Empty until the run is linked.
    std::string    processor;
    std::string    link;
    HardwareFacts  facts;      // what its API reported
    DeviceEvidence evidence;
    RuntimeInfo    runtime;    // ML backends
    std::string    ext;        // experimental fields, raw JSON; "" for none

    std::vector<TestResult> tests;

    // `driver` is metadata, not identity: a baseline stays comparable across
    // a driver update, which is exactly the comparison people want.  The
    // enumeration index IS identity, because a name is not unique -- MoltenVK
    // exposes one GPU twice, and a multi-GPU box has N identical cards.  Left
    // out, their readings fold into a single block and each test ends up with
    // two of every reading.
    std::string key() const
    {
        return backend + "/" + platform + "/" + deviceKey(name, deviceIndex);
    }

    // Find an already-recorded test by TestResult::key(), or nullptr.  This is
    // what lets a backend reopen a test to append readings measured in a later
    // category phase (cuBLASLt's int metrics joining its fp ones).
    TestResult *findTest(const std::string &testKey);
};

// ---- Run ------------------------------------------------------------------

// One line of the run's diagnostic stream: what XpuPeak (or a library it
// called) had to say, at what moment, inside which scope.  The whole stream
// in emission order is the document's `log`; it is what a maintainer reads
// when a number from a machine they cannot reach looks wrong, so it carries
// enough to stand alone -- a `jq` over it reproduces the terminal transcript.
//
// Scope fields are empty when no such scope was open, and the writer omits
// them.  `test` is the test's key (`id@variant`), joinable to `tests[]`.
struct LogEntry {
    double           elapsedS = 0.0;     // seconds since the run started
    xpupeak::LogLevel level    = xpupeak::LogLevel::Info;
    std::string      source;             // "" = XpuPeak; else the library relayed
    std::string      backend;
    std::string      device;
    int              deviceIndex = -1;
    std::string      test;
    std::string      message;
};

// One backend compiled into the binary, with what it reaches.
struct BuildBackend {
    std::string name;                    // as the document's devices name it: "CUDA"
    std::string kind;                    // BackendKind: "gpu" | "cpu" | "ai"
    bool        portable = false;        // runs on more than one vendor's silicon
};

// The entry for one of this build's backends.
BuildBackend buildBackendFor(Backend b);

// What the binary that produced the file is made of.  The first question on
// any "backend X is missing" report is whether it was ever compiled in.
struct BuildInfo {
    std::vector<BuildBackend> backends;  // registry order
    std::string config;                  // CMake configuration ("Release", "Debug"), when the build stamped it
};

// How XpuPeak was asked to run.  Recorded because every number here is
// sensitive to it: a shorter --max-time-gpu measures a different thing, and a
// selective run is not a full one even though the file looks the same shape.
struct Invocation {
    std::vector<std::string> argv;
    unsigned targetTimeUs      = 0;
    unsigned targetTimeUsCpu   = 0;
    unsigned targetTimeUsMlGpu = 0;
    unsigned warmup            = 0;
    unsigned iters             = 0;   // 0 = calibrated per test, not forced
    std::vector<std::string> categories;  // enabled category names
    std::vector<std::string> tests;       // empty = every test enabled

    // --verbose: the log carries Debug entries.  Recorded so an empty debug
    // log reads as "was not asked", not "nothing happened".
    bool verbose = false;

    // The run used this build's default budgets, warm-up and iterations, and
    // ran every category and test: the run a leaderboard ranks.  Choosing
    // backends and devices does not change it.
    bool standard = false;
};

// Where in a run something happened: a test on a device, named as the log
// names it.
struct RunPlace {
    std::string backend;
    std::string device;
    int         deviceIndex = -1;
    std::string test;               // the test's key, id@variant

    bool empty() const { return backend.empty(); }
};

// One part of a run after its first.  The process a run was in ended
// part-way -- a phone closed the app, the process crashed -- and the run was
// continued from its checkpoint (--resume, include/common/run_resume.h) into
// the same document, so its devices were measured in more than one sitting.
struct ResumeRecord {
    std::string at;                 // ISO-8601 UTC: when this part started
    double      elapsedS = 0.0;     // the same moment on the log's clock
    std::string after;              // why the part before ended, when known
    RunPlace    from;               // the first test this part measured
    RunPlace    skipped;            // --skip-crashed: where the part before died, not run again
};

// What a run has finished, kept in its checkpoint so that another process can
// continue it: the backends finished whole, and per device either the whole
// device or the Benchmarks finished on it, by their flag names.
struct RunProgress {
    struct Device {
        std::string backend;        // as the document names it: "CPU"
        int         deviceIndex = -1;
        bool        done = false;   // every Benchmark that was going to run on it did
        std::vector<std::string> benchmarks;   // "single-precision-compute", ...
        // Tests ("id" or "id@variant") of a Benchmark not finished, saved
        // with the readings they had when the run held (RunCheckpoint::held).
        std::vector<std::string> held;
    };
    std::vector<std::string> backendsDone;
    std::vector<Device>      devices;

    Device &device(const std::string &backend, int deviceIndex);
    bool    backendDone(const std::string &backend) const;
};

// What wrote the document: `generator` in the file.
struct GeneratorInfo {
    std::string name;               // RESULT_GENERATOR_NAME
    std::string version;            // the build's XPUPEAK_VERSION_STR
};

struct RunMeta {
    GeneratorInfo generator;
    std::string generatedAt;        // ISO-8601 UTC, "2026-08-29T14:03:11Z"

    // A random UUID per run, kept across --resume, so one run uploaded twice
    // is recognised as one.  It identifies the run, not who ran it.
    std::string runId;

    // The format_minor a loaded file was written in; 0 for a file that says
    // none.  The writer always writes RESULT_FORMAT_MINOR: what this build
    // writes is what it knows.
    int         formatMinor = 0;

    // The run's key for linking evidence (DeviceEvidence::tokens), as hex.
    // Made per run, kept in the checkpoint so a resumed run's tokens match the
    // part before's, and never written to the result or the sidecar.
    std::string evidenceKey;

    // Seconds spent running.  For a resumed run, the parts summed: the time
    // between them is not the run's.
    double      durationS = 0.0;

    // A cancelled run is a partial one.  Without this flag a file written
    // after a cancel is indistinguishable from a complete run, and every
    // absent test reads as unsupported hardware.
    bool        cancelled = false;

    BuildInfo     build;
    SystemInfo    system;         // the machine (host_info.h)
    RunConditions conditions;     // what it was doing (host_info.h)
    Invocation    invocation;

    // Facts only a person knows -- a name for the system, its cooling, a
    // power limit, notes -- as a flat string map.  Never filled by a
    // measurement, and never overriding one.
    std::map<std::string, std::string> annotations;

    // Each part after the first, in order; empty for a run done in one go.
    std::vector<ResumeRecord> resumed;

    std::string ext;              // the run's experimental fields, raw JSON
};

// ---- Hardware ---------------------------------------------------------------
//
// The run's hardware, which the devices are views of: packages (an SoC, a CPU
// package, a discrete card) and the processors in them, with run-local ids
// that are deterministic for one machine -- `cpu0`, `gpu0`, `npu0`.

struct PackageInfo {
    std::string id;               // "pkg0"
    std::string kind;             // PackageKind: "soc" | "cpu" | "discrete"
    std::string vendor;           // Vendor, when known
    std::string name;             // "Apple M1 Pro", "NVIDIA GeForce RTX 5060"
    std::string ext;
};

struct ProcessorInfo {
    std::string   id;             // "cpu0", "gpu0", "npu0"
    std::string   kind;           // ProcessorKind
    std::string   package;        // PackageInfo::id
    std::string   vendor;         // Vendor, when known
    std::string   vendorName;     // the vendor's own string: "NVIDIA Corporation"
    std::string   name;
    HardwareFacts facts;          // merged; written on the processor itself
    std::string   ext;
};

struct RunDocument {
    RunMeta                    meta;
    std::vector<PackageInfo>   packages;
    std::vector<ProcessorInfo> processors;
    std::vector<DeviceResult>  devices;

    // Every backend's `--list-devices` view, taken before the run: which
    // backends were available (and why not), and every device each one saw
    // -- including the ones a --devices list left out.  Written under
    // --verbose only: in the CLI it costs an enumeration pass, and it is the
    // evidence for "my NPU is not listed", which is a debugging question.
    std::vector<BackendInventory> inventory;

    // The diagnostic stream, in emission order, for the whole run.  Filled by
    // RunLog (run_log.h) across every backend's logger, so append() leaves it
    // alone.
    std::vector<LogEntry>     log;

    bool empty() const { return devices.empty(); }

    // Find a device by DeviceResult::key(), or nullptr.
    DeviceResult *findDevice(const std::string &deviceKey);

    // Append another document's devices.  Used to fold the per-backend
    // loggers into one file; `meta`, `packages`, `processors`, `inventory`
    // and `log` are the caller's.
    void append(const RunDocument &other);
};

// ---- Baselines ------------------------------------------------------------

// `--compare` lookup.  Keyed on
//   backend/platform/device/test.key()/metric.id
// -- driver and category are deliberately absent, so a baseline survives a
// driver update and a test being re-filed under another category.
typedef std::map<std::string, double> BaselineMap;

std::string baselineKey(const std::string &backend, const std::string &platform,
                        const std::string &device, const std::string &testKey,
                        const std::string &metricId);

// Only Ok readings participate; there is nothing to compare a skip against.
BaselineMap buildBaselineMap(const RunDocument &doc);

// ---- Helpers --------------------------------------------------------------

// Current UTC time as ISO-8601 ("2026-08-29T14:03:11Z").  Both hosts (CLI and
// the GUI bridge) stamp a run with this, so it lives next to the field it
// fills rather than being written twice.
std::string isoTimestampUtc();

// The same form back to seconds since the epoch; false when it is not one.
bool parseIsoTimestampUtc(const std::string &iso, long long &secondsSinceEpoch);

// A new run's `run_id`: a random (version 4) UUID, "6f1c2e8a-0d4b-4a59-…".
std::string newRunId();

// A new run's evidence key: 32 random bytes, as hex (RunMeta::evidenceKey).
std::string newEvidenceKey();

// ---- Serialization --------------------------------------------------------

// An `ext` as the document keeps it: `json` when it is an object, as compact
// JSON text, and "" when it is not one -- which then is not written.
std::string extFromJson(const std::string &json);

// Dims as the compact JSON object the document writes ("{}" for none): what
// the GUI event stream carries (src/ffi/logger_ffi.cpp).
std::string dimsToJson(const Dims &dims);

// One log entry as a single-line JSON object -- the form the document's
// `log` array holds one per line, the run-log sidecar holds one per line, and
// the GUI event stream carries.  One serializer for all three.
std::string logEntryToJson(const LogEntry &entry);

// The first line of the run-log sidecar (run_log.h): the run's identity as a
// single-line object, so a sidecar left behind by a crash says which run it
// belonged to without the document that was never written.
std::string runLogHeaderJson(const RunMeta &meta);

// One line of the sidecar back as an entry; false for a line that is not one
// (the header, a line a crash cut short).
bool logEntryFromJson(const std::string &line, LogEntry &out);

// Write the document as JSON.  Returns false (after a stderr message) when
// the file cannot be opened or the stream fails while writing.
bool saveRunJson(const RunDocument &doc, const std::string &filename);

// Same document, returned as a string.
std::string runDocumentToJson(const RunDocument &doc);

// Read a document back.  Returns false (after a stderr message) for a missing
// file, malformed JSON, a file clpeak 3.x wrote, a format_version this build
// does not write, or a run's checkpoint, which is not a result.  Any
// format_minor of format 4 is read, and what this build does not know -- a
// key, a vocabulary value, a dim -- is ignored or kept as written, never an
// error; a file from a newer minor says so on stderr.
bool loadRunJson(const std::string &filename, RunDocument &out);

// The same, from the document's text; `name` is what a message calls it.
// Silent: false with `error` set, and `newerMinor` true for a file a newer
// minor wrote.
bool parseRunJson(const std::string &text, const std::string &name, RunDocument &out,
                  std::string &error, bool &newerMinor);

// A run's checkpoint (include/common/run_checkpoint.h): the document so far
// -- meta (the evidence key included), hardware and devices, no log or
// inventory -- with schema "xpupeak/run-partial" and its `progress`, on one
// line.
std::string partialRunToJson(const RunDocument &doc, const RunProgress &progress);

// Read one back.  false with `error` set for a missing file, malformed JSON,
// a checkpoint clpeak 3.x wrote, or a schema or format_version this build
// does not write.
bool loadPartialJson(const std::string &filename, RunDocument &out,
                     RunProgress &progress, std::string &error);

#endif  // XPUPEAK_RUN_DOCUMENT_H
