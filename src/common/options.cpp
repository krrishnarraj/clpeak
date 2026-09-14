#include <common/options.h>
#include <common/benchmark_enums.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <version.h>

// The command line has three kinds of selection flag, all spelt the same
// way: --<x> runs only what is named (an allow-list; several combine) and
// --no-<x> subtracts.  Backends say WHERE, categories and tests say WHAT,
// and the two are independent -- a test flag applies to every backend that
// runs.  So there is no --cuda-gemm or --onnx-gemm: `--cuda --gemm` is
// cuBLASLt and `--onnx --gemm` is a MatMul through an execution provider.
//
// Every flag parses in every build.  A script that says --no-cuda must work
// on a Mac, and the GUI emits the same argv on every platform.  A backend
// that is not built in is simply absent from the run (and named, when it
// was asked for -- see CliOptions::requestedButNotBuilt); a test no built-in
// backend implements runs nothing.

// ---- Backend table --------------------------------------------------------

#ifdef ENABLE_OPENCL
#define CLPEAK_BUILT_OPENCL true
#else
#define CLPEAK_BUILT_OPENCL false
#endif
#ifdef ENABLE_VULKAN
#define CLPEAK_BUILT_VULKAN true
#else
#define CLPEAK_BUILT_VULKAN false
#endif
#ifdef ENABLE_CUDA
#define CLPEAK_BUILT_CUDA true
#else
#define CLPEAK_BUILT_CUDA false
#endif
#ifdef ENABLE_ROCM
#define CLPEAK_BUILT_ROCM true
#else
#define CLPEAK_BUILT_ROCM false
#endif
#ifdef ENABLE_METAL
#define CLPEAK_BUILT_METAL true
#else
#define CLPEAK_BUILT_METAL false
#endif
#ifdef ENABLE_ONEAPI
#define CLPEAK_BUILT_ONEAPI true
#else
#define CLPEAK_BUILT_ONEAPI false
#endif
#ifdef ENABLE_CPU
#define CLPEAK_BUILT_CPU true
#else
#define CLPEAK_BUILT_CPU false
#endif
#ifdef ENABLE_ONNX
#define CLPEAK_BUILT_ONNX true
#else
#define CLPEAK_BUILT_ONNX false
#endif
#ifdef ENABLE_COREML
#define CLPEAK_BUILT_COREML true
#else
#define CLPEAK_BUILT_COREML false
#endif

struct BackendRow {
  BackendInfo info;
  const char *blurb;   // one line for --help
};

// Indexed by Backend; the static_assert below keeps it that way.
static const BackendRow backendTable[] = {
  {{Backend::OpenCL, "OpenCL", "opencl", CLPEAK_BUILT_OPENCL}, "OpenCL"},
  {{Backend::Vulkan, "Vulkan", "vulkan", CLPEAK_BUILT_VULKAN}, "Vulkan"},
  {{Backend::Cuda,   "CUDA",   "cuda",   CLPEAK_BUILT_CUDA},   "CUDA"},
  {{Backend::Rocm,   "ROCm",   "rocm",   CLPEAK_BUILT_ROCM},   "ROCm/HIP"},
  {{Backend::Metal,  "Metal",  "metal",  CLPEAK_BUILT_METAL},  "Metal"},
  {{Backend::Oneapi, "oneAPI", "oneapi", CLPEAK_BUILT_ONEAPI}, "oneAPI/SYCL"},
  {{Backend::Cpu,    "CPU",    "cpu",    CLPEAK_BUILT_CPU},    "native CPU"},
  {{Backend::Onnx,   "ONNX",   "onnx",   CLPEAK_BUILT_ONNX},   "ONNX Runtime (NPUs via execution providers)"},
  {{Backend::Coreml, "CoreML", "coreml", CLPEAK_BUILT_COREML}, "Core ML (Apple Neural Engine / GPU / CPU)"},
};
static const int numBackends = sizeof(backendTable) / sizeof(backendTable[0]);
static_assert(numBackends == static_cast<int>(Backend::COUNT),
              "backendTable must have one row per Backend, in enum order");

const BackendInfo &backendInfo(Backend b)
{
  return backendTable[static_cast<size_t>(b)].info;
}

std::vector<Backend> CliOptions::requestedButNotBuilt() const
{
  std::vector<Backend> out;
  for (int i = 0; i < numBackends; i++)
    if (requestedBackends.test(static_cast<size_t>(i)) && !backendTable[i].info.builtIn)
      out.push_back(backendTable[i].info.id);
  return out;
}

// ---- Category and test tables ------------------------------------------------

struct CategoryFlag {
  const char *name;
  Category    cat;
  const char *blurb;
};

static const CategoryFlag categoryFlags[] = {
  {"compute",   Category::Compute,   "arithmetic, matrix engines and library GEMM (flops / ops)"},
  {"crypto",    Category::Crypto,    "crypto/hash silicon (bps)"},
  {"string",    Category::String,    "string/text processing (bps)"},
  {"bandwidth", Category::Bandwidth, "memory and transfer bandwidth (bps)"},
  {"latency",   Category::Latency,   "launch, memory and micro-architectural latency (s)"},
  {"ai",        Category::Ai,        "AI composites: the transformer block (flops, bps, s)"},
};
static const int numCategoryFlags = sizeof(categoryFlags) / sizeof(categoryFlags[0]);

struct TestFlag {
  const char *name;        // flag suffix; e.g. "gemm" matches --gemm / --no-gemm
  Benchmark   test;
  const char *blurb;
};

// --help groups these by categoryOf(test), in table order within a group,
// so a test can never be listed under the wrong heading.
static const TestFlag testFlags[] = {
  {"single-precision-compute",  Benchmark::ComputeSP,       "fp32"},
  {"half-precision-compute",    Benchmark::ComputeHP,       "fp16"},
  {"double-precision-compute",  Benchmark::ComputeDP,       "fp64"},
  {"mixed-precision-compute",   Benchmark::ComputeMP,       "fp16 inputs, fp32 accumulate"},
  {"bfloat16-compute",          Benchmark::ComputeBF16,     "bf16"},
  {"integer-compute",           Benchmark::ComputeInt,      "int32"},
  {"integer-compute-fast",      Benchmark::ComputeIntFast,  "24-bit integer (mad24)"},
  {"integer-compute-char",      Benchmark::ComputeChar,     "8-bit integer vectors"},
  {"integer-compute-short",     Benchmark::ComputeShort,    "16-bit integer vectors"},
  {"int8-dot-product-compute",  Benchmark::ComputeInt8DP,   "int8 dot product (DP4a / VNNI / SDOT / dot())"},
  {"int16-dot-product-compute", Benchmark::ComputeInt16DP,  "int16 dot product (x86 VNNI)"},
  {"fp8-dot-product-compute",   Benchmark::ComputeFP8DP,    "fp8 dot product (ARM FP8)"},
  {"divide-sqrt-compute",       Benchmark::ComputeDivSqrt,  "fp divide and sqrt throughput"},
  {"integer-divide-compute",    Benchmark::ComputeIntDiv,   "64-bit integer divide throughput"},
  {"matrix-compute",            Benchmark::MatrixCompute,   "matrix engine via intrinsics: tensor cores, MFMA/WMMA,\n"
                                                            "coopmat, simdgroup_matrix, joint_matrix, AMX/SME"},
  {"gemm",                      Benchmark::Gemm,            "the vendor library's tuned matmul: cuBLASLt, hipBLASLt,\n"
                                                            "oneMKL, MPS, Accelerate, ONNX MatMul, Core ML"},
  {"attention",                 Benchmark::Attention,       "scaled-dot-product attention through the vendor library"},
  {"convolution",               Benchmark::Conv,            "2-D convolution peak through a graph runtime"},
  {"numeric-error",             Benchmark::NumericError,    "accuracy cost of each dtype vs an fp32 reference (ppm)"},
  {"smt-scaling",               Benchmark::SmtScaling,      "fp32 FMA at one thread per core vs every SMT thread"},

  {"aes",                       Benchmark::CryptoAes,       "AES-128 (AES-NI / VAES / ARM AES)"},
  {"sha256",                    Benchmark::CryptoSha256,    "SHA-256 (SHA-NI / ARM SHA2)"},
  {"sha512",                    Benchmark::CryptoSha512,    "SHA-512 (ARM SHA512)"},
  {"crc32c",                    Benchmark::CryptoCrc32c,    "CRC32-C"},
  {"string-scan",               Benchmark::StringScan,      "memchr-style SIMD byte scan"},
  {"utf8-validate",             Benchmark::Utf8Validate,    "UTF-8 validation (PSHUFB / TBL)"},

  {"global-memory-bandwidth",   Benchmark::GlobalBW,        "device memory"},
  {"local-memory-bandwidth",    Benchmark::LocalBW,         "work-group local / shared memory"},
  {"image-memory-bandwidth",    Benchmark::ImageBW,         "image / texture memory"},
  {"transfer-bandwidth",        Benchmark::TransferBW,      "host <-> device, each direction"},
  {"tensor-bandwidth",          Benchmark::TensorBW,        "resident-tensor read through a graph runtime"},
  {"activation",                Benchmark::Activation,      "softmax / layer-norm / SiLU throughput through a graph\n"
                                                            "runtime"},
  {"cache-bandwidth",           Benchmark::CacheBandwidth,  "per cache level and DRAM"},
  {"texture-sample",            Benchmark::TextureSample,   "bilinear texel rate"},

  {"kernel-launch-latency",     Benchmark::KernelLatency,   "the fixed cost of one submission: a kernel launch, a\n"
                                                            "session run, a prediction"},
  {"memory-latency",            Benchmark::MemoryLatency,   "pointer chase per memory level, MLP and TLB"},
  {"atomics",                   Benchmark::Atomics,         "atomic fetch-add, uncontended and contended"},
  {"branch-penalty",            Benchmark::BranchPenalty,   "branch mispredict cost"},
  {"store-forward",             Benchmark::StoreForward,    "store-to-load forwarding round trip"},

  {"transformer-block",         Benchmark::TransformerBlock, "one decoder block: prefill, decode and latency at each\n"
                                                             "precision"},
};
static const int numTestFlags = sizeof(testFlags) / sizeof(testFlags[0]);
static_assert(numTestFlags == static_cast<int>(Benchmark::COUNT),
              "every Benchmark needs exactly one flag");

// ---- Help ---------------------------------------------------------------------

// One help entry: the flag, padded to the description column, then the
// text; a '\n' inside the text continues on the next line at that column.
static void helpLine(std::string &s, const std::string &flag, const char *text,
                     int indent = 2)
{
  const size_t column = 33;
  s += std::string(static_cast<size_t>(indent), ' ');
  s += flag;
  if (flag.size() + static_cast<size_t>(indent) + 2 > column)
  {
    s += "\n";
    s += std::string(column, ' ');
  }
  else
  {
    s += std::string(column - flag.size() - static_cast<size_t>(indent), ' ');
  }
  for (const char *c = text; *c; c++)
  {
    s += *c;
    if (*c == '\n')
      s += std::string(column, ' ');
  }
  s += "\n";
}

static std::string helpText()
{
  std::string s;
  s += "\n clpeak [OPTIONS]\n";
  s += "\n";
  s += " Selection flags come in pairs: --<x> runs only what is named (an allow-list;\n";
  s += " several combine) and --no-<x> subtracts.  Backends say where, categories and\n";
  s += " tests say what, and a test flag applies to every backend that runs.\n";
  s += "\n";
  s += " GLOBAL OPTIONS:\n";
  helpLine(s, "-h, --help",        "display help message");
  helpLine(s, "-v, --version",     "display version");
  helpLine(s, "-i, --iters num",   "force a fixed iter count (overrides --max-time calibration)");
  helpLine(s, "-w, --warmup num",  "number of warm-up kernel runs before timing (default: 2)");
  helpLine(s, "--max-time ms",     "per-test time budget for the timed phase, every backend\n"
                                   "except CPU (default: 500 ms).  Iters are picked to fit it,\n"
                                   "so set it lower if you hit a GPU watchdog");
  helpLine(s, "--max-time-cpu ms", "per-test time budget for the CPU backend (default: 2000 ms)");
  helpLine(s, "--verbose",         "print backend debug logs (kernel build logs, API errors)");
  helpLine(s, "--describe",        "explain what each test and each reading measures");
  helpLine(s, "--list-devices",    "list available devices for every backend and exit");
  helpLine(s, "-o, --output file", "save results to a JSON file");
  helpLine(s, "--compare file",    "compare results against a saved run");
  helpLine(s, "--onnx-lib path",   "onnxruntime shared library to load\n"
                                   "(default: the platform's conventional names)");
  s += "\n";
  s += " BACKENDS (--<backend> / --no-<backend>; default: every one in this build):\n";
  for (int i = 0; i < numBackends; i++)
  {
    std::string text = backendTable[i].blurb;
    if (!backendTable[i].info.builtIn)
      text += "  (not in this build)";
    helpLine(s, std::string("--") + backendTable[i].info.flag, text.c_str());
  }
  s += "\n";
  s += " DEVICES (default: every device of every backend that runs):\n";
  helpLine(s, "--device list",     "run only these devices: comma-separated backend:index\n"
                                   "items, exactly as --list-devices prints them\n"
                                   "(e.g. --device cuda:0,vulkan:1)");
  s += "\n";
  s += " CATEGORIES (--<category> / --no-<category>; default: all):\n";
  for (int i = 0; i < numCategoryFlags; i++)
    helpLine(s, std::string("--") + categoryFlags[i].name, categoryFlags[i].blurb);
  s += "\n";
  s += " TESTS (--<test> / --no-<test>; default: every test a backend supports):\n";
  for (int c = 0; c < numCategoryFlags; c++)
  {
    s += "  ";
    s += categoryFlags[c].name;
    s += "\n";
    for (int t = 0; t < numTestFlags; t++)
      if (categoryOf(testFlags[t].test) == categoryFlags[c].cat)
        helpLine(s, std::string("--") + testFlags[t].name, testFlags[t].blurb, 3);
  }
  s += "\n";
  return s;
}

static void printHelpAndExit(int code)
{
  std::cout << helpText();
  std::cout.flush();
  std::exit(code);
}

// ---- Value parsing ---------------------------------------------------------------

static bool parseUnsignedLongArg(const char *arg, unsigned long &value)
{
  char *end = nullptr;
  errno = 0;
  value = strtoul(arg, &end, 0);
  return (errno != ERANGE) && (end != arg) && (*end == '\0');
}

static bool parseUIntArg(const char *arg, unsigned int &value, bool allowZero = true)
{
  unsigned long parsed;
  if (!parseUnsignedLongArg(arg, parsed) ||
      parsed > std::numeric_limits<unsigned int>::max() ||
      (!allowZero && parsed == 0))
    return false;
  value = static_cast<unsigned int>(parsed);
  return true;
}

static bool parseIntArg(const char *arg, int &value)
{
  unsigned long parsed;
  if (!parseUnsignedLongArg(arg, parsed) ||
      parsed > static_cast<unsigned long>(std::numeric_limits<int>::max()))
    return false;
  value = static_cast<int>(parsed);
  return true;
}

static const BackendRow *findBackendFlag(const std::string &flag)
{
  for (int i = 0; i < numBackends; i++)
    if (flag == backendTable[i].info.flag)
      return &backendTable[i];
  return nullptr;
}

// Parse the --device list: comma-separated `backend:index` items, the
// tokens --list-devices prints.  Empty items ("cuda:0,,cuda:2"), unknown
// backends and negative indices fail; `why` names the offending item.
static bool parseDeviceList(const char *arg, std::vector<DeviceSelector> &out,
                            std::bitset<static_cast<size_t>(Backend::COUNT)> &requested,
                            std::string &why)
{
  std::vector<DeviceSelector> parsed;
  std::stringstream ss(arg);
  std::string tok;
  while (std::getline(ss, tok, ','))
  {
    DeviceSelector sel;
    const size_t colon = tok.find(':');
    if (colon == std::string::npos)
    {
      why = "'" + tok + "': expected backend:index, as --list-devices prints it";
      return false;
    }
    const std::string backendPart = tok.substr(0, colon);
    const BackendRow *row = findBackendFlag(backendPart);
    if (!row)
    {
      why = "'" + tok + "': unknown backend '" + backendPart + "'";
      return false;
    }
    sel.backend = row->info.id;
    requested.set(static_cast<size_t>(row->info.id));
    const std::string indexPart = tok.substr(colon + 1);
    if (indexPart.empty() || !parseIntArg(indexPart.c_str(), sel.index))
    {
      why = "'" + tok + "': expected backend:index, as --list-devices prints it";
      return false;
    }
    parsed.push_back(sel);
  }
  if (parsed.empty())  // arg was empty string
  {
    why = "empty list";
    return false;
  }
  out.insert(out.end(), parsed.begin(), parsed.end());
  return true;
}

// ---- Parser ------------------------------------------------------------------------

// Parse outcome of the exit-free core.  parseCliOptions maps Help/Version/
// Error onto the historical print-and-exit behavior; parseCliOptionsNoExit
// surfaces them as a bool + message so embedders (clpeak_ffi) never die on
// a bad argv.
enum class ParseResult { Ok, Help, Version, Error };

static const char *nextArg(int argc, char **argv, int &i)
{
  if (i + 1 >= argc)
    return nullptr;
  return argv[++i];
}

static ParseResult missingArg(std::string &err, const char *flag)
{
  err = std::string("clpeak: missing argument for ") + flag + "\n";
  return ParseResult::Error;
}

static ParseResult invalidValue(std::string &err, const char *flag, const char *v)
{
  err = std::string("clpeak: invalid value for ") + flag + ": " + v + "\n";
  return ParseResult::Error;
}

// Split "--<name>" / "--no-<name>" into the name and its polarity.  Returns
// false for anything that is not a long flag.
static bool splitSelectionFlag(const char *flag, std::string &name, bool &negated)
{
  if (flag[0] != '-' || flag[1] != '-') return false;
  const char *body = flag + 2;
  if (strncmp(body, "no-", 3) == 0)
  {
    name    = body + 3;
    negated = true;
  }
  else
  {
    name    = body;
    negated = false;
  }
  return !name.empty();
}

// One allow-list flip, shared by backends, categories and tests: the first
// positive flag clears the set and switches to "only what is named";
// --no-<x> always subtracts.
template <size_t N>
static void applySelection(std::bitset<N> &set, size_t bit, bool negated, bool &forced)
{
  if (negated)
  {
    set.reset(bit);
    return;
  }
  if (!forced)
  {
    set.reset();
    forced = true;
  }
  set.set(bit);
}

static ParseResult parseCore(int argc, char **argv, CliOptions &out,
                             std::string &err)
{
  bool forcedBackends   = false;
  bool forcedTests      = false;
  bool forcedCategories = false;

  for (int i = 1; i < argc; i++)
  {
    const char *a = argv[i];

    // ---- help / version / modes --------------------------------------------
    if (!strcmp(a, "-h") || !strcmp(a, "--help"))
      return ParseResult::Help;
    if (!strcmp(a, "-v") || !strcmp(a, "--version"))
      return ParseResult::Version;
    if (!strcmp(a, "--verbose"))      { out.verbose     = true; continue; }
    if (!strcmp(a, "--describe"))     { out.describe    = true; continue; }
    if (!strcmp(a, "--list-devices")) { out.listDevices = true; continue; }

    // ---- iters / warmup / budgets --------------------------------------------
    if (!strcmp(a, "-i") || !strcmp(a, "--iters"))
    {
      const char *v = nextArg(argc, argv, i);
      if (!v)
        return missingArg(err, a);
      unsigned int parsed;
      if (!parseUIntArg(v, parsed, /*allowZero=*/false))
        return invalidValue(err, a, v);
      out.forceIters = true;
      out.iters = parsed;
      continue;
    }
    if (!strcmp(a, "-w") || !strcmp(a, "--warmup"))
    {
      const char *v = nextArg(argc, argv, i);
      if (!v)
        return missingArg(err, a);
      unsigned int parsed;
      if (!parseUIntArg(v, parsed))
        return invalidValue(err, a, v);
      out.warmupCount = parsed;
      continue;
    }
    if (!strcmp(a, "--max-time") || !strcmp(a, "--max-time-cpu"))
    {
      const char *v = nextArg(argc, argv, i);
      if (!v)
        return missingArg(err, a);
      unsigned int parsed;
      if (!parseUIntArg(v, parsed, /*allowZero=*/false) ||
          parsed > std::numeric_limits<unsigned int>::max() / 1000u)
        return invalidValue(err, a, v);
      if (!strcmp(a, "--max-time"))
        out.targetTimeUs = parsed * 1000u;    // ms -> us
      else
        out.targetTimeUsCpu = parsed * 1000u;
      continue;
    }

    // ---- devices ---------------------------------------------------------------
    if (!strcmp(a, "--device"))
    {
      const char *v = nextArg(argc, argv, i);
      if (!v)
        return missingArg(err, a);
      // Naming a device is asking for its backend, so a backend this build
      // lacks gets reported (CliOptions::requestedButNotBuilt) rather than
      // silently running nothing.
      std::string why;
      if (!parseDeviceList(v, out.devices, out.requestedBackends, why))
      {
        err = std::string("clpeak: invalid --device ") + why + "\n";
        return ParseResult::Error;
      }
      continue;
    }

    // ---- output / compare / runtime ---------------------------------------------
    if (!strcmp(a, "-o") || !strcmp(a, "--output"))
    {
      const char *v = nextArg(argc, argv, i);
      if (!v)
        return missingArg(err, a);
      out.outputFile   = v;
      out.enableOutput = true;
      continue;
    }
    if (!strcmp(a, "--compare"))
    {
      const char *v = nextArg(argc, argv, i);
      if (!v)
        return missingArg(err, a);
      out.compareFile = v;
      continue;
    }
    if (!strcmp(a, "--onnx-lib"))
    {
      const char *v = nextArg(argc, argv, i);
      if (!v)
        return missingArg(err, a);
      out.onnxLibPath = v;
      continue;
    }

    // ---- backend / category / test selection -------------------------------------
    std::string name;
    bool negated = false;
    if (splitSelectionFlag(a, name, negated))
    {
      if (const BackendRow *row = findBackendFlag(name))
      {
        const size_t bit = static_cast<size_t>(row->info.id);
        applySelection(out.enabledBackends, bit, negated, forcedBackends);
        if (!negated)
          out.requestedBackends.set(bit);
        continue;
      }
      bool matched = false;
      for (int t = 0; t < numCategoryFlags && !matched; t++)
        if (name == categoryFlags[t].name)
        {
          applySelection(out.enabledCategories,
                         static_cast<size_t>(categoryFlags[t].cat), negated,
                         forcedCategories);
          matched = true;
        }
      for (int t = 0; t < numTestFlags && !matched; t++)
        if (name == testFlags[t].name)
        {
          applySelection(out.enabledTests,
                         static_cast<size_t>(testFlags[t].test), negated,
                         forcedTests);
          matched = true;
        }
      if (matched)
        continue;
    }

    err = std::string("clpeak: unknown option '") + a + "'\n";
    return ParseResult::Error;
  }

  // `--device cuda:0 --vulkan` selects nothing: every device named is on a
  // backend the backend flags switched off.  Contradictory, so say so.
  if (!out.devices.empty())
  {
    bool any = false;
    for (const DeviceSelector &sel : out.devices)
      if (out.enabledBackends.test(static_cast<size_t>(sel.backend)))
        any = true;
    if (!any)
    {
      err = "clpeak: --device names only devices of backends that are switched off\n";
      return ParseResult::Error;
    }
  }

  return ParseResult::Ok;
}

int parseCliOptions(int argc, char **argv, CliOptions &out)
{
  std::string err;
  switch (parseCore(argc, argv, out, err))
  {
  case ParseResult::Ok:
    break;
  case ParseResult::Help:
    printHelpAndExit(0);
    break;
  case ParseResult::Version:
    std::cout << "clpeak version: " << CLPEAK_VERSION_STR << "\n";
    std::exit(0);
    break;
  case ParseResult::Error:
    std::cerr << err;
    printHelpAndExit(-1);
    break;
  }
  return 0;
}

bool parseCliOptionsNoExit(int argc, char **argv, CliOptions &out,
                           std::string &errorMsg)
{
  switch (parseCore(argc, argv, out, errorMsg))
  {
  case ParseResult::Ok:
    return true;
  case ParseResult::Help:
    errorMsg = "clpeak: --help is not available in embedded mode\n";
    return false;
  case ParseResult::Version:
    errorMsg = "clpeak: --version is not available in embedded mode\n";
    return false;
  case ParseResult::Error:
  default:
    return false;
  }
}

// ---- Invocation record ----------------------------------------------------

Invocation invocationFrom(const CliOptions &opts, int argc, char **argv)
{
  Invocation inv;

  for (int i = 0; i < argc; i++)
    if (argv[i]) inv.argv.push_back(argv[i]);

  inv.targetTimeUs    = opts.targetTimeUs;
  inv.targetTimeUsCpu = opts.targetTimeUsCpu;
  inv.warmup          = opts.warmupCount;
  // Only when pinned with -i.  Left at 0 the run calibrated each test to a
  // time budget instead, which is the normal mode and the comparable one.
  inv.iters           = opts.forceIters ? opts.iters : 0;

  for (int c = 0; c < numCategoryFlags; c++)
    if (opts.enabledCategories.test(static_cast<size_t>(categoryFlags[c].cat)))
      inv.categories.push_back(categoryString(categoryFlags[c].cat));

  // Recorded only when the run was narrowed.  A full run would list every
  // test there is, which says nothing.
  if (!opts.enabledTests.all())
    for (int t = 0; t < numTestFlags; t++)
      if (opts.enabledTests.test(static_cast<size_t>(testFlags[t].test)))
        inv.tests.push_back(testFlags[t].name);

  return inv;
}
