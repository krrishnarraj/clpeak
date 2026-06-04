#include <common/options.h>
#include <common/benchmark_enums.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <version.h>

// Help text.  Backend-specific flags are gated by the same ENABLE_* macros
// as the runtime so the help reflects the binary's actual capability.  v2
// flag surface: 4 category flags, canonical test names, --no-<x> for both.
static const char *helpStr =
    "\n clpeak [OPTIONS]"
    "\n"
    "\n GLOBAL OPTIONS:"
    "\n  -h, --help                  display help message"
    "\n  -v, --version               display version"
    "\n  -i, --iters num             force a fixed iter count (overrides --max-time calibration)"
    "\n  -w, --warmup num            number of warm-up kernel runs before timing (default: 2)"
    "\n  --max-time ms               per-test time budget for the timed phase (default: 500 ms)"
    "\n                              picks iters automatically; set lower if you hit a GPU watchdog"
    "\n  --verbose                   print backend debug logs (kernel build logs, API errors)"
    "\n  --list-devices              list available devices for every backend and exit"
    "\n  --xml-file file             save results to an XML file"
    "\n  --json-file file            save results to a JSON file"
    "\n  --csv-file file             save results to a CSV file"
    "\n  --compare file              compare results against a baseline (JSON / CSV / XML)"
    "\n"
    "\n BACKEND SELECTION (default: run every available backend):"
#ifdef ENABLE_OPENCL
    "\n  --opencl                    run only the OpenCL backend"
#endif
#ifdef ENABLE_VULKAN
    "\n  --vulkan                    run only the Vulkan backend"
#endif
#ifdef ENABLE_CUDA
    "\n  --cuda                      run only the CUDA backend"
#endif
#ifdef ENABLE_ROCM
    "\n  --rocm                      run only the ROCm/HIP backend"
#endif
#ifdef ENABLE_METAL
    "\n  --metal                     run only the Metal backend"
#endif
#ifdef ENABLE_ONEAPI
    "\n  --oneapi                    run only the oneAPI/SYCL backend"
#endif
    "\n  (multiple --<backend> flags can be combined)"
#ifdef ENABLE_OPENCL
    "\n  --no-opencl                 skip the OpenCL backend"
#endif
#ifdef ENABLE_VULKAN
    "\n  --no-vulkan                 skip the Vulkan backend"
#endif
#ifdef ENABLE_CUDA
    "\n  --no-cuda                   skip the CUDA backend"
#endif
#ifdef ENABLE_ROCM
    "\n  --no-rocm                   skip the ROCm/HIP backend"
#endif
#ifdef ENABLE_METAL
    "\n  --no-metal                  skip the Metal backend"
#endif
#ifdef ENABLE_ONEAPI
    "\n  --no-oneapi                 skip the oneAPI/SYCL backend"
#endif
    "\n"
    "\n DEVICE SELECTION (indices are 0-based; comma-separated for multiple,"
    "\n default: run every device):"
#ifdef ENABLE_OPENCL
    "\n  --cl-platform list          OpenCL platform index/indices (e.g. 0 or 0,1)"
    "\n  --cl-device list            OpenCL device index/indices within the platform"
#endif
#ifdef ENABLE_VULKAN
    "\n  --vk-device list            Vulkan physical-device index/indices"
#endif
#ifdef ENABLE_CUDA
    "\n  --cuda-device list          CUDA device ordinal(s) (e.g. 0 or 0,2)"
#endif
#ifdef ENABLE_ROCM
    "\n  --rocm-device list          ROCm/HIP device ordinal(s)"
#endif
#ifdef ENABLE_METAL
    "\n  --mtl-device list           Metal device index/indices"
#endif
#ifdef ENABLE_ONEAPI
    "\n  --oneapi-device list        oneAPI/SYCL device index/indices"
#endif
    "\n"
    "\n TEST CATEGORY SELECTION (default: run every category):"
    "\n  --fp-compute / --no-fp-compute       floating-point compute (gflops / tflops)"
    "\n  --int-compute / --no-int-compute     integer compute (gops / tops)"
    "\n  --bandwidth   / --no-bandwidth       memory & transfer bandwidth (gbps)"
    "\n  --latency     / --no-latency         kernel-launch latency (us)"
    "\n  Any positive --<category> flag switches to allow-list mode."
    "\n"
    "\n TEST SELECTION (default: every test the backend supports;"
    "\n any positive --<test> flag switches to allow-list mode;"
    "\n --no-<test> always subtracts):"
    "\n  --single-precision-compute        | --no-single-precision-compute"
    "\n  --half-precision-compute          | --no-half-precision-compute"
    "\n  --double-precision-compute        | --no-double-precision-compute"
    "\n  --mixed-precision-compute         | --no-mixed-precision-compute"
    "\n  --bfloat16-compute                | --no-bfloat16-compute"
    "\n  --integer-compute                 | --no-integer-compute"
#ifdef ENABLE_OPENCL
    "\n  --integer-compute-fast            | --no-integer-compute-fast      [OpenCL]"
    "\n  --integer-compute-char            | --no-integer-compute-char      [OpenCL]"
    "\n  --integer-compute-short           | --no-integer-compute-short     [OpenCL]"
#endif
    "\n  --int8-dot-product-compute        | --no-int8-dot-product-compute"
#ifdef ENABLE_CUDA
    "\n  --wmma                            | --no-wmma                      [CUDA]"
    "\n  --bmma                            | --no-bmma                      [CUDA]"
    "\n  --cublas                          | --no-cublas                    [CUDA]"
#endif
#ifdef ENABLE_ROCM
    "\n  --rocwmma                         | --no-rocwmma                   [ROCm]"
    "\n  --mfma                            | --no-mfma                      [ROCm]"
    "\n  --rocblas                         | --no-rocblas                   [ROCm]"
#endif
#ifdef ENABLE_VULKAN
    "\n  --coopmat                         | --no-coopmat                   [Vulkan]"
#endif
#ifdef ENABLE_METAL
    "\n  --simdgroup-matrix                | --no-simdgroup-matrix          [Metal]"
    "\n  --mps-gemm                        | --no-mps-gemm                  [Metal]"
#endif
#ifdef ENABLE_ONEAPI
    "\n  --joint-matrix                    | --no-joint-matrix              [oneAPI]"
    "\n  --onemkl                          | --no-onemkl                    [oneAPI]"
#endif
    "\n  --global-memory-bandwidth         | --no-global-memory-bandwidth"
    "\n  --local-memory-bandwidth          | --no-local-memory-bandwidth"
    "\n  --image-memory-bandwidth          | --no-image-memory-bandwidth"
    "\n  --transfer-bandwidth              | --no-transfer-bandwidth"
    "\n  --atomic-throughput               | --no-atomic-throughput"
    "\n  --kernel-launch-latency           | --no-kernel-launch-latency"
    "\n"
#ifdef ENABLE_OPENCL
    "\n OPENCL-SPECIFIC:"
    "\n  --use-event-timer           time using cl events instead of std chrono"
    "\n"
#endif
;

// ---- Flag tables ----------------------------------------------------------

struct TestFlag {
  const char *name;        // flag suffix; e.g. "wmma" matches --wmma / --no-wmma
  Benchmark   test;
};

static const TestFlag testFlags[] = {
  {"single-precision-compute",  Benchmark::ComputeSP},
  {"half-precision-compute",    Benchmark::ComputeHP},
  {"double-precision-compute",  Benchmark::ComputeDP},
  {"mixed-precision-compute",   Benchmark::ComputeMP},
  {"bfloat16-compute",          Benchmark::ComputeBF16},
  {"integer-compute",           Benchmark::ComputeInt},
#ifdef ENABLE_OPENCL
  {"integer-compute-fast",      Benchmark::ComputeIntFast},
  {"integer-compute-char",      Benchmark::ComputeChar},
  {"integer-compute-short",     Benchmark::ComputeShort},
#endif
  {"int8-dot-product-compute",  Benchmark::ComputeInt8DP},
#ifdef ENABLE_CUDA
  {"wmma",                      Benchmark::Wmma},
  {"bmma",                      Benchmark::Bmma},
#endif
#ifdef ENABLE_VULKAN
  {"coopmat",                   Benchmark::CoopMatrix},
#endif
#ifdef ENABLE_METAL
  {"simdgroup-matrix",          Benchmark::SimdgroupMatrix},
#endif
#ifdef ENABLE_CUDA
  {"cublas",                    Benchmark::Cublas},
#endif
#ifdef ENABLE_ROCM
  {"rocwmma",                   Benchmark::Rocwmma},
  {"mfma",                      Benchmark::Mfma},
  {"rocblas",                   Benchmark::Rocblas},
#endif
#ifdef ENABLE_METAL
  {"mps-gemm",                  Benchmark::MpsGemm},
#endif
#ifdef ENABLE_ONEAPI
  {"joint-matrix",              Benchmark::JointMatrix},
  {"onemkl",                    Benchmark::Onemkl},
#endif
  {"global-memory-bandwidth",   Benchmark::GlobalBW},
  {"local-memory-bandwidth",    Benchmark::LocalBW},
  {"image-memory-bandwidth",    Benchmark::ImageBW},
  {"transfer-bandwidth",        Benchmark::TransferBW},
  {"atomic-throughput",         Benchmark::AtomicThroughput},
  {"kernel-launch-latency",     Benchmark::KernelLatency},
};
static const int numTestFlags = sizeof(testFlags) / sizeof(testFlags[0]);

struct CategoryFlag {
  const char *name;
  Category    cat;
};

static const CategoryFlag categoryFlags[] = {
  {"fp-compute",  Category::FpCompute},
  {"int-compute", Category::IntCompute},
  {"bandwidth",   Category::Bandwidth},
  {"latency",     Category::Latency},
};
static const int numCategoryFlags = sizeof(categoryFlags) / sizeof(categoryFlags[0]);

// ---- Helpers --------------------------------------------------------------

static void printHelpAndExit(int code)
{
  std::cout << helpStr << "\n";
  std::cout.flush();
  std::exit(code);
}

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

// Parse a comma-separated list of indices (single value = list of one).  Each
// token must be a valid non-negative index; empty tokens (e.g. "0,,2") fail.
static bool parseIndexList(const char *arg, std::vector<unsigned long> &out)
{
  std::vector<unsigned long> parsed;
  std::stringstream ss(arg);
  std::string tok;
  while (std::getline(ss, tok, ','))
  {
    unsigned long v;
    if (tok.empty() || !parseUnsignedLongArg(tok.c_str(), v))
      return false;
    parsed.push_back(v);
  }
  if (parsed.empty())  // arg was empty string
    return false;
  out = std::move(parsed);
  return true;
}

static bool parseIndexList(const char *arg, std::vector<int> &out)
{
  std::vector<int> parsed;
  std::stringstream ss(arg);
  std::string tok;
  while (std::getline(ss, tok, ','))
  {
    int v;
    if (tok.empty() || !parseIntArg(tok.c_str(), v))
      return false;
    parsed.push_back(v);
  }
  if (parsed.empty())
    return false;
  out = std::move(parsed);
  return true;
}

static const char *requireArg(int argc, char **argv, int &i, const char *flag)
{
  if (i + 1 >= argc)
  {
    std::cerr << "clpeak: missing argument for " << flag << "\n";
    printHelpAndExit(-1);
  }
  return argv[++i];
}

// Return true if `flag` matches "--<name>" or "--no-<name>".  In the latter
// case `out_negated` is set; otherwise it's cleared.
static bool matchFlag(const char *flag, const char *name, bool &out_negated)
{
  // strip leading '--'
  if (flag[0] != '-' || flag[1] != '-') return false;
  const char *body = flag + 2;
  if (strncmp(body, "no-", 3) == 0)
  {
    if (strcmp(body + 3, name) == 0) { out_negated = true; return true; }
    return false;
  }
  if (strcmp(body, name) == 0) { out_negated = false; return true; }
  return false;
}

// Apply one test-selection flip.  Honours allow-list semantics.
static void applyTestFlag(CliOptions &out, Benchmark b, bool negated, bool &forcedTests)
{
  if (negated)
  {
    out.enabledTests.reset(static_cast<size_t>(b));
    return;
  }
  if (!forcedTests)
  {
    out.enabledTests.reset();
    forcedTests = true;
  }
  out.enabledTests.set(static_cast<size_t>(b));
}

static void applyCategoryFlag(CliOptions &out, Category c, bool negated, bool &forcedCategories)
{
  if (negated)
  {
    out.enabledCategories.reset(static_cast<size_t>(c));
    return;
  }
  if (!forcedCategories)
  {
    out.enabledCategories.reset();
    forcedCategories = true;
  }
  out.enabledCategories.set(static_cast<size_t>(c));
}

int parseCliOptions(int argc, char **argv, CliOptions &out)
{
  // Positive backend includes.  When any --<backend> flag is present, only
  // listed backends run; everything else gets skipped at the end of parsing.
  bool includeAny = false;
  bool incOpenCL = false, incVulkan = false, incCuda = false, incRocm = false, incMetal = false, incOneapi = false;
  bool forcedTests = false;
  bool forcedCategories = false;

  for (int i = 1; i < argc; i++)
  {
    const char *a = argv[i];
    bool negated = false;

    // ---- help / version ---------------------------------------------------
    if (!strcmp(a, "-h") || !strcmp(a, "--help"))
    {
      printHelpAndExit(0);
    }
    else if (!strcmp(a, "-v") || !strcmp(a, "--version"))
    {
      std::cout << "clpeak version: " << CLPEAK_VERSION_STR << "\n";
      std::exit(0);
    }
    else if (!strcmp(a, "--verbose"))
    {
      out.verbose = true;
    }
    // ---- backend selection ----------------------------------------------
#ifdef ENABLE_OPENCL
    else if (!strcmp(a, "--no-opencl")) out.skipOpenCL = true;
    else if (!strcmp(a, "--opencl"))    { incOpenCL = true; includeAny = true; }
#endif
#ifdef ENABLE_VULKAN
    else if (!strcmp(a, "--no-vulkan")) out.skipVulkan = true;
    else if (!strcmp(a, "--vulkan"))    { incVulkan = true; includeAny = true; }
#endif
#ifdef ENABLE_CUDA
    else if (!strcmp(a, "--no-cuda"))   out.skipCuda   = true;
    else if (!strcmp(a, "--cuda"))      { incCuda   = true; includeAny = true; }
#endif
#ifdef ENABLE_ROCM
    else if (!strcmp(a, "--no-rocm")) out.skipRocm = true;
    else if (!strcmp(a, "--rocm"))    { incRocm = true; includeAny = true; }
#endif
#ifdef ENABLE_METAL
    else if (!strcmp(a, "--no-metal"))  out.skipMetal  = true;
    else if (!strcmp(a, "--metal"))     { incMetal  = true; includeAny = true; }
#endif
#ifdef ENABLE_ONEAPI
    else if (!strcmp(a, "--no-oneapi")) out.skipOneapi = true;
    else if (!strcmp(a, "--oneapi"))    { incOneapi = true; includeAny = true; }
#endif

    // ---- iters / warmup -------------------------------------------------
    else if (!strcmp(a, "-i") || !strcmp(a, "--iters"))
    {
      const char *v = requireArg(argc, argv, i, a);
      unsigned int parsed;
      if (!parseUIntArg(v, parsed, /*allowZero=*/false))
      {
        std::cerr << "clpeak: invalid value for " << a << ": " << v << "\n";
        printHelpAndExit(-1);
      }
      out.forceIters = true;
      out.iters = parsed;
    }
    else if (!strcmp(a, "-w") || !strcmp(a, "--warmup"))
    {
      const char *v = requireArg(argc, argv, i, a);
      unsigned int parsed;
      if (!parseUIntArg(v, parsed))
      {
        std::cerr << "clpeak: invalid value for " << a << ": " << v << "\n";
        printHelpAndExit(-1);
      }
      out.warmupCount = parsed;
    }
    else if (!strcmp(a, "--max-time"))
    {
      const char *v = requireArg(argc, argv, i, a);
      unsigned int parsed;
      if (!parseUIntArg(v, parsed, /*allowZero=*/false))
      {
        std::cerr << "clpeak: invalid value for " << a << ": " << v << "\n";
        printHelpAndExit(-1);
      }
      out.targetTimeUs = parsed * 1000u; // ms -> us
    }

    // ---- OpenCL device selection ----------------------------------------
#ifdef ENABLE_OPENCL
    else if (!strcmp(a, "--cl-platform"))
    {
      const char *v = requireArg(argc, argv, i, a);
      if (!parseIndexList(v, out.platformIndices))
      {
        std::cerr << "clpeak: invalid platform index list: " << v << "\n";
        printHelpAndExit(-1);
      }
    }
    else if (!strcmp(a, "--cl-device"))
    {
      const char *v = requireArg(argc, argv, i, a);
      if (!parseIndexList(v, out.deviceIndices))
      {
        std::cerr << "clpeak: invalid device index list: " << v << "\n";
        printHelpAndExit(-1);
      }
    }
#endif

    // ---- Per-backend device selection -----------------------------------
#ifdef ENABLE_VULKAN
    else if (!strcmp(a, "--vk-device"))
    {
      const char *v = requireArg(argc, argv, i, a);
      if (!parseIndexList(v, out.vkDeviceIndices))
      {
        std::cerr << "clpeak: invalid Vulkan device index list: " << v << "\n";
        printHelpAndExit(-1);
      }
    }
#endif
#ifdef ENABLE_CUDA
    else if (!strcmp(a, "--cuda-device"))
    {
      const char *v = requireArg(argc, argv, i, a);
      if (!parseIndexList(v, out.cudaDeviceIndices))
      {
        std::cerr << "clpeak: invalid CUDA device index list: " << v << "\n";
        printHelpAndExit(-1);
      }
    }
#endif
#ifdef ENABLE_ROCM
    else if (!strcmp(a, "--rocm-device"))
    {
      const char *v = requireArg(argc, argv, i, a);
      if (!parseIndexList(v, out.rocmDeviceIndices))
      {
        std::cerr << "clpeak: invalid ROCm device index list: " << v << "\n";
        printHelpAndExit(-1);
      }
    }
#endif
#ifdef ENABLE_METAL
    else if (!strcmp(a, "--mtl-device"))
    {
      const char *v = requireArg(argc, argv, i, a);
      if (!parseIndexList(v, out.mtlDeviceIndices))
      {
        std::cerr << "clpeak: invalid Metal device index list: " << v << "\n";
        printHelpAndExit(-1);
      }
    }
#endif
#ifdef ENABLE_ONEAPI
    else if (!strcmp(a, "--oneapi-device"))
    {
      const char *v = requireArg(argc, argv, i, a);
      if (!parseIndexList(v, out.oneapiDeviceIndices))
      {
        std::cerr << "clpeak: invalid oneAPI device index list: " << v << "\n";
        printHelpAndExit(-1);
      }
    }
#endif

    // ---- OpenCL-specific timer ------------------------------------------
#ifdef ENABLE_OPENCL
    else if (!strcmp(a, "--use-event-timer"))
    {
      out.useEventTimer = true;
    }
#endif

    // ---- Modes ----------------------------------------------------------
    else if (!strcmp(a, "--list-devices"))
    {
      out.listDevices = true;
    }

    // ---- Output ---------------------------------------------------------
    else if (!strcmp(a, "--xml-file"))
    {
      out.xmlFile    = requireArg(argc, argv, i, a);
      out.enableXml  = true;
    }
    else if (!strcmp(a, "--json-file"))
    {
      out.jsonFile   = requireArg(argc, argv, i, a);
      out.enableJson = true;
    }
    else if (!strcmp(a, "--csv-file"))
    {
      out.csvFile    = requireArg(argc, argv, i, a);
      out.enableCsv  = true;
    }
    else if (!strcmp(a, "--compare"))
    {
      out.compareFile = requireArg(argc, argv, i, a);
    }

    // ---- Category / test selection --------------------------------------
    else
    {
      bool matched = false;

      for (int t = 0; t < numCategoryFlags && !matched; t++)
      {
        if (matchFlag(a, categoryFlags[t].name, negated))
        {
          applyCategoryFlag(out, categoryFlags[t].cat, negated, forcedCategories);
          matched = true;
        }
      }

      for (int t = 0; t < numTestFlags && !matched; t++)
      {
        if (matchFlag(a, testFlags[t].name, negated))
        {
          applyTestFlag(out, testFlags[t].test, negated, forcedTests);
          matched = true;
        }
      }

      if (matched) continue;

      std::cerr << "clpeak: unknown option '" << a << "'\n";
      printHelpAndExit(-1);
    }
  }

  // Apply positive backend selection: any --<backend> flag means "only run
  // the listed backends".  Skips set by --no-<backend> still apply.
  if (includeAny)
  {
    if (!incOpenCL) out.skipOpenCL = true;
    if (!incVulkan) out.skipVulkan = true;
    if (!incCuda)   out.skipCuda   = true;
    if (!incRocm)   out.skipRocm   = true;
    if (!incMetal)  out.skipMetal  = true;
    if (!incOneapi) out.skipOneapi = true;
  }

  return 0;
}
