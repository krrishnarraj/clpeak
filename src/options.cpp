#include <options.h>
#include <clpeak.h>
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
    "\n  -i, --iters num             number of iterations per kernel (default: CPU=10, GPU=30)"
    "\n  -w, --warmup num            number of warm-up kernel runs before timing (default: 2)"
    "\n  --list-devices              list available devices for every backend and exit"
    "\n  --xml-file file             save results to an XML file"
    "\n  --json-file file            save results to a JSON file"
    "\n  --csv-file file             save results to a CSV file"
    "\n  --compare file              compare results against a baseline (JSON / CSV / XML)"
    "\n"
    "\n BACKEND SELECTION (default: run every available backend):"
    "\n  --opencl                    run only the OpenCL backend"
#ifdef ENABLE_VULKAN
    "\n  --vulkan                    run only the Vulkan backend"
#endif
#ifdef ENABLE_CUDA
    "\n  --cuda                      run only the CUDA backend"
#endif
#ifdef ENABLE_METAL
    "\n  --metal                     run only the Metal backend"
#endif
    "\n  (multiple --<backend> flags can be combined: --cuda --vulkan)"
#if defined(ENABLE_VULKAN) || defined(ENABLE_CUDA) || defined(ENABLE_METAL)
    "\n  --no-opencl                 skip the OpenCL backend"
#endif
#ifdef ENABLE_VULKAN
    "\n  --no-vulkan                 skip the Vulkan backend"
#endif
#ifdef ENABLE_CUDA
    "\n  --no-cuda                   skip the CUDA backend"
#endif
#ifdef ENABLE_METAL
    "\n  --no-metal                  skip the Metal backend"
#endif
    "\n"
    "\n DEVICE SELECTION:"
    "\n  --cl-platform num           OpenCL platform index (0-based)"
    "\n  --cl-device num             OpenCL device index within the platform"
    "\n  --cl-platform-name str      match OpenCL platform by name"
    "\n  --cl-device-name str        match OpenCL device by name"
#ifdef ENABLE_VULKAN
    "\n  --vk-device num             Vulkan physical-device index (0-based)"
#endif
#ifdef ENABLE_CUDA
    "\n  --cuda-device num           CUDA device ordinal (0-based)"
#endif
#ifdef ENABLE_METAL
    "\n  --mtl-device num            Metal device index (0-based)"
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
    "\n  --integer-compute-fast            | --no-integer-compute-fast      [OpenCL]"
    "\n  --integer-compute-char            | --no-integer-compute-char      [OpenCL]"
    "\n  --integer-compute-short           | --no-integer-compute-short     [OpenCL]"
    "\n  --int8-dot-product-compute        | --no-int8-dot-product-compute"
    "\n  --int4-packed-compute             | --no-int4-packed-compute"
#ifdef ENABLE_CUDA
    "\n  --wmma                            | --no-wmma                      [CUDA]"
    "\n  --bmma                            | --no-bmma                      [CUDA]"
    "\n  --cublas                          | --no-cublas                    [CUDA]"
#endif
#ifdef ENABLE_VULKAN
    "\n  --coopmat                         | --no-coopmat                   [Vulkan]"
#endif
#ifdef ENABLE_METAL
    "\n  --simdgroup-matrix                | --no-simdgroup-matrix          [Metal]"
    "\n  --mps-gemm                        | --no-mps-gemm                  [Metal]"
#endif
    "\n  --global-memory-bandwidth         | --no-global-memory-bandwidth"
    "\n  --local-memory-bandwidth          | --no-local-memory-bandwidth"
    "\n  --image-memory-bandwidth          | --no-image-memory-bandwidth"
    "\n  --transfer-bandwidth              | --no-transfer-bandwidth"
    "\n  --atomic-throughput               | --no-atomic-throughput"
    "\n  --kernel-launch-latency           | --no-kernel-launch-latency"
    "\n"
    "\n OPENCL-SPECIFIC:"
    "\n  --use-event-timer           time using cl events instead of std chrono"
    "\n";

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
  {"integer-compute-fast",      Benchmark::ComputeIntFast},
  {"integer-compute-char",      Benchmark::ComputeChar},
  {"integer-compute-short",     Benchmark::ComputeShort},
  {"int8-dot-product-compute",  Benchmark::ComputeInt8DP},
  {"int4-packed-compute",       Benchmark::ComputeInt4Packed},
  {"wmma",                      Benchmark::Wmma},
  {"bmma",                      Benchmark::Bmma},
  {"coopmat",                   Benchmark::CoopMatrix},
  {"simdgroup-matrix",          Benchmark::SimdgroupMatrix},
  {"cublas",                    Benchmark::Cublas},
  {"mps-gemm",                  Benchmark::MpsGemm},
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

static const char *requireArg(int argc, char **argv, int &i, const char *flag)
{
  if (i + 1 >= argc)
  {
    std::cerr << "clpeak: missing argument for " << flag << "\n";
    printHelpAndExit(-1);
  }
  return argv[++i];
}

static void deprecate(const char *oldFlag, const char *newFlag)
{
  std::cerr << "warning: " << oldFlag
            << " is deprecated; use " << newFlag << "\n";
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
static void applyTestFlag(CliOptions &out, Benchmark b, bool negated)
{
  if (negated)
  {
    out.enabledTests.reset(static_cast<size_t>(b));
    return;
  }
  if (!out.forcedTests)
  {
    out.enabledTests.reset();
    out.forcedTests = true;
  }
  out.enabledTests.set(static_cast<size_t>(b));
}

static void applyCategoryFlag(CliOptions &out, Category c, bool negated)
{
  if (negated)
  {
    out.enabledCategories.reset(static_cast<size_t>(c));
    return;
  }
  if (!out.forcedCategories)
  {
    out.enabledCategories.reset();
    out.forcedCategories = true;
  }
  out.enabledCategories.set(static_cast<size_t>(c));
}

int parseCliOptions(int argc, char **argv, CliOptions &out)
{
  // Positive backend includes.  When any --<backend> flag is present, only
  // listed backends run; everything else gets skipped at the end of parsing.
  bool includeAny = false;
  bool incOpenCL = false, incVulkan = false, incCuda = false, incMetal = false;

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
    // ---- backend selection ----------------------------------------------
    else if (!strcmp(a, "--no-opencl")) out.skipOpenCL = true;
    else if (!strcmp(a, "--no-vulkan")) out.skipVulkan = true;
    else if (!strcmp(a, "--no-cuda"))   out.skipCuda   = true;
    else if (!strcmp(a, "--no-metal"))  out.skipMetal  = true;
    else if (!strcmp(a, "--opencl"))    { incOpenCL = true; includeAny = true; }
    else if (!strcmp(a, "--vulkan"))    { incVulkan = true; includeAny = true; }
    else if (!strcmp(a, "--cuda"))      { incCuda   = true; includeAny = true; }
    else if (!strcmp(a, "--metal"))     { incMetal  = true; includeAny = true; }

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

    // ---- OpenCL device selection (canonical + deprecated aliases) -------
    else if (!strcmp(a, "--cl-platform") ||
             !strcmp(a, "-p") || !strcmp(a, "--platform"))
    {
      if (strcmp(a, "--cl-platform")) deprecate(a, "--cl-platform");
      const char *v = requireArg(argc, argv, i, a);
      if (!parseUnsignedLongArg(v, out.platformIndex))
      {
        std::cerr << "clpeak: invalid platform index: " << v << "\n";
        printHelpAndExit(-1);
      }
      out.forcePlatform = true;
    }
    else if (!strcmp(a, "--cl-device") ||
             !strcmp(a, "-d") || !strcmp(a, "--device"))
    {
      if (strcmp(a, "--cl-device")) deprecate(a, "--cl-device");
      const char *v = requireArg(argc, argv, i, a);
      if (!parseUnsignedLongArg(v, out.deviceIndex))
      {
        std::cerr << "clpeak: invalid device index: " << v << "\n";
        printHelpAndExit(-1);
      }
      out.forceDevice = true;
    }
    else if (!strcmp(a, "--cl-platform-name") ||
             !strcmp(a, "-pn") || !strcmp(a, "--platformName"))
    {
      if (strcmp(a, "--cl-platform-name")) deprecate(a, "--cl-platform-name");
      out.platformName = requireArg(argc, argv, i, a);
      out.forcePlatformName = true;
    }
    else if (!strcmp(a, "--cl-device-name") ||
             !strcmp(a, "-dn") || !strcmp(a, "--deviceName"))
    {
      if (strcmp(a, "--cl-device-name")) deprecate(a, "--cl-device-name");
      out.deviceName = requireArg(argc, argv, i, a);
      out.forceDeviceName = true;
    }

    // ---- Per-backend device selection -----------------------------------
    else if (!strcmp(a, "--vk-device"))
    {
      const char *v = requireArg(argc, argv, i, a);
      if (!parseIntArg(v, out.vkDeviceIndex))
      {
        std::cerr << "clpeak: invalid Vulkan device index: " << v << "\n";
        printHelpAndExit(-1);
      }
    }
    else if (!strcmp(a, "--cuda-device"))
    {
      const char *v = requireArg(argc, argv, i, a);
      if (!parseIntArg(v, out.cudaDeviceIndex))
      {
        std::cerr << "clpeak: invalid CUDA device index: " << v << "\n";
        printHelpAndExit(-1);
      }
    }
    else if (!strcmp(a, "--mtl-device"))
    {
      const char *v = requireArg(argc, argv, i, a);
      if (!parseIntArg(v, out.mtlDeviceIndex))
      {
        std::cerr << "clpeak: invalid Metal device index: " << v << "\n";
        printHelpAndExit(-1);
      }
    }

    // ---- OpenCL-specific timer ------------------------------------------
    else if (!strcmp(a, "--use-event-timer"))
    {
      out.useEventTimer = true;
    }

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
          applyCategoryFlag(out, categoryFlags[t].cat, negated);
          matched = true;
        }
      }

      for (int t = 0; t < numTestFlags && !matched; t++)
      {
        if (matchFlag(a, testFlags[t].name, negated))
        {
          applyTestFlag(out, testFlags[t].test, negated);
          matched = true;
        }
      }

      if (matched) continue;

      // Drop --testName (was parsed by OpenCL, never plumbed end-to-end).
      if (!strcmp(a, "-tn") || !strcmp(a, "--testName"))
      {
        std::cerr << "warning: " << a
                  << " is deprecated and ignored; use individual --<test> flags\n";
        if (i + 1 < argc && argv[i + 1][0] != '-') i++;
        continue;
      }

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
    if (!incMetal)  out.skipMetal  = true;
  }

  return 0;
}
