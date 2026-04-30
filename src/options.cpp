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
// as the runtime so the help reflects the binary's actual capability.
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
    "\n BACKEND SELECTION:"
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
    "\n TEST SELECTION (default: run every test the backend supports):"
    "\n  --compute-sp                single-precision compute"
    "\n  --compute-dp                double-precision compute"
    "\n  --compute-hp                half-precision compute"
    "\n  --compute-mp                mixed-precision (fp16xfp16+fp32) compute"
    "\n  --compute-bf16              BF16 compute"
    "\n  --compute-int8-dp           INT8 dot-product (DP4a) compute"
    "\n  --compute-int4-packed       packed INT4 compute (emulated)"
    "\n  --compute-integer           integer compute                 [OpenCL]"
    "\n  --compute-intfast           integer 24-bit compute          [OpenCL]"
    "\n  --compute-char              integer 8-bit compute           [OpenCL]"
    "\n  --compute-short             integer 16-bit compute          [OpenCL]"
    "\n  --global-bandwidth          global memory bandwidth"
    "\n  --local-bandwidth           local memory bandwidth"
    "\n  --image-bandwidth           image (texture) bandwidth"
    "\n  --transfer-bandwidth        host<->device transfer bandwidth"
    "\n  --atomic-throughput         atomic throughput"
    "\n  --kernel-latency            kernel launch latency"
#ifdef ENABLE_VULKAN
    "\n  --coop-matrix               cooperative-matrix tensor cores [Vulkan]"
#endif
#ifdef ENABLE_CUDA
    "\n  --wmma                      WMMA tensor cores               [CUDA]"
#endif
#ifdef ENABLE_METAL
    "\n  --simdgroup-matrix          simdgroup_matrix tensor engine  [Metal]"
#endif
    "\n"
    "\n OPENCL-SPECIFIC:"
    "\n  --use-event-timer           time using cl events instead of std chrono"
    "\n";

// Map from CLI flag to Benchmark enum.  Used both by the parser and (via
// a count) by callers that want to know how many test-selection flags exist.
struct TestFlag {
  const char *flag;
  Benchmark   test;
};

static const TestFlag testFlags[] = {
  {"--global-bandwidth",    Benchmark::GlobalBW},
  {"--local-bandwidth",     Benchmark::LocalBW},
  {"--image-bandwidth",     Benchmark::ImageBW},
  {"--transfer-bandwidth",  Benchmark::TransferBW},
  {"--atomic-throughput",   Benchmark::AtomicThroughput},
  {"--kernel-latency",      Benchmark::KernelLatency},
  {"--compute-hp",          Benchmark::ComputeHP},
  {"--compute-mp",          Benchmark::ComputeMP},
  {"--compute-sp",          Benchmark::ComputeSP},
  {"--compute-dp",          Benchmark::ComputeDP},
  {"--compute-integer",     Benchmark::ComputeInt},
  {"--compute-intfast",     Benchmark::ComputeIntFast},
  {"--compute-char",        Benchmark::ComputeChar},
  {"--compute-short",       Benchmark::ComputeShort},
  {"--compute-int8-dp",     Benchmark::ComputeInt8DP},
  {"--compute-int4-packed", Benchmark::ComputeInt4Packed},
  {"--compute-bf16",        Benchmark::ComputeBF16},
  {"--coop-matrix",         Benchmark::CoopMatrix},
  {"--wmma",                Benchmark::Wmma},
  {"--simdgroup-matrix",    Benchmark::SimdgroupMatrix},
};
static const int numTestFlags = sizeof(testFlags) / sizeof(testFlags[0]);

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

// Require the next argument to exist; caller provides the flag name for the
// error message.
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

int parseCliOptions(int argc, char **argv, CliOptions &out)
{
  for (int i = 1; i < argc; i++)
  {
    const char *a = argv[i];

    // ---- help / version ----------------------------------------------------
    if (!strcmp(a, "-h") || !strcmp(a, "--help"))
    {
      printHelpAndExit(0);
    }
    else if (!strcmp(a, "-v") || !strcmp(a, "--version"))
    {
      std::cout << "clpeak version: " << CLPEAK_VERSION_STR << "\n";
      std::exit(0);
    }
    // ---- backend skip -----------------------------------------------------
    else if (!strcmp(a, "--no-opencl")) out.skipOpenCL = true;
    else if (!strcmp(a, "--no-vulkan")) out.skipVulkan = true;
    else if (!strcmp(a, "--no-cuda"))   out.skipCuda   = true;
    else if (!strcmp(a, "--no-metal"))  out.skipMetal  = true;

    // ---- iters / warmup ---------------------------------------------------
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

    // ---- OpenCL device selection (canonical + deprecated aliases) ---------
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

    // ---- Per-backend device selection -------------------------------------
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

    // ---- OpenCL-specific timer --------------------------------------------
    else if (!strcmp(a, "--use-event-timer"))
    {
      out.useEventTimer = true;
    }

    // ---- Modes ------------------------------------------------------------
    else if (!strcmp(a, "--list-devices"))
    {
      out.listDevices = true;
    }

    // ---- Output -----------------------------------------------------------
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

    // ---- Test selection ---------------------------------------------------
    else
    {
      bool matched = false;
      for (int t = 0; t < numTestFlags; t++)
      {
        if (!strcmp(a, testFlags[t].flag))
        {
          if (!out.forcedTests)
          {
            out.enabledTests.reset();
            out.forcedTests = true;
          }
          out.enabledTests.set(static_cast<size_t>(testFlags[t].test));
          matched = true;
          break;
        }
      }
      if (matched) continue;

      // Drop --testName (was parsed by OpenCL, never plumbed end-to-end).
      if (!strcmp(a, "-tn") || !strcmp(a, "--testName"))
      {
        std::cerr << "warning: " << a
                  << " is deprecated and ignored; use individual --compute-* flags\n";
        // consume value if present
        if (i + 1 < argc && argv[i + 1][0] != '-') i++;
        continue;
      }

      std::cerr << "clpeak: unknown option '" << a << "'\n";
      printHelpAndExit(-1);
    }
  }

  return 0;
}
