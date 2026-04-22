#include <clpeak.h>
#include <cerrno>
#include <limits>
#include <version.h>

static const char *helpStr =
    "\n clpeak [OPTIONS]"
    "\n"
    "\n OPTIONS:"
    "\n  -p, --platform num          choose platform (num starts with 0)"
    "\n  -d, --device num            choose device (num starts with 0)"
    "\n  -pn, --platformName name    choose platform name"
    "\n  -dn, --deviceName name      choose device name"
    "\n  -tn, --testName name        choose test name"
    "\n  -i, --iters                 choose the number of iterations per kernel (default: CPU=10, GPU=30)"
    "\n  -w, --warmup num            number of warm-up kernel runs before timing (default: 2)"
    "\n  --use-event-timer           time using cl events instead of std chrono timer"
    "\n                              hide driver latencies [default: No]"
    "\n  --global-bandwidth          selectively run global bandwidth test"
    "\n  --local-bandwidth           selectively run local memory bandwidth test"
    "\n  --image-bandwidth           selectively run image (texture) bandwidth test"
    "\n  --atomic-throughput         selectively run atomic throughput test"
    "\n  --compute-hp                selectively run half precision compute test"
    "\n  --compute-mp                selectively run mixed-precision (fp16xfp16+fp32) compute test"
    "\n  --compute-sp                selectively run single precision compute test"
    "\n  --compute-dp                selectively run double precision compute test"
    "\n  --compute-integer           selectively run integer compute test"
    "\n  --compute-intfast           selectively run integer 24bit compute test"
    "\n  --compute-char              selectively run char (integer 8bit) compute test"
    "\n  --compute-short             selectively run short (integer 16bit) compute test"
    "\n  --compute-int8-dp           selectively run INT8 dot-product (DP4a) compute test"
    "\n  --transfer-bandwidth        selectively run transfer bandwidth test"
    "\n  --kernel-latency            selectively run kernel latency test"
    "\n  --all-tests                 run all above tests [default]"
    "\n  --list-devices              list available platforms/devices and exit"
#ifdef ENABLE_VULKAN
    "\n  --no-opencl                 skip the OpenCL backend (Vulkan only)"
    "\n  --no-vulkan                 skip the Vulkan backend (OpenCL only)"
#endif
    "\n  --xml-file file_name        save results to an XML file"
    "\n  --json-file file_name       save results to a JSON file"
    "\n  --csv-file file_name        save results to a CSV file"
    "\n  --compare file_name         compare results against a baseline file"
    "\n                              (supports JSON, CSV, and XML formats)"
    "\n  -v, --version               display version"
    "\n  -h, --help                  display help message"
    "\n";

static void printParseErrorAndExit()
{
  std::cout << helpStr;
  std::cout << NEWLINE;
  std::cout.flush();
  exit(-1);
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

  if (!parseUnsignedLongArg(arg, parsed) || (parsed > std::numeric_limits<unsigned int>::max()) || (!allowZero && (parsed == 0)))
    return false;

  value = static_cast<unsigned int>(parsed);
  return true;
}

static void printParseMessage(const std::string &message)
{
  std::cout << message;
  std::cout.flush();
}

// Map from CLI flag to Benchmark enum
struct TestFlag {
  const char *flag;
  Benchmark test;
};

static const TestFlag testFlags[] = {
  {"--global-bandwidth",  Benchmark::GlobalBW},
  {"--local-bandwidth",   Benchmark::LocalBW},
  {"--image-bandwidth",   Benchmark::ImageBW},
  {"--atomic-throughput",  Benchmark::AtomicThroughput},
  {"--compute-hp",        Benchmark::ComputeHP},
  {"--compute-mp",        Benchmark::ComputeMP},
  {"--compute-sp",        Benchmark::ComputeSP},
  {"--compute-dp",        Benchmark::ComputeDP},
  {"--compute-integer",   Benchmark::ComputeInt},
  {"--compute-intfast",   Benchmark::ComputeIntFast},
  {"--compute-char",      Benchmark::ComputeChar},
  {"--compute-short",     Benchmark::ComputeShort},
  {"--compute-int8-dp",   Benchmark::ComputeInt8DP},
  {"--transfer-bandwidth", Benchmark::TransferBW},
  {"--kernel-latency",    Benchmark::KernelLatency},
};
static const int numTestFlags = sizeof(testFlags) / sizeof(testFlags[0]);

int clPeak::parseArgs(int argc, char **argv)
{
  bool forcedTests = false;
  bool enableXml = false;
  std::string xmlFileName;
  std::string jsonFile;
  std::string csvFile;
  std::string compareFile;

  for (int i = 1; i < argc; i++)
  {
    if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0))
    {
      printParseMessage(helpStr);
      printParseMessage(NEWLINE);
      exit(0);
    }
    else if ((strcmp(argv[i], "-v") == 0) || (strcmp(argv[i], "--version") == 0))
    {
      std::stringstream versionStr;
      versionStr << "clpeak version: " << CLPEAK_VERSION_STR;

      printParseMessage(versionStr.str());
      printParseMessage(NEWLINE);
      exit(0);
    }
    else if ((strcmp(argv[i], "-p") == 0) || (strcmp(argv[i], "--platform") == 0))
    {
      if ((i + 1) < argc)
      {
        unsigned long parsed;

        if (!parseUnsignedLongArg(argv[i + 1], parsed))
          printParseErrorAndExit();

        forcePlatform = true;
        specifiedPlatform = parsed;
        i++;
      }
    }
    else if ((strcmp(argv[i], "-d") == 0) || (strcmp(argv[i], "--device") == 0))
    {
      if ((i + 1) < argc)
      {
        unsigned long parsed;

        if (!parseUnsignedLongArg(argv[i + 1], parsed))
          printParseErrorAndExit();

        forceDevice = true;
        specifiedDevice = parsed;
        i++;
      }
    }
    else if ((strcmp(argv[i], "-pn") == 0) || (strcmp(argv[i], "--platformName") == 0))
    {
      if ((i + 1) < argc)
      {
        forcePlatformName = true;
        specifiedPlatformName = argv[i + 1];
        i++;
      }
    }
    else if ((strcmp(argv[i], "-dn") == 0) || (strcmp(argv[i], "--deviceName") == 0))
    {
      if ((i + 1) < argc)
      {
        forceDeviceName = true;
        specifiedDeviceName = argv[i + 1];
        i++;
      }
    }
    else if ((strcmp(argv[i], "-tn") == 0) || (strcmp(argv[i], "--testName") == 0))
    {
      if ((i + 1) < argc)
      {
        forceTest = true;
        specifiedTestName = argv[i + 1];
        i++;
      }
    }
    else if ((strcmp(argv[i], "-i") == 0) || (strcmp(argv[i], "--iters") == 0))
    {
      if ((i + 1) < argc)
      {
        unsigned int parsed;

        if (!parseUIntArg(argv[i + 1], parsed, false))
          printParseErrorAndExit();

        forceIters = true;
        specifiedIters = parsed;
        i++;
      }
    }
    else if ((strcmp(argv[i], "-w") == 0) || (strcmp(argv[i], "--warmup") == 0))
    {
      if ((i + 1) < argc)
      {
        unsigned int parsed;

        if (!parseUIntArg(argv[i + 1], parsed))
          printParseErrorAndExit();

        warmupCount = parsed;
        i++;
      }
    }
    else if (strcmp(argv[i], "--use-event-timer") == 0)
    {
      useEventTimer = true;
    }
    else if (strcmp(argv[i], "--list-devices") == 0)
    {
      listDevices = true;
    }
    else if (strcmp(argv[i], "--all-tests") == 0)
    {
      enableAll();
    }
    else if (strcmp(argv[i], "--xml-file") == 0)
    {
      if ((i + 1) < argc)
      {
        enableXml = true;
        xmlFileName = argv[i + 1];
        i++;
      }
    }
    else if (strcmp(argv[i], "--json-file") == 0)
    {
      if ((i + 1) < argc)
      {
        enableJson = true;
        jsonFile = argv[i + 1];
        i++;
      }
    }
    else if (strcmp(argv[i], "--csv-file") == 0)
    {
      if ((i + 1) < argc)
      {
        enableCsv = true;
        csvFile = argv[i + 1];
        i++;
      }
    }
    else if (strcmp(argv[i], "--compare") == 0)
    {
      if ((i + 1) < argc)
      {
        compareFileName = argv[i + 1];
        i++;
      }
    }
    else if (strcmp(argv[i], "--no-opencl") == 0 || strcmp(argv[i], "--no-vulkan") == 0)
    {
      // Backend-selection flags consumed in entry.cpp; ignore here.
    }
    else
    {
      // Check if it's a test selection flag
      bool matched = false;
      for (int t = 0; t < numTestFlags; t++)
      {
        if (strcmp(argv[i], testFlags[t].flag) == 0)
        {
          if (!forcedTests)
          {
            disableAll();
            forcedTests = true;
          }
          enableTest(testFlags[t].test);
          matched = true;
          break;
        }
      }
      if (!matched)
      {
        printParseMessage(helpStr);
        printParseMessage(NEWLINE);
        exit(-1);
      }
    }
  }

  jsonFileName = jsonFile;
  csvFileName  = csvFile;

  // Allocate logger after parsing (not needed for --list-devices but harmless)
  log.reset(new logger(enableXml, xmlFileName,
                       enableJson, jsonFileName,
                       enableCsv,  csvFileName,
                       compareFileName));
  return 0;
}
