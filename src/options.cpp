#include <clpeak.h>

#define DEFAULT_XML_FILE_NAME "clpeak_results.xml"

static const char *helpStr =
    "\n clpeak [OPTIONS]"
    "\n"
    "\n OPTIONS:"
    "\n  -p, --platform num          choose platform (num starts with 0)"
    "\n  -d, --device num            choose device   (num starts with 0)"
    "\n  --use-event-timer           time using cl events instead of std chrono timer"
    "\n                              hide driver latencies [default: No]"
    "\n  --global-bandwidth          selectively run global bandwidth test"
    "\n  --compute-sp                selectively run single precision compute test"
    "\n  --compute-dp                selectively run double precision compute test"
    "\n  --compute-integer           selectively run integer compute test"
    "\n  --transfer-bandwidth        selectively run transfer bandwidth test"
    "\n  --kernel-latency            selectively run kernel latency test"
    "\n  --all-tests                 run all above tests [default]"
    "\n  --enable-xml-dump           Dump results to xml file"
    "\n  -f, --xml-file file_name    specify file name for xml dump"
    "\n  -v, --version               display version"
    "\n  -h, --help                  display help message"
    "\n";

int clPeak::parseArgs(int argc, char **argv)
{
  bool forcedTests = false;
  bool enableXml = false;
  string xmlFileName;

  for (int i = 1; i < argc; i++)
  {
    if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0))
    {
      log->print(helpStr);
      log->print(NEWLINE);
      exit(0);
    }
    else if ((strcmp(argv[i], "-v") == 0) || (strcmp(argv[i], "--version") == 0))
    {
      stringstream versionStr;
      versionStr << "clpeak version: " << VERSION_STR;

      log->print(versionStr.str().c_str());
      log->print(NEWLINE);
      exit(0);
    }
    else if ((strcmp(argv[i], "-p") == 0) || (strcmp(argv[i], "--platform") == 0))
    {
      if ((i + 1) < argc)
      {
        forcePlatform = true;
        specifiedPlatform = strtoul(argv[i + 1], NULL, 0);
        i++;
      }
    }
    else if ((strcmp(argv[i], "-d") == 0) || (strcmp(argv[i], "--device") == 0))
    {
      if ((i + 1) < argc)
      {
        forceDevice = true;
        specifiedDevice = strtoul(argv[i + 1], NULL, 0);
        i++;
      }
    }
    else if (strcmp(argv[i], "--use-event-timer") == 0)
    {
      useEventTimer = true;
    }
    else if ((strcmp(argv[i], "--global-bandwidth") == 0) || (strcmp(argv[i], "--compute-sp") == 0) || (strcmp(argv[i], "--compute-dp") == 0) || (strcmp(argv[i], "--compute-integer") == 0) || (strcmp(argv[i], "--transfer-bandwidth") == 0) || (strcmp(argv[i], "--kernel-latency") == 0))
    {
      // Disable all and enable only selected ones
      if (!forcedTests)
      {
        isGlobalBW = isComputeSP = isComputeDP = isComputeInt = isTransferBW = isKernelLatency = false;
        forcedTests = true;
      }

      if (strcmp(argv[i], "--global-bandwidth") == 0)
      {
        isGlobalBW = true;
      }
      else if (strcmp(argv[i], "--compute-sp") == 0)
      {
        isComputeSP = true;
      }
      else if (strcmp(argv[i], "--compute-dp") == 0)
      {
        isComputeDP = true;
      }
      else if (strcmp(argv[i], "--compute-integer") == 0)
      {
        isComputeInt = true;
      }
      else if (strcmp(argv[i], "--transfer-bandwidth") == 0)
      {
        isTransferBW = true;
      }
      else if (strcmp(argv[i], "--kernel-latency") == 0)
      {
        isKernelLatency = true;
      }
    }
    else if (strcmp(argv[i], "--all-tests") == 0)
    {
      isGlobalBW = isComputeSP = isComputeDP = isComputeInt = isTransferBW = isKernelLatency = true;
    }
    else if (strcmp(argv[i], "--enable-xml-dump") == 0)
    {
      enableXml = true;
      if (xmlFileName.length() < 1)
      {
        xmlFileName = DEFAULT_XML_FILE_NAME;
      }
    }
    else if ((strcmp(argv[i], "-f") == 0) || (strcmp(argv[i], "--xml-file") == 0))
    {
      if ((i + 1) < argc)
      {
        enableXml = true;
        xmlFileName = argv[i + 1];
        i++;
      }
    }
    else
    {
      log->print(helpStr);
      log->print(NEWLINE);
      exit(-1);
    }
  }

  // Allocate logger after parsing
  log = new logger(enableXml, xmlFileName);
  return 0;
}
