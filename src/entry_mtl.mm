#ifdef ENABLE_METAL

#include <mtl_peak.h>
#include <cstring>
#include <cstdlib>

int MetalPeak::parseArgs(int argc, char **argv)
{
  bool enableXml = false;
  std::string xmlFileName, jsonFile, csvFile, compareFile;

  for (int i = 1; i < argc; i++)
  {
    if (strcmp(argv[i], "--list-devices") == 0)
      listDevices = true;
    else if ((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iters") == 0) && i + 1 < argc)
    {
      forceIters = true;
      specifiedIters = (unsigned int)atoi(argv[++i]);
    }
    else if ((strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--warmup") == 0) && i + 1 < argc)
    {
      warmupCount = (unsigned int)atoi(argv[++i]);
    }
    else if (strcmp(argv[i], "--json-file") == 0 && i + 1 < argc)
      jsonFile = argv[++i];
    else if (strcmp(argv[i], "--csv-file") == 0 && i + 1 < argc)
      csvFile = argv[++i];
    else if (strcmp(argv[i], "--compare") == 0 && i + 1 < argc)
      compareFile = argv[++i];
    // Other backend / OpenCL-specific flags are silently ignored.
  }

  log.reset(new logger(enableXml, xmlFileName,
                       !jsonFile.empty(), jsonFile,
                       !csvFile.empty(), csvFile,
                       compareFile));
  return 0;
}

#endif // ENABLE_METAL
