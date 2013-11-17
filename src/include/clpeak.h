#ifndef CLPEAK_HPP
#define CLPEAK_HPP

#define __CL_ENABLE_EXCEPTIONS

#define WGS_PER_CU                  2048
#define PROFILE_ITERS_BANDWIDTH     20
#define PROFILE_ITERS_COMPUTE       50

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include <iostream>
#include <stdio.h>
#include <iomanip>
#include <string.h>
#include <common.h>

using namespace std;

class clPeak
{
public:

    bool forcePlatform, forceDevice;
    int specifiedPlatform, specifiedDevice;
    
    clPeak();

    int parseArgs(int argc, char **argv);
        
    int runBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);
    
    int runComputeSP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);
    
    int runComputeDP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo);

    int runAll();
};

#endif  // CLPEAK_HPP
