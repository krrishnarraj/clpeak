#ifndef CLPEAK_HPP
#define CLPEAK_HPP

#define __CL_ENABLE_EXCEPTIONS

#define ITEMS_PER_CU    4096
#define PROFILE_ITERS   100

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <CL/cl.hpp>
#include <common.h>

using namespace std;
using namespace cl;

class clPeak
{
    public:

        bool verbose;

        int parseArgs(int argc, char **argv);
        
        int runBandwidthTest(CommandQueue &queue, Program &prog, device_info_t &devInfo); 

        int runAll();
};

#endif  // CLPEAK_HPP
