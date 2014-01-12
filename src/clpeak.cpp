#include <clpeak.h>

#define MSTRINGIFY(...) #__VA_ARGS__

static const char *stringifiedKernels = 
#include "global_bandwidth_kernels.cl"
#include "compute_sp_kernels.cl"
#include "compute_dp_kernels.cl"
#include "compute_integer_kernels.cl"
;

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
"\n  -h, --help                  display help message"
"\n"
;


clPeak::clPeak():forcePlatform(false),forceDevice(false), specifiedPlatform(-1), specifiedDevice(-1), useEventTimer(false),
       isGlobalBW(true), isComputeSP(true), isComputeDP(true), isComputeInt(true), isTransferBW(true), isKernelLatency(true)
{
}

int clPeak::parseArgs(int argc, char **argv)
{
    bool forcedTests = false;

    for(int i=1; i<argc; i++)
    {
        if((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0))
        {
            cout << helpStr << endl;
            exit(0);
        } else
        if((strcmp(argv[i], "-p") == 0) || (strcmp(argv[i], "--platform") == 0))
        {
            if((i+1) < argc)
            {
                forcePlatform = true;
                specifiedPlatform = atoi(argv[i+1]);
                i++;
            }
        } else
        if((strcmp(argv[i], "-d") == 0) || (strcmp(argv[i], "--device") == 0))
        {
            if((i+1) < argc)
            {
                forceDevice = true;
                specifiedDevice = atoi(argv[i+1]);
                i++;
            }
        } else
        if(strcmp(argv[i], "--use-event-timer") == 0)
        {
            useEventTimer = true;
        } else
        if((strcmp(argv[i], "--global-bandwidth") == 0)   || (strcmp(argv[i], "--compute-sp") == 0)
                || (strcmp(argv[i], "--compute-dp") == 0) || (strcmp(argv[i], "--compute-integer") == 0)
                || (strcmp(argv[i], "--transfer-bandwidth") == 0) || (strcmp(argv[i], "--kernel-latency") == 0))
        {
            // Disable all and enable only selected ones
            if(!forcedTests) {
                isGlobalBW = isComputeSP = isComputeDP = isComputeInt = isTransferBW = isKernelLatency = false;
                forcedTests = true;
            }
            
            if(strcmp(argv[i], "--global-bandwidth") == 0) {
                isGlobalBW = true;
            } else
            if(strcmp(argv[i], "--compute-sp") == 0) {
                isComputeSP = true;
            } else
            if(strcmp(argv[i], "--compute-dp") == 0) {
                isComputeDP = true;
            } else
            if(strcmp(argv[i], "--compute-integer") == 0) {
                isComputeInt = true;
            } else
            if(strcmp(argv[i], "--transfer-bandwidth") == 0) {
                isTransferBW = true;
            } else
            if(strcmp(argv[i], "--kernel-latency") == 0) {
                isKernelLatency = true;
            }
            
        } else
        if(strcmp(argv[i], "-all-tests") == 0)
        {
            isGlobalBW = isComputeSP = isComputeDP = isComputeInt = isTransferBW = isKernelLatency = true;
        } else
        {
            cout << helpStr << endl;
            exit(-1);
        }
    }
    return 0;
}


int clPeak::runAll()
{
    try
    {
        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        for(int p=0; p < (int)platforms.size(); p++)
        {
            if(forcePlatform && (p != specifiedPlatform))
                continue;
            
            cout << NEWLINE "Platform: " << platforms[p].getInfo<CL_PLATFORM_NAME>() << endl;
            
            cl_context_properties cps[3] = { 
                    CL_CONTEXT_PLATFORM, 
                    (cl_context_properties)(platforms[p])(), 
                    0 
                };
            cl::Context ctx(CL_DEVICE_TYPE_ALL, cps);
            vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
            
            cl::Program::Sources source(1, make_pair(stringifiedKernels, (strlen(stringifiedKernels)+1)));
            cl::Program prog = cl::Program(ctx, source);
            
            try {
                prog.build(devices, BUILD_OPTIONS);
            }
            catch (cl::Error error) {
                cerr << TAB "Build Log: " << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << endl;
                throw error;
            }
            
            for(int d=0; d < (int)devices.size(); d++)
            {
                if(forceDevice && (d != specifiedDevice))
                    continue;
                
                device_info_t devInfo = getDeviceInfo(devices[d]);
                
                cout << TAB "Device: " << devInfo.deviceName << endl;
                cout << TAB TAB "Driver version : " << devInfo.driverVersion << " (" << OS_NAME << ")" << endl;
                cout << TAB TAB "Compute units  : " << devInfo.numCUs << endl;
                
                cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d], CL_QUEUE_PROFILING_ENABLE);
                
                runGlobalBandwidthTest(queue, prog, devInfo);
                runComputeSP(queue, prog, devInfo);
                runComputeDP(queue, prog, devInfo);
                runComputeInteger(queue, prog, devInfo);
                runTransferBandwidthTest(queue, prog, devInfo);
                runKernelLatency(queue, prog, devInfo);
                
                cout << NEWLINE;
            }
        }
    }
    catch(cl::Error error)
    {
        cerr << error.what() << "( " << error.err() << " )" << endl;
        return -1;
    }
    
    return 0;
}


float clPeak::run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel, cl::NDRange &globalSize, cl::NDRange &localSize, int iters)
{
    float timed = 0;
    
    // Dummy calls
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.finish();
    
    if(useEventTimer)
    {
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize, NULL, &timeEvent);
            queue.finish();
            timed += timeInUS(timeEvent);
        }
    } else      // std timer
    {
        Timer timer;
        
        timer.start();
        for(int i=0; i<iters; i++)
        {
            queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
        }
        queue.finish();
        timed = timer.stopAndTime();
    }
    
    return (timed / iters);
}    


