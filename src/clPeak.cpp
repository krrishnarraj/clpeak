#include <clPeak.h>

#define MSTRINGIFY(A) #A

static const char* stringifiedKernels = 
#include "bandwidth_kernels.cl"
#include "compute_kernels.cl"
;

int clPeak::parseArgs(int argc, char **argv)
{
}

int clPeak::runBandwidthTest(CommandQueue &queue, Program &prog, device_info_t &devInfo)
{
    Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl_uint numItems = devInfo.numCUs * ITEMS_PER_CU;
    
    float *arr = new float[numItems];
    populate(arr, numItems);

    Buffer inputBuf = Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
    queue.enqueueWriteBuffer(inputBuf, CL_TRUE, 0, (numItems * sizeof(float)), arr);
    
    Kernel kernel(prog, "bandwidth");
    kernel.setArg(0, inputBuf);
    kernel.setArg(1, numItems);
    
    NDRange global(numItems);
    NDRange local(devInfo.maxWGSize);
    
    // 2 dummy calls
    queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
    queue.enqueueNDRangeKernel(kernel, NullRange, global, local);

    INIT_TIMER(timeIt);
    for(int i=0; i<PROFILE_ITERS; i++)
    {
        queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
    }
    float timed = ELAPSED_TIME(timeIt);
    float gbps = numItems * sizeof(float) / timed / PROFILE_ITERS;

    std::cout << TAB TAB "Bandwidth: " << gbps << " GBPS" << endl;

    delete [] arr;
    return 0;
}

int clPeak::runAll()
{
    try
    {
        vector<Platform> platforms;
        Platform::get(&platforms);

        for(int p=0; p < platforms.size(); p++)
        {
            std::cout << NEWLINE "Platform: " << platforms[p].getInfo<CL_PLATFORM_NAME>() << std::endl;
            
            cl_context_properties cps[3] = { 
                    CL_CONTEXT_PLATFORM, 
                    (cl_context_properties)(platforms[p])(), 
                    0 
                };
            Context ctx(CL_DEVICE_TYPE_ALL, cps);
            vector<Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
            
            Program::Sources source(1, std::make_pair(stringifiedKernels, (strlen(stringifiedKernels)+1)));
            Program prog = Program(ctx, source);
            
            try {
                prog.build(devices);
            }
            catch (Error error){
                std::cerr << TAB "Build Log: " << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
                throw error;
            }
            
            for(int d=0; d < devices.size(); d++)
            {
                std::cout << TAB "Device: " << devices[d].getInfo<CL_DEVICE_NAME>() << std::endl;
                std::cout << TAB TAB "Driver version: " << devices[d].getInfo<CL_DRIVER_VERSION>() << std::endl;
                
                device_info_t devInfo = getDeviceInfo(devices[d]);
                CommandQueue queue = CommandQueue(ctx, devices[d]);
                
                runBandwidthTest(queue, prog, devInfo);
            }
            std::cout << NEWLINE;
        }
    }
    catch(Error error)
    {
        std::cerr << error.what() << "( " << error.err() << " )" << std::endl;
        return -1;
    }
    
    return 0;
}

