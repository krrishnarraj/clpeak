#include <clPeak.h>

#define MSTRINGIFY(A) #A

static const char* stringifiedKernels = 
#include "bandwidth_kernels.cl"
#include "compute_kernels.cl"
;

int clPeak::parseArgs(int argc, char **argv)
{
    return 0;
}

int clPeak::runBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
    Timer timer;
    float timed, gbps;
    cl::NDRange globalSize, localSize;
    
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl_uint numItems = roundToPowOf2(devInfo.maxAllocSize / sizeof(float));
    
    float *arr = new float[numItems];
    populate(arr, numItems);

    cl::Buffer inputBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
    queue.enqueueWriteBuffer(inputBuf, CL_TRUE, 0, (numItems * sizeof(float)), arr);
    
    cl::Kernel kernel_v1(prog, "bandwidth_v1");
    kernel_v1.setArg(0, inputBuf);
    
    cl::Kernel kernel_v2(prog, "bandwidth_v2");
    kernel_v2.setArg(0, inputBuf);
    
    cl::Kernel kernel_v4(prog, "bandwidth_v4");
    kernel_v4.setArg(0, inputBuf);
    
    cl::Kernel kernel_v8(prog, "bandwidth_v8");
    kernel_v8.setArg(0, inputBuf);
    
    cl::Kernel kernel_v16(prog, "bandwidth_v16");
    kernel_v16.setArg(0, inputBuf);
    
    localSize = devInfo.maxWGSize;
    
    cout << TAB TAB "Memory bandwidth (GBPS)" << endl;
    cout << setprecision(2) << fixed;
    
    ///////////////////////////////////////////////////////////////////////////
    // Vector width 1
    globalSize = numItems;
    
    // 2 dummy calls
    queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
    queue.finish();

    timer.start();
    for(int i=0; i<PROFILE_ITERS; i++)
    {
        queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
    }
    queue.finish();
    timed = timer.stopAndTime() / PROFILE_ITERS;

    gbps = (numItems * sizeof(float)) / timed / 1e3;
    cout << TAB TAB TAB "float   : " << gbps << endl;
    ///////////////////////////////////////////////////////////////////////////
    
    // Vector width 2
    globalSize = (numItems / 2);
    
    queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, globalSize, localSize);
    queue.finish();

    timer.start();
    for(int i=0; i<PROFILE_ITERS; i++)
    {
        queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, globalSize, localSize);
    }
    queue.finish();
    timed = timer.stopAndTime() / PROFILE_ITERS;

    gbps = (numItems * sizeof(float)) / timed / 1e3;
    cout << TAB TAB TAB "float2  : " << gbps << endl;
    ///////////////////////////////////////////////////////////////////////////
    
    // Vector width 4
    globalSize = (numItems / 4);
    
    queue.enqueueNDRangeKernel(kernel_v4, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v4, cl::NullRange, globalSize, localSize);
    queue.finish();

    timer.start();
    for(int i=0; i<PROFILE_ITERS; i++)
    {
        queue.enqueueNDRangeKernel(kernel_v4, cl::NullRange, globalSize, localSize);
    }
    queue.finish();
    timed = timer.stopAndTime() / PROFILE_ITERS;

    gbps = (numItems * sizeof(float)) / timed / 1e3;
    cout << TAB TAB TAB "float4  : " << gbps << endl;
    ///////////////////////////////////////////////////////////////////////////
    
    // Vector width 8
    globalSize = (numItems / 8);
    
    queue.enqueueNDRangeKernel(kernel_v8, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v8, cl::NullRange, globalSize, localSize);
    queue.finish();

    timer.start();
    for(int i=0; i<PROFILE_ITERS; i++)
    {
        queue.enqueueNDRangeKernel(kernel_v8, cl::NullRange, globalSize, localSize);
    }
    queue.finish();
    timed = timer.stopAndTime() / PROFILE_ITERS;

    gbps = (numItems * sizeof(float)) / timed / 1e3;
    cout << TAB TAB TAB "float8  : " << gbps << endl;
    ///////////////////////////////////////////////////////////////////////////
    
    // Vector width 16
    globalSize = (numItems / 16);
    
    queue.enqueueNDRangeKernel(kernel_v16, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v16, cl::NullRange, globalSize, localSize);
    queue.finish();

    timer.start();
    for(int i=0; i<PROFILE_ITERS; i++)
    {
        queue.enqueueNDRangeKernel(kernel_v16, cl::NullRange, globalSize, localSize);
    }
    queue.finish();
    timed = timer.stopAndTime() / PROFILE_ITERS;

    gbps = (numItems * sizeof(float)) / timed / 1e3;
    cout << TAB TAB TAB "float16 : " << gbps << endl;
    ///////////////////////////////////////////////////////////////////////////

    delete [] arr;
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
                prog.build(devices);
            }
            catch (cl::Error error){
                cerr << TAB "Build Log: " << prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << endl;
                throw error;
            }
            
            for(int d=0; d < (int)devices.size(); d++)
            {
                cout << TAB "Device: " << devices[d].getInfo<CL_DEVICE_NAME>() << endl;
                cout << TAB TAB "Driver version: " << devices[d].getInfo<CL_DRIVER_VERSION>() << endl;
                
                device_info_t devInfo = getDeviceInfo(devices[d]);
                cl::CommandQueue queue = cl::CommandQueue(ctx, devices[d]);
                
                runBandwidthTest(queue, prog, devInfo);
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

