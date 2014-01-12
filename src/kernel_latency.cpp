#include <clpeak.h>

#define FETCH_PER_WI        16


int clPeak::runKernelLatency(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
    if(!isKernelLatency)
        return 0;
        
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl_uint numItems = (devInfo.maxWGSize) * (devInfo.numCUs) * FETCH_PER_WI;
    cl::NDRange globalSize = (numItems / FETCH_PER_WI);
    cl::NDRange localSize = devInfo.maxWGSize;
    int iters = devInfo.kernelLatencyIters;
    float latency;

    try
    {
        cout << NEWLINE TAB TAB "Kernel launch latency : "; cout.flush();
        
        cl::Buffer inputBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
        cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (numItems * sizeof(float)));
        
        cl::Kernel kernel_v1(prog, "global_bandwidth_v1");
        kernel_v1.setArg(0, inputBuf), kernel_v1.setArg(1, outputBuf);
        
        // Dummy calls
        queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
        queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
        queue.finish();
        
        latency = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize, NULL, &timeEvent);
            queue.finish();
            cl_ulong start = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() / 1000;
            cl_ulong end = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
            latency += (float)((int)end - (int)start);
        }
        latency /= iters;
        
        cout << setprecision(2) << fixed;
        cout << latency << " us" << endl;
    }
    catch(cl::Error error)
    {
        cerr << error.what() << "(" << error.err() << ")" << endl;
        cerr << TAB TAB "Tests skipped" << endl;
        return -1;
    }       
        
    return 0;
}

