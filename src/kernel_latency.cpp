#include <clpeak.h>


int clPeak::runKernelLatency(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
    if(!isKernelLatency)
        return 0;
        
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl_uint numItems = (devInfo.maxWGSize) * (devInfo.numCUs);
    cl::NDRange globalSize = numItems, localSize = devInfo.maxWGSize;
    int iters = devInfo.kernelLatencyIters;
    float latency;

    try
    {
        cout << NEWLINE TAB TAB "Kernel launch latency : "; cout.flush();
        
        cl::Buffer inputBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
        cl::Buffer inputBuf2 = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
        
        cl::Kernel kernel_v1(prog, "bandwidth_v1");
        kernel_v1.setArg(0, inputBuf), kernel_v1.setArg(1, inputBuf2);
        
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
            latency += (end - start);
        }
        latency /= iters;
        
        cout << setprecision(2) << fixed;
        cout << latency << " us" << endl;
    }
    catch(cl::Error error)
    {
        if(error.err() == CL_OUT_OF_RESOURCES)
        {
            cout << "Out of resources! Skipped" << endl;
        } else {
            throw error;
        }
    }       
        
    return 0;
}

