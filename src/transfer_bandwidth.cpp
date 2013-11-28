#include <clpeak.h>


int clPeak::runTransferBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
    if(!isTransferBW)
        return 0;
        
    float timed, gbps;
    cl::NDRange globalSize, localSize;
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl_uint numItems = roundToPowOf2(devInfo.maxAllocSize / sizeof(float));
    int iters = devInfo.transferBWIters;
    
    float *arr = new float[numItems];
    
    try
    {
        cl::Buffer clBuffer = cl::Buffer(ctx, CL_MEM_READ_WRITE, (numItems * sizeof(float)));
        
        cout << TAB TAB "Transfer bandwidth (GBPS)" << endl;
        cout << setprecision(2) << fixed;
        
        ///////////////////////////////////////////////////////////////////////////
        // enqueueWriteBuffer
        
        // Dummy warm-up
        queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
        queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
        queue.finish();
        
        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr, NULL, &timeEvent);
            queue.finish();
            cl_ulong start = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
            cl_ulong end = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;
            timed += (end - start);
        }
        timed /= iters;

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << TAB TAB TAB "enqueueWriteBuffer         : " << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        // enqueueReadBuffer
        
        // Dummy warm-up
        queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
        queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
        queue.finish();
        
        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr, NULL, &timeEvent);
            queue.finish();
            cl_ulong start = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
            cl_ulong end = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;
            timed += (end - start);
        }
        timed /= iters;

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << TAB TAB TAB "enqueueReadBuffer          : " << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        // enqueueMapBuffer
        
        queue.finish();
        
        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent; cl_int err;
            void *mapPtr;
            
            mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, (numItems * sizeof(float)), NULL, &timeEvent);
            queue.finish();
            queue.enqueueUnmapMemObject(clBuffer, mapPtr, NULL, NULL);
            queue.finish();
            cl_ulong start = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
            cl_ulong end = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;
            timed += (end - start);
        }
        timed /= iters;

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << TAB TAB TAB "enqueueMapBuffer(for read) : " << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        // enqueueUnmap
        
        queue.finish();
        
        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent; cl_int err;
            void *mapPtr;
            
            mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, (numItems * sizeof(float)));
            queue.finish();
            queue.enqueueUnmapMemObject(clBuffer, mapPtr, NULL, &timeEvent);
            queue.finish();
            cl_ulong start = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() / 1000;
            cl_ulong end = timeEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() / 1000;
            timed += (end - start);
        }
        timed /= iters;

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << TAB TAB TAB "enqueueUnmap(for write)    : " << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////

            
    }
    catch(cl::Error error)
    {
        if(error.err() == CL_OUT_OF_RESOURCES)
        {
            cout << TAB TAB TAB "Out of resources! Skipped" << endl;
        } else {
            if(arr)     delete [] arr;
            throw error;
        }
    }

    if(arr)     delete [] arr;
    return 0;
}



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
        cout << TAB TAB "Kernel launch latency : " << latency << " us" << endl;
    }
    catch(cl::Error error)
    {
        if(error.err() == CL_OUT_OF_RESOURCES)
        {
            cout << TAB TAB TAB "Out of resources! Skipped" << endl;
        } else {
            throw error;
        }
    }       
        
    return 0;
}
