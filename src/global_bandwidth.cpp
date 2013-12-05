#include <clpeak.h>


int clPeak::runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
    float timed, gbps;
    cl::NDRange globalSize, localSize;
    
    if(!isGlobalBW)
        return 0;
    
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    cl_uint numItems = roundToPowOf2(devInfo.maxAllocSize / sizeof(float));
    int iters = devInfo.computeIters;
    
    float *arr = new float[numItems];
    populate(arr, numItems);
    
    try
    {
        cl::Buffer inputBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
        cl::Buffer inputBuf2 = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
        queue.enqueueWriteBuffer(inputBuf, CL_TRUE, 0, (numItems * sizeof(float)), arr);
        
        cl::Kernel kernel_v1(prog, "bandwidth_v1");
        kernel_v1.setArg(0, inputBuf), kernel_v1.setArg(1, inputBuf2);
        
        cl::Kernel kernel_v2(prog, "bandwidth_v2");
        kernel_v2.setArg(0, inputBuf), kernel_v2.setArg(1, inputBuf2);
        
        cl::Kernel kernel_v4(prog, "bandwidth_v4");
        kernel_v4.setArg(0, inputBuf), kernel_v4.setArg(1, inputBuf2);
        
        cl::Kernel kernel_v8(prog, "bandwidth_v8");
        kernel_v8.setArg(0, inputBuf), kernel_v8.setArg(1, inputBuf2);
        
        cl::Kernel kernel_v16(prog, "bandwidth_v16");
        kernel_v16.setArg(0, inputBuf), kernel_v16.setArg(1, inputBuf2);
        
        localSize = devInfo.maxWGSize;
        
        cout << NEWLINE TAB TAB "Global memory bandwidth (GBPS)" << endl;
        cout << setprecision(2) << fixed;
        
        ///////////////////////////////////////////////////////////////////////////
        // Vector width 1
        globalSize = numItems;
        
        // Dummy calls
        queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
        queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
        queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
        queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
        queue.finish();

        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize, NULL, &timeEvent);
            queue.finish();
            timed += timeInUS(timeEvent);
        }
        timed /= iters;

        gbps = ((float)numItems * 2 * sizeof(float)) / timed / 1e3;
        cout << TAB TAB TAB "float   : " << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 2
        globalSize = (numItems / 2);
        
        queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, globalSize, localSize);
        queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, globalSize, localSize);
        queue.finish();

        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, globalSize, localSize, NULL, &timeEvent);
            queue.finish();
            timed += timeInUS(timeEvent);
        }
        timed /= iters;

        gbps = ((float)numItems * 2 * sizeof(float)) / timed / 1e3;
        cout << TAB TAB TAB "float2  : " << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 4
        globalSize = (numItems / 4);
        
        queue.enqueueNDRangeKernel(kernel_v4, cl::NullRange, globalSize, localSize);
        queue.enqueueNDRangeKernel(kernel_v4, cl::NullRange, globalSize, localSize);
        queue.finish();

        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            queue.enqueueNDRangeKernel(kernel_v4, cl::NullRange, globalSize, localSize, NULL, &timeEvent);
            queue.finish();
            timed += timeInUS(timeEvent);
        }
        timed /= iters;

        gbps = ((float)numItems * 2 * sizeof(float)) / timed / 1e3;
        cout << TAB TAB TAB "float4  : " << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 8
        globalSize = (numItems / 8);
        
        queue.enqueueNDRangeKernel(kernel_v8, cl::NullRange, globalSize, localSize);
        queue.enqueueNDRangeKernel(kernel_v8, cl::NullRange, globalSize, localSize);
        queue.finish();

        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            queue.enqueueNDRangeKernel(kernel_v8, cl::NullRange, globalSize, localSize, NULL, &timeEvent);
            queue.finish();
            timed += timeInUS(timeEvent);
        }
        timed /= iters;

        gbps = ((float)numItems * 2 * sizeof(float)) / timed / 1e3;
        cout << TAB TAB TAB "float8  : " << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 16
        globalSize = (numItems / 16);
        
        queue.enqueueNDRangeKernel(kernel_v16, cl::NullRange, globalSize, localSize);
        queue.enqueueNDRangeKernel(kernel_v16, cl::NullRange, globalSize, localSize);
        queue.finish();

        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            queue.enqueueNDRangeKernel(kernel_v16, cl::NullRange, globalSize, localSize, NULL, &timeEvent);
            queue.finish();
            timed += timeInUS(timeEvent);
        }
        timed /= iters;

        gbps = ((float)numItems * 2 * sizeof(float)) / timed / 1e3;
        cout << TAB TAB TAB "float16 : " << gbps << endl;
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

