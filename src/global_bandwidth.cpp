#include <clpeak.h>

#define FETCH_PER_WI        16


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
        cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (numItems * sizeof(float)));
        queue.enqueueWriteBuffer(inputBuf, CL_TRUE, 0, (numItems * sizeof(float)), arr);
        
        cl::Kernel kernel_v1(prog, "bandwidth_v1");
        kernel_v1.setArg(0, inputBuf), kernel_v1.setArg(1, outputBuf);
        
        cl::Kernel kernel_v2(prog, "bandwidth_v2");
        kernel_v2.setArg(0, inputBuf), kernel_v2.setArg(1, outputBuf);
        
        cl::Kernel kernel_v4(prog, "bandwidth_v4");
        kernel_v4.setArg(0, inputBuf), kernel_v4.setArg(1, outputBuf);
        
        cl::Kernel kernel_v8(prog, "bandwidth_v8");
        kernel_v8.setArg(0, inputBuf), kernel_v8.setArg(1, outputBuf);
        
        cl::Kernel kernel_v16(prog, "bandwidth_v16");
        kernel_v16.setArg(0, inputBuf), kernel_v16.setArg(1, outputBuf);
        
        localSize = devInfo.maxWGSize;
        
        cout << NEWLINE TAB TAB "Global memory bandwidth (GBPS)" << endl;
        cout << setprecision(2) << fixed;
        
        ///////////////////////////////////////////////////////////////////////////
        // Vector width 1
        cout << TAB TAB TAB "float   : ";   cout.flush();
        
        globalSize = numItems / FETCH_PER_WI;
        
        timed = run_kernel(queue, kernel_v1, globalSize, localSize, iters);
        
        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 2
        cout << TAB TAB TAB "float2  : ";   cout.flush();
        
        globalSize = (numItems / 2 / FETCH_PER_WI);
        
        timed = run_kernel(queue, kernel_v2, globalSize, localSize, iters);

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 4
        cout << TAB TAB TAB "float4  : ";   cout.flush();
        
        globalSize = (numItems / 4 / FETCH_PER_WI);
        
        timed = run_kernel(queue, kernel_v4, globalSize, localSize, iters);

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 8
        cout << TAB TAB TAB "float8  : ";   cout.flush();
        
        globalSize = (numItems / 8 / FETCH_PER_WI);
        
        timed = run_kernel(queue, kernel_v8, globalSize, localSize, iters);

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 16
        cout << TAB TAB TAB "float16 : ";   cout.flush();
        
        globalSize = (numItems / 16 / FETCH_PER_WI);
        
        timed = run_kernel(queue, kernel_v16, globalSize, localSize, iters);

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
    }
    catch(cl::Error error)
    {
        if(error.err() == CL_OUT_OF_RESOURCES)
        {
            cout << "Out of resources! Skipped" << endl;
        } else {
            if(arr)     delete [] arr;
            throw error;
        }
    }

    if(arr)     delete [] arr;
    return 0;
}

