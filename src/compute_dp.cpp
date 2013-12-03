#include <clpeak.h>


int clPeak::runComputeDP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
    float timed, gflops;
    cl_uint workPerWI;
    cl::NDRange globalSize, localSize;
    cl_double A = 1.3f;
    int iters = devInfo.computeIters;
    
    if(!isComputeDP)
        return 0;
        
    if(!devInfo.doubleSupported)
    {
        cout << TAB TAB "Double precision is not supported! Skipped" << endl;
        return 0;
    }
        
    try
    {
        cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
        
        uint globalWIs = (devInfo.numCUs) * (devInfo.computeWgsPerCU) * (devInfo.maxWGSize);
        uint t = MIN((globalWIs * sizeof(cl_double)), devInfo.maxAllocSize);
        t = roundToPowOf2(t);
        globalWIs = t / sizeof(cl_double);
        cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_double)));
        
        globalSize = globalWIs;
        localSize = devInfo.maxWGSize;
        
        cl::Kernel kernel_v1(prog, "compute_dp_v1");
        kernel_v1.setArg(0, outputBuf), kernel_v1.setArg(1, A);

        cl::Kernel kernel_v2(prog, "compute_dp_v2");
        kernel_v2.setArg(0, outputBuf), kernel_v2.setArg(1, A);

        cl::Kernel kernel_v4(prog, "compute_dp_v4");
        kernel_v4.setArg(0, outputBuf), kernel_v4.setArg(1, A);

        cl::Kernel kernel_v8(prog, "compute_dp_v8");
        kernel_v8.setArg(0, outputBuf), kernel_v8.setArg(1, A);

        cl::Kernel kernel_v16(prog, "compute_dp_v16");
        kernel_v16.setArg(0, outputBuf), kernel_v16.setArg(1, A);
        
        cout << TAB TAB "Double-precision compute (GFLOPS)" << endl;
        cout << setprecision(2) << fixed;
        
        ///////////////////////////////////////////////////////////////////////////
        // Vector width 1
        
        workPerWI = 4096;      // Indicates flops executed per work-item
            
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

        gflops = ((float)globalWIs * workPerWI) / timed / 1e3;
        cout << TAB TAB TAB "double   : " << gflops << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 2
        workPerWI = 4096;
            
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

        gflops = ((float)globalWIs * workPerWI) / timed / 1e3;
        cout << TAB TAB TAB "double2  : " << gflops << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 4
        workPerWI = 4096;
            
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

        gflops = ((float)globalWIs * workPerWI) / timed / 1e3;
        cout << TAB TAB TAB "double4  : " << gflops << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 8
        workPerWI = 4096;
            
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

        gflops = ((float)globalWIs * workPerWI) / timed / 1e3;
        cout << TAB TAB TAB "double8  : " << gflops << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 16
        workPerWI = 4096;
            
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

        gflops = ((float)globalWIs * workPerWI) / timed / 1e3;
        cout << TAB TAB TAB "double16 : " << gflops << endl;
        ///////////////////////////////////////////////////////////////////////////
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

