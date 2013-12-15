#include <clpeak.h>


int clPeak::runComputeSP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
    float timed, gflops;
    cl_uint workPerWI;
    cl::NDRange globalSize, localSize;
    cl_float A = 1.3f;
    int iters = devInfo.computeIters;
    
    if(!isComputeSP)
        return 0;
    
    try
    {
        cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
        
        uint globalWIs = (devInfo.numCUs) * (devInfo.computeWgsPerCU) * (devInfo.maxWGSize);
        uint t = MIN((globalWIs * sizeof(cl_float)), devInfo.maxAllocSize);
        t = roundToPowOf2(t);
        globalWIs = t / sizeof(cl_float);
    
        cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * sizeof(cl_float)));
        
        globalSize = globalWIs;
        localSize = devInfo.maxWGSize;
        
        cl::Kernel kernel_v1(prog, "compute_sp_v1");
        kernel_v1.setArg(0, outputBuf), kernel_v1.setArg(1, A);

        cl::Kernel kernel_v2(prog, "compute_sp_v2");
        kernel_v2.setArg(0, outputBuf), kernel_v2.setArg(1, A);

        cl::Kernel kernel_v4(prog, "compute_sp_v4");
        kernel_v4.setArg(0, outputBuf), kernel_v4.setArg(1, A);

        cl::Kernel kernel_v8(prog, "compute_sp_v8");
        kernel_v8.setArg(0, outputBuf), kernel_v8.setArg(1, A);

        cl::Kernel kernel_v16(prog, "compute_sp_v16");
        kernel_v16.setArg(0, outputBuf), kernel_v16.setArg(1, A);
        
        cout << NEWLINE TAB TAB "Single-precision compute (GFLOPS)" << endl;
        cout << setprecision(2) << fixed;
        
        ///////////////////////////////////////////////////////////////////////////
        // Vector width 1
        cout << TAB TAB TAB "float   : ";   cout.flush();
        
        workPerWI = 4096;      // Indicates flops executed per work-item
            
        timed = run_kernel(queue, kernel_v1, globalSize, localSize, iters);

        gflops = ((float)globalWIs * workPerWI) / timed / 1e3;
        cout << gflops << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 2
        cout << TAB TAB TAB "float2  : ";   cout.flush();
        
        workPerWI = 4096;
            
        timed = run_kernel(queue, kernel_v2, globalSize, localSize, iters);

        gflops = ((float)globalWIs * workPerWI) / timed / 1e3;
        cout << gflops << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 4
        cout << TAB TAB TAB "float4  : ";   cout.flush();
        
        workPerWI = 4096;
            
        timed = run_kernel(queue, kernel_v4, globalSize, localSize, iters);

        gflops = ((float)globalWIs * workPerWI) / timed / 1e3;
        cout << gflops << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 8
        cout << TAB TAB TAB "float8  : ";   cout.flush();
        
        workPerWI = 4096;
            
        timed = run_kernel(queue, kernel_v8, globalSize, localSize, iters);

        gflops = ((float)globalWIs * workPerWI) / timed / 1e3;
        cout << gflops << endl;
        ///////////////////////////////////////////////////////////////////////////
        
        // Vector width 16
        cout << TAB TAB TAB "float16 : ";   cout.flush();
        
        workPerWI = 4096;
            
        timed = run_kernel(queue, kernel_v16, globalSize, localSize, iters);

        gflops = ((float)globalWIs * workPerWI) / timed / 1e3;
        cout << gflops << endl;
        ///////////////////////////////////////////////////////////////////////////
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

