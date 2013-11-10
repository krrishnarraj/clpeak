#include <clpeak.h>


int clPeak::runComputeSP(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
    Timer timer;
    float timed, gflops;
    cl_uint workPerWI;
    cl::NDRange globalSize, localSize;
    cl_float A = 1.3f, B = 1.4f;
    
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    
    uint globalWIs = (devInfo.numCUs) * WGS_PER_CU * devInfo.maxWGSize;
    // Allocate enough buffer for float16
    uint t = MIN((globalWIs * 16 * sizeof(cl_float)), devInfo.maxAllocSize);
    t = roundToPowOf2(t);
    globalWIs = t / 16 / sizeof(cl_float);
    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWIs * 16 * sizeof(cl_float)));
    
    globalSize = globalWIs;
    localSize = devInfo.maxWGSize;
    
    cl::Kernel kernel_v1(prog, "compute_sp_v1");
    kernel_v1.setArg(0, outputBuf), kernel_v1.setArg(1, A), kernel_v1.setArg(2, B);

    cl::Kernel kernel_v2(prog, "compute_sp_v2");
    kernel_v2.setArg(0, outputBuf), kernel_v2.setArg(1, A), kernel_v2.setArg(2, B);

    cl::Kernel kernel_v4(prog, "compute_sp_v4");
    kernel_v4.setArg(0, outputBuf), kernel_v4.setArg(1, A), kernel_v4.setArg(2, B);

    cl::Kernel kernel_v8(prog, "compute_sp_v8");
    kernel_v8.setArg(0, outputBuf), kernel_v8.setArg(1, A), kernel_v8.setArg(2, B);

    cl::Kernel kernel_v16(prog, "compute_sp_v16");
    kernel_v16.setArg(0, outputBuf), kernel_v16.setArg(1, A), kernel_v16.setArg(2, B);
    
    cout << TAB TAB "Single-precision compute (GFLOPS)" << endl;
    cout << setprecision(2) << fixed;
    
    ///////////////////////////////////////////////////////////////////////////
    // Vector width 1
    
    workPerWI = 256;      // Indicates flops executed per work-item
        
    // 2 dummy calls
    queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
    queue.finish();

    timer.start();
    for(int i=0; i<PROFILE_ITERS_COMPUTE; i++)
    {
        queue.enqueueNDRangeKernel(kernel_v1, cl::NullRange, globalSize, localSize);
    }
    queue.finish();
    timed = timer.stopAndTime() / PROFILE_ITERS_COMPUTE;

    gflops = (globalWIs * workPerWI) / timed / 1e3;
    cout << TAB TAB TAB "float   : " << gflops << endl;
    ///////////////////////////////////////////////////////////////////////////
    
    // Vector width 2
    workPerWI = 256;
        
    queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, globalSize, localSize);
    queue.finish();

    timer.start();
    for(int i=0; i<PROFILE_ITERS_COMPUTE; i++)
    {
        queue.enqueueNDRangeKernel(kernel_v2, cl::NullRange, globalSize, localSize);
    }
    queue.finish();
    timed = timer.stopAndTime() / PROFILE_ITERS_COMPUTE;

    gflops = (globalWIs * workPerWI) / timed / 1e3;
    cout << TAB TAB TAB "float2  : " << gflops << endl;
    ///////////////////////////////////////////////////////////////////////////
    
    // Vector width 4
    workPerWI = 512;
        
    queue.enqueueNDRangeKernel(kernel_v4, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v4, cl::NullRange, globalSize, localSize);
    queue.finish();

    timer.start();
    for(int i=0; i<PROFILE_ITERS_COMPUTE; i++)
    {
        queue.enqueueNDRangeKernel(kernel_v4, cl::NullRange, globalSize, localSize);
    }
    queue.finish();
    timed = timer.stopAndTime() / PROFILE_ITERS_COMPUTE;

    gflops = (globalWIs * workPerWI) / timed / 1e3;
    cout << TAB TAB TAB "float4  : " << gflops << endl;
    ///////////////////////////////////////////////////////////////////////////
    
    // Vector width 8
    workPerWI = 512;
        
    queue.enqueueNDRangeKernel(kernel_v8, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v8, cl::NullRange, globalSize, localSize);
    queue.finish();

    timer.start();
    for(int i=0; i<PROFILE_ITERS_COMPUTE; i++)
    {
        queue.enqueueNDRangeKernel(kernel_v8, cl::NullRange, globalSize, localSize);
    }
    queue.finish();
    timed = timer.stopAndTime() / PROFILE_ITERS_COMPUTE;

    gflops = (globalWIs * workPerWI) / timed / 1e3;
    cout << TAB TAB TAB "float8  : " << gflops << endl;
    ///////////////////////////////////////////////////////////////////////////
    
    // Vector width 16
    workPerWI = 512;
        
    queue.enqueueNDRangeKernel(kernel_v16, cl::NullRange, globalSize, localSize);
    queue.enqueueNDRangeKernel(kernel_v16, cl::NullRange, globalSize, localSize);
    queue.finish();

    timer.start();
    for(int i=0; i<PROFILE_ITERS_COMPUTE; i++)
    {
        queue.enqueueNDRangeKernel(kernel_v16, cl::NullRange, globalSize, localSize);
    }
    queue.finish();
    timed = timer.stopAndTime() / PROFILE_ITERS_COMPUTE;

    gflops = (globalWIs * workPerWI) / timed / 1e3;
    cout << TAB TAB TAB "float16 : " << gflops << endl;
    ///////////////////////////////////////////////////////////////////////////

    return 0;
}

