#include <clpeak.h>

#define FETCH_PER_WI        16


int clPeak::runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
    float timed, gbps;
    cl::NDRange globalSize, localSize;

    if(!isGlobalBW)
        return 0;

    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
    int iters = devInfo.gloalBWIters;

    cl_uint maxItems = devInfo.maxAllocSize / sizeof(float) / 2;
    cl_uint numItems;

    // Set an upper-limit for cpu devies
    if(devInfo.deviceType & CL_DEVICE_TYPE_CPU) {
        numItems = roundToPowOf2(maxItems, 25);
    } else {
        numItems = roundToPowOf2(maxItems);
    }

    float *arr = new float[numItems];
    populate(arr, numItems);

    try
    {
        cl::Buffer inputBuf = cl::Buffer(ctx, CL_MEM_READ_ONLY, (numItems * sizeof(float)));
        cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (numItems * sizeof(float)));
        queue.enqueueWriteBuffer(inputBuf, CL_TRUE, 0, (numItems * sizeof(float)), arr);

        cl::Kernel kernel_v1(prog, "global_bandwidth_v1");
        kernel_v1.setArg(0, inputBuf), kernel_v1.setArg(1, outputBuf);

        cl::Kernel kernel_v2(prog, "global_bandwidth_v2");
        kernel_v2.setArg(0, inputBuf), kernel_v2.setArg(1, outputBuf);

        cl::Kernel kernel_v4(prog, "global_bandwidth_v4");
        kernel_v4.setArg(0, inputBuf), kernel_v4.setArg(1, outputBuf);

        cl::Kernel kernel_v8(prog, "global_bandwidth_v8");
        kernel_v8.setArg(0, inputBuf), kernel_v8.setArg(1, outputBuf);

        cl::Kernel kernel_v16(prog, "global_bandwidth_v16");
        kernel_v16.setArg(0, inputBuf), kernel_v16.setArg(1, outputBuf);

        localSize = devInfo.maxWGSize;

        log->print(NEWLINE TAB TAB "Global memory bandwidth (GBPS)" NEWLINE);

        ///////////////////////////////////////////////////////////////////////////
        // Vector width 1
        log->print(TAB TAB TAB "float   : ");

        globalSize = numItems / FETCH_PER_WI;

        timed = run_kernel(queue, kernel_v1, globalSize, localSize, iters);

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

        log->print(gbps);   log->print(NEWLINE);
        log->xmlRecord("bandwidth_float", gbps);
        ///////////////////////////////////////////////////////////////////////////

        // Vector width 2
        log->print(TAB TAB TAB "float2  : ");

        globalSize = (numItems / 2 / FETCH_PER_WI);

        timed = run_kernel(queue, kernel_v2, globalSize, localSize, iters);

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

        log->print(gbps);   log->print(NEWLINE);
        log->xmlRecord("bandwidth_float2", gbps);
        ///////////////////////////////////////////////////////////////////////////

        // Vector width 4
        log->print(TAB TAB TAB "float4  : ");

        globalSize = (numItems / 4 / FETCH_PER_WI);

        timed = run_kernel(queue, kernel_v4, globalSize, localSize, iters);

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

        log->print(gbps);   log->print(NEWLINE);
        log->xmlRecord("bandwidth_float4", gbps);
        ///////////////////////////////////////////////////////////////////////////

        // Vector width 8
        log->print(TAB TAB TAB "float8  : ");

        globalSize = (numItems / 8 / FETCH_PER_WI);

        timed = run_kernel(queue, kernel_v8, globalSize, localSize, iters);

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

        log->print(gbps);   log->print(NEWLINE);
        log->xmlRecord("bandwidth_float8", gbps);
        ///////////////////////////////////////////////////////////////////////////

        // Vector width 16
        log->print(TAB TAB TAB "float16 : ");
        globalSize = (numItems / 16 / FETCH_PER_WI);

        timed = run_kernel(queue, kernel_v16, globalSize, localSize, iters);

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

        log->print(gbps);   log->print(NEWLINE);
        log->xmlRecord("bandwidth_float16", gbps);
        ///////////////////////////////////////////////////////////////////////////
    }
    catch(cl::Error error)
    {
        log->print(error.err() + NEWLINE);
        log->print(TAB TAB TAB "Tests skipped" NEWLINE);

        if(arr)     delete [] arr;
        return -1;
    }

    if(arr)     delete [] arr;
    return 0;
}

