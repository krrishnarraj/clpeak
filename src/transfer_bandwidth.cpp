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
    Timer timer;
    
    float *arr = new float[numItems];
    
    try
    {
        cl::Buffer clBuffer = cl::Buffer(ctx, (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR), (numItems * sizeof(float)));
        
        cout << NEWLINE TAB TAB "Transfer bandwidth (GBPS)" << endl;
        cout << setprecision(2) << fixed;
        
        ///////////////////////////////////////////////////////////////////////////
        // enqueueWriteBuffer
        cout << TAB TAB TAB "enqueueWriteBuffer         : ";    cout.flush();
        
        // Dummy warm-up
        queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
        queue.finish();
        
        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr, NULL, &timeEvent);
            queue.finish();
            timed += timeInUS(timeEvent);
        }
        timed /= iters;

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        // enqueueReadBuffer
        cout << TAB TAB TAB "enqueueReadBuffer          : ";    cout.flush();
        
        // Dummy warm-up
        queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
        queue.finish();
        
        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr, NULL, &timeEvent);
            queue.finish();
            timed += timeInUS(timeEvent);
        }
        timed /= iters;

        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        cout << gbps << endl;
        ///////////////////////////////////////////////////////////////////////////
        // enqueueMapBuffer
        cout << TAB TAB TAB "enqueueMapBuffer(for read) : ";    cout.flush();
        
        queue.finish();
        
        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            void *mapPtr;
            
            mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, (numItems * sizeof(float)), NULL, &timeEvent);
            queue.finish();
            queue.enqueueUnmapMemObject(clBuffer, mapPtr, NULL, NULL);
            queue.finish();
            timed += timeInUS(timeEvent);
        }
        timed /= iters;
        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        
        if(gbps > BANDWIDTH_UPPER_LIMIT)
        {
            cout << "Zero copy" << endl;
        }   else
        {
            cout << gbps << endl;
        }
        ///////////////////////////////////////////////////////////////////////////
        
        // memcpy from mapped ptr
        cout << TAB TAB TAB TAB "memcpy from mapped ptr   : ";  cout.flush();
        queue.finish();
        
        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            void *mapPtr;
            
            mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, (numItems * sizeof(float)));
            queue.finish();
            
            timer.start();
            memcpy(arr, mapPtr, (numItems * sizeof(float)));
            timed += timer.stopAndTime();
            
            queue.enqueueUnmapMemObject(clBuffer, mapPtr, NULL, NULL);
            queue.finish();
        }
        timed /= iters;
        
        // Count read & write bytes
        gbps = ((float)numItems * 2 * sizeof(float)) / timed / 1e3;
        cout << gbps << endl;

        ///////////////////////////////////////////////////////////////////////////
        
        // enqueueUnmap
        cout << TAB TAB TAB "enqueueUnmap(after write)  : ";    cout.flush();
        
        queue.finish();
        
        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            void *mapPtr;
            
            mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, (numItems * sizeof(float)));
            queue.finish();
            queue.enqueueUnmapMemObject(clBuffer, mapPtr, NULL, &timeEvent);
            queue.finish();
            timed += timeInUS(timeEvent);
        }
        timed /= iters;
        gbps = ((float)numItems * sizeof(float)) / timed / 1e3;
        
        if(gbps > BANDWIDTH_UPPER_LIMIT)
        {
            cout << "Zero copy" << endl;
        }   else
        {
            cout << gbps << endl;
        }
        ///////////////////////////////////////////////////////////////////////////
        
        // memcpy to mapped ptr
        cout << TAB TAB TAB TAB "memset on mapped ptr     : ";  cout.flush();
        queue.finish();
        
        timed = 0;
        for(int i=0; i<iters; i++)
        {
            cl::Event timeEvent;
            void *mapPtr;
            
            mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, (numItems * sizeof(float)));
            queue.finish();
            
            timer.start();
            memset(mapPtr, 0x1F, (numItems * sizeof(float)));
            timed += timer.stopAndTime();
            
            queue.enqueueUnmapMemObject(clBuffer, mapPtr, NULL, NULL);
            queue.finish();
        }
        timed /= iters;
        
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

