#include <clpeak.h>

int clPeak::runTransferBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
  if (!isTransferBW)
    return 0;

  float timed, gbps;
  cl::NDRange globalSize, localSize;
  cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
  uint iters = devInfo.transferBWIters;
  Timer timer;
  float *arr = NULL;

  uint64_t maxItems = devInfo.maxAllocSize / sizeof(float) / 2;
  uint64_t numItems = roundToMultipleOf(maxItems, devInfo.maxWGSize, devInfo.transferBWMaxSize);

  try
  {
    arr = new float[numItems];
    memset(arr, 0, numItems * sizeof(float));
    cl::Buffer clBuffer = cl::Buffer(ctx, (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR), (numItems * sizeof(float)));

    log->print(NEWLINE TAB TAB "Transfer bandwidth (GBPS)" NEWLINE);
    log->xmlOpenTag("transfer_bandwidth");
    log->xmlAppendAttribs("unit", "gbps");

    ///////////////////////////////////////////////////////////////////////////
    // enqueueWriteBuffer
    log->print(TAB TAB TAB "enqueueWriteBuffer         : ");

    // Dummy warm-up
    queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
    queue.finish();

    timed = 0;

    if (useEventTimer)
    {
      for (uint i = 0; i < iters; i++)
      {
        cl::Event timeEvent;
        queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr, NULL, &timeEvent);
        queue.finish();
        timed += timeInUS(timeEvent);
      }
    }
    else
    {
      Timer timer;

      timer.start();
      for (uint i = 0; i < iters; i++)
      {
        queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
      }
      queue.finish();
      timed = timer.stopAndTime();
    }
    timed /= static_cast<float>(iters);

    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("enqueuewritebuffer", gbps);
    ///////////////////////////////////////////////////////////////////////////
    // enqueueReadBuffer
    log->print(TAB TAB TAB "enqueueReadBuffer          : ");

    // Dummy warm-up
    queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
    queue.finish();

    timed = 0;
    if (useEventTimer)
    {
      for (uint i = 0; i < iters; i++)
      {
        cl::Event timeEvent;
        queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr, NULL, &timeEvent);
        queue.finish();
        timed += timeInUS(timeEvent);
      }
    }
    else
    {
      Timer timer;

      timer.start();
      for (uint i = 0; i < iters; i++)
      {
        queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
      }
      queue.finish();
      timed = timer.stopAndTime();
    }
    timed /= static_cast<float>(iters);

    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("enqueuereadbuffer", gbps);
    ///////////////////////////////////////////////////////////////////////////
    // enqueueMapBuffer
    log->print(TAB TAB TAB "enqueueMapBuffer(for read) : ");

    queue.finish();

    timed = 0;
    if (useEventTimer)
    {
      for (uint i = 0; i < iters; i++)
      {
        cl::Event timeEvent;
        void *mapPtr;

        mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, (numItems * sizeof(float)), NULL, &timeEvent);
        queue.finish();
        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
        timed += timeInUS(timeEvent);
      }
    }
    else
    {
      for (uint i = 0; i < iters; i++)
      {
        Timer timer;
        void *mapPtr;

        timer.start();
        mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, (numItems * sizeof(float)));
        queue.finish();
        timed += timer.stopAndTime();

        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
      }
    }
    timed /= static_cast<float>(iters);

    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("enqueuemapbuffer", gbps);
    ///////////////////////////////////////////////////////////////////////////

    // memcpy from mapped ptr
    log->print(TAB TAB TAB TAB "memcpy from mapped ptr   : ");
    queue.finish();

    timed = 0;
    for (uint i = 0; i < iters; i++)
    {
      cl::Event timeEvent;
      void *mapPtr;

      mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, (numItems * sizeof(float)));
      queue.finish();

      timer.start();
      memcpy(arr, mapPtr, (numItems * sizeof(float)));
      timed += timer.stopAndTime();

      queue.enqueueUnmapMemObject(clBuffer, mapPtr);
      queue.finish();
    }
    timed /= static_cast<float>(iters);

    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("memcpy_from_mapped_ptr", gbps);

    ///////////////////////////////////////////////////////////////////////////

    // enqueueUnmap
    log->print(TAB TAB TAB "enqueueUnmap(after write)  : ");

    queue.finish();

    timed = 0;
    if (useEventTimer)
    {
      for (uint i = 0; i < iters; i++)
      {
        cl::Event timeEvent;
        void *mapPtr;

        mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, (numItems * sizeof(float)));
        queue.finish();
        queue.enqueueUnmapMemObject(clBuffer, mapPtr, NULL, &timeEvent);
        queue.finish();
        timed += timeInUS(timeEvent);
      }
    }
    else
    {
      for (uint i = 0; i < iters; i++)
      {
        Timer timer;
        void *mapPtr;

        mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, (numItems * sizeof(float)));
        queue.finish();

        timer.start();
        queue.enqueueUnmapMemObject(clBuffer, mapPtr);
        queue.finish();
        timed += timer.stopAndTime();
      }
    }
    timed /= static_cast<float>(iters);
    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("enqueueunmap", gbps);
    ///////////////////////////////////////////////////////////////////////////

    // memcpy to mapped ptr
    log->print(TAB TAB TAB TAB "memcpy to mapped ptr     : ");
    queue.finish();

    timed = 0;
    for (uint i = 0; i < iters; i++)
    {
      cl::Event timeEvent;
      void *mapPtr;

      mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, (numItems * sizeof(float)));
      queue.finish();

      timer.start();
      memcpy(mapPtr, arr, (numItems * sizeof(float)));
      timed += timer.stopAndTime();

      queue.enqueueUnmapMemObject(clBuffer, mapPtr);
      queue.finish();
    }
    timed /= static_cast<float>(iters);

    gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
    log->print(gbps);
    log->print(NEWLINE);
    log->xmlRecord("memcpy_to_mapped_ptr", gbps);

    ///////////////////////////////////////////////////////////////////////////
    log->xmlCloseTag(); // transfer_bandwidth

    if (arr)
      delete[] arr;
  }
  catch (cl::Error &error)
  {
    stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());

    if (arr)
    {
      delete[] arr;
    }
    return -1;
  }

  return 0;
}
