#include <clpeak.h>
#include <cstdlib>
#if defined(_WIN32) || defined(__ANDROID__)
#include <malloc.h>
#endif

int clPeak::runTransferBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo)
{
  if (!isTransferBW)
    return 0;

  float timed, gbps;
  cl::NDRange globalSize, localSize;
  cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
  uint iters = devInfo.transferBWIters;
  float *arr = NULL;

  uint64_t maxItems = devInfo.maxAllocSize / sizeof(float) / 2;
  uint64_t numItems = roundToMultipleOf(maxItems, devInfo.maxWGSize, devInfo.transferBWMaxSize);

  try
  {
#if defined(_WIN32)
    arr = static_cast<float *>(_aligned_malloc(numItems * sizeof(float), 64));
#elif defined(__ANDROID__)
    arr = static_cast<float *>(memalign(64, numItems * sizeof(float)));
#else
    arr = static_cast<float *>(aligned_alloc(64, numItems * sizeof(float)));
#endif

    memset(arr, 0, numItems * sizeof(float));
    cl::Buffer clBuffer = cl::Buffer(ctx, (CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR), (numItems * sizeof(float)));

    log->print(NEWLINE TAB TAB "Transfer bandwidth (GBPS)" NEWLINE);
    log->xmlOpenTag("transfer_bandwidth");
    log->xmlAppendAttribs("unit", "gbps");

    ///////////////////////////////////////////////////////////////////////////
    // enqueueWriteBuffer
    if (!forceTest || strcmp(specifiedTestName, "enqueuewritebuffer") == 0)
    {
      log->print(TAB TAB TAB "enqueueWriteBuffer              : ");

      for (uint w = 0; w < warmupCount; w++)
      {
        queue.enqueueWriteBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
        queue.finish();
      }

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
    }
    ///////////////////////////////////////////////////////////////////////////
    // enqueueReadBuffer
    if (!forceTest || strcmp(specifiedTestName, "enqueuereadbuffer") == 0)
    {
      log->print(TAB TAB TAB "enqueueReadBuffer               : ");

      for (uint w = 0; w < warmupCount; w++)
      {
        queue.enqueueReadBuffer(clBuffer, CL_TRUE, 0, (numItems * sizeof(float)), arr);
        queue.finish();
      }

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
    }
    ///////////////////////////////////////////////////////////////////////////
    // enqueueWriteBuffer non-blocking
    if (!forceTest || strcmp(specifiedTestName, "enqueuewritebuffer_nonblocking") == 0)
    {
      log->print(TAB TAB TAB "enqueueWriteBuffer non-blocking : ");

      for (uint w = 0; w < warmupCount; w++)
      {
        queue.enqueueWriteBuffer(clBuffer, CL_FALSE, 0, (numItems * sizeof(float)), arr);
        queue.finish();
      }

      timed = 0;

      if (useEventTimer)
      {
        for (uint i = 0; i < iters; i++)
        {
          cl::Event timeEvent;
          queue.enqueueWriteBuffer(clBuffer, CL_FALSE, 0, (numItems * sizeof(float)), arr, NULL, &timeEvent);
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
          queue.enqueueWriteBuffer(clBuffer, CL_FALSE, 0, (numItems * sizeof(float)), arr);
        }
        queue.finish();
        timed = timer.stopAndTime();
      }
      timed /= static_cast<float>(iters);

      gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord("enqueuewritebuffer_nonblocking", gbps);
    }
    ///////////////////////////////////////////////////////////////////////////
    // enqueueReadBuffer non-blocking
    if (!forceTest || strcmp(specifiedTestName, "enqueuereadbuffer_nonblocking") == 0)
    {
      log->print(TAB TAB TAB "enqueueReadBuffer non-blocking  : ");

      for (uint w = 0; w < warmupCount; w++)
      {
        queue.enqueueReadBuffer(clBuffer, CL_FALSE, 0, (numItems * sizeof(float)), arr);
        queue.finish();
      }

      timed = 0;
      if (useEventTimer)
      {
        for (uint i = 0; i < iters; i++)
        {
          cl::Event timeEvent;
          queue.enqueueReadBuffer(clBuffer, CL_FALSE, 0, (numItems * sizeof(float)), arr, NULL, &timeEvent);
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
          queue.enqueueReadBuffer(clBuffer, CL_FALSE, 0, (numItems * sizeof(float)), arr);
        }
        queue.finish();
        timed = timer.stopAndTime();
      }
      timed /= static_cast<float>(iters);

      gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord("enqueuereadbuffer_nonblocking", gbps);
    }
    ///////////////////////////////////////////////////////////////////////////
    // enqueueMapBuffer
    if (!forceTest || strcmp(specifiedTestName, "enqueuemapbuffer") == 0)
    {
      log->print(TAB TAB TAB "enqueueMapBuffer(for read)      : ");

      queue.finish();

      // Always use wall-clock for map/unmap: CL events measure only GPU command
      // processing time, which is zero on unified-memory platforms (Apple Silicon),
      // causing division-by-zero / inf with --use-event-timer.
      timed = 0;
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
      timed /= static_cast<float>(iters);

      gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;
      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord("enqueuemapbuffer", gbps);
    }
    ///////////////////////////////////////////////////////////////////////////

    // memcpy from mapped ptr
    if (!forceTest || strcmp(specifiedTestName, "memcpy_from_mapped_ptr") == 0)
    {
      log->print(TAB TAB TAB TAB "memcpy from mapped ptr        : ");
      queue.finish();

      timed = 0;
      for (uint i = 0; i < iters; i++)
      {
        void *mapPtr;

        mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_READ, 0, (numItems * sizeof(float)));
        queue.finish();

        Timer timer;
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
    }

    ///////////////////////////////////////////////////////////////////////////

    // enqueueUnmap
    if (!forceTest || strcmp(specifiedTestName, "enqueueunmap") == 0)
    {
      log->print(TAB TAB TAB "enqueueUnmap(after write)       : ");

      queue.finish();

      // Always use wall-clock for map/unmap (see enqueueMapBuffer comment above).
      timed = 0;
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
      timed /= static_cast<float>(iters);
      gbps = ((float)numItems * sizeof(float)) / timed / 1e3f;

      log->print(gbps);
      log->print(NEWLINE);
      log->xmlRecord("enqueueunmap", gbps);
    }
    ///////////////////////////////////////////////////////////////////////////

    // memcpy to mapped ptr
    if (!forceTest || strcmp(specifiedTestName, "memcpy_to_mapped_ptr") == 0)
    {
      log->print(TAB TAB TAB TAB "memcpy to mapped ptr          : ");
      queue.finish();

      timed = 0;
      for (uint i = 0; i < iters; i++)
      {
        void *mapPtr;

        mapPtr = queue.enqueueMapBuffer(clBuffer, CL_TRUE, CL_MAP_WRITE, 0, (numItems * sizeof(float)));
        queue.finish();

        Timer timer;
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
    }

    ///////////////////////////////////////////////////////////////////////////
    log->xmlCloseTag(); // transfer_bandwidth

    if (arr)
#if defined(_WIN32)
      _aligned_free(arr);
#else
      std::free(arr);
#endif
  }
  catch (cl::Error &error)
  {
    stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
       << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());

    if (arr)
    {
#if defined(_WIN32)
      _aligned_free(arr);
#else
      std::free(arr);
#endif
    }
    return -1;
  }

  return 0;
}
