#include <runtime_overhead.h>
#include <runtime_overhead_logger.h>

void clPeak::runRuntimeOverheadTests(const cl::Context &ctx, const cl::Device &device, cl::Program &prog)
{
  cl::NDRange globalSize, localSize;
  const uint32_t globalWorkSize = 2048;

  try
  {
    log->print(NEWLINE TAB TAB "Host calls performance" NEWLINE);
    log->print(TAB TAB TAB "Single-precision compute (us)" NEWLINE);
    log->xmlOpenTag("host_calls_performance");
    log->xmlOpenTag("single_precision_compute");
    log->xmlAppendAttribs("unit", "us");

    {
      cl::CommandQueue queue = cl::CommandQueue(ctx, device);

      cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, (globalWorkSize * sizeof(cl_float)));

      globalSize = globalWorkSize;
      localSize = cl::NullRange;

      cl::Kernel kernel(prog, "runtime_overhead_sp");
      kernel.setArg(0, outputBuf);

      ///////////////////////////////////////////////////////////////////////////
      log->print(TAB TAB TAB TAB "1 queue, N/A threads, GlobalWorgroupSize set to 2048, LocalWorkgroupSize = NULL" NEWLINE);
      log->xmlOpenTag("Configuration");
      log->xmlAppendAttribs("queues", "1");
      log->xmlAppendAttribs("threads", "N/A");
      log->xmlAppendAttribs("gws", std::to_string(globalWorkSize));
      log->xmlAppendAttribs("lws", "NullRange");

      log->xmlOpenTag("Case");
      log->xmlAppendAttribs("enqueue_count", "1");
      log->xmlAppendAttribs("flush_count", "1");

      durationTimesVec firstEnqueue(1u, 1u);
      runKernel<1u, 1u, 1u>(queue, kernel, firstEnqueue, cl::NullRange, globalSize, localSize);

      log->print(TAB TAB TAB TAB TAB "First enqueue and flush duration" NEWLINE);
      log->print(TAB TAB TAB TAB TAB TAB "Enqueue duration   : ");
      log->print(firstEnqueue.enqueueTimesVector[0]);     log->print(NEWLINE);
      log->print(TAB TAB TAB TAB TAB TAB "Flush queue duration   : ");
      log->print(firstEnqueue.flushTimesVector[0]);     log->print(NEWLINE NEWLINE);
      log->xmlRecord("enqueue_duration", firstEnqueue.enqueueTimesVector[0]);
      log->xmlRecord("flush_queue_duration", firstEnqueue.flushTimesVector[0]);
      log->xmlCloseTag();     // case

      queue.finish();
      ///////////////////////////////////////////////////////////////////////////
      using namespace constants;

      {
        constexpr uint32_t batchSize = 1u;
        constexpr uint32_t enqueueIterations = thousandIterations;
        constexpr uint32_t flushIterations = thousandIterations;

        generateTestCase<batchSize, enqueueIterations, flushIterations>(queue, kernel, cl::NullRange, globalSize, localSize);
      }

      {
        constexpr uint32_t batchSize = 1u;
        constexpr uint32_t enqueueIterations = thousandIterations;
        constexpr uint32_t flushIterations = hundredIterations;

        generateTestCase<batchSize, enqueueIterations, flushIterations>(queue, kernel, cl::NullRange, globalSize, localSize);
      }

      {
        constexpr uint32_t batchSize = tenIterations;
        constexpr uint32_t enqueueIterations = hundredIterations;
        constexpr uint32_t flushIterations = hundredIterations;

        generateTestCase<batchSize, enqueueIterations, flushIterations>(queue, kernel, cl::NullRange, globalSize, localSize);
      }
    }

    ///////////////////////////////////////////////////////////////////////////

    log->xmlCloseTag();     // configuration
    log->xmlCloseTag();     // single_precision_compute
    log->xmlCloseTag();     // host_calls_performance
  }
  catch (cl::Error error)
  {
    stringstream ss;
    ss << error.what() << " (" << error.err() << ")" NEWLINE
      << TAB TAB TAB "Tests skipped" NEWLINE;
    log->print(ss.str());
  }

}

