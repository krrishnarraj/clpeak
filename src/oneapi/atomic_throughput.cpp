#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>

// Global + SLM atomic add throughput.  Each WI does ATOMIC_REPS adds on its
// own slot (no contention) so we measure raw atomic-issue rate, not
// contention-serialization.  Reported as Gop/s.

class atomic_global_kernel;
class atomic_local_kernel;

int OneapiPeak::runAtomicThroughput(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"atomic_throughput", "Atomic throughput", "gops"});

  const uint32_t blockSize = 256;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);
  uint32_t numBlocks = (uint32_t)(globalThreads / blockSize);

  // ---- Global atomics ----------------------------------------------------
  {
    int *buf = sycl::malloc_device<int>(globalThreads, dev.stream);
    if (!buf)
    {
      test.skip("int_global", ResultStatus::Error, "Buffer alloc failed");
    }
    else
    {
      try { dev.stream.memset(buf, 0, globalThreads * sizeof(int)).wait(); } catch (...) {}

      auto submit = [=](sycl::queue &q) -> sycl::event {
        return q.submit([&](sycl::handler &h) {
          h.parallel_for<atomic_global_kernel>(
            sycl::nd_range<1>(globalThreads, blockSize),
            [=](sycl::nd_item<1> it) {
              size_t gid = it.get_global_id(0);
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  a(buf[gid]);
              #pragma unroll 1
              for (int i = 0; i < (int)ATOMIC_REPS; i++)
                a.fetch_add(1);
            });
        });
      };
      float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
      if (us > 0.0f)
        test.emit("int_global", ((float)globalThreads * (float)ATOMIC_REPS) / us / 1e3f);
      else
        test.skip("int_global", ResultStatus::Error, "kernel launch failed");

      sycl::free(buf, dev.stream);
    }
  }

  // ---- Local (SLM) atomics ------------------------------------------------
  {
    int *outBuf = sycl::malloc_device<int>(numBlocks, dev.stream);
    if (!outBuf)
    {
      test.skip("int_local", ResultStatus::Error, "Buffer alloc failed");
    }
    else
    {
      auto submit = [=](sycl::queue &q) -> sycl::event {
        return q.submit([&](sycl::handler &h) {
          sycl::local_accessor<int, 1> scratch{sycl::range<1>(1), h};
          h.parallel_for<atomic_local_kernel>(
            sycl::nd_range<1>(globalThreads, blockSize),
            [=](sycl::nd_item<1> it) {
              if (it.get_local_id(0) == 0) scratch[0] = 0;
              sycl::group_barrier(it.get_group());

              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::work_group,
                               sycl::access::address_space::local_space>
                  a(scratch[0]);
              #pragma unroll 1
              for (int i = 0; i < (int)ATOMIC_REPS; i++)
                a.fetch_add(1);

              sycl::group_barrier(it.get_group());
              if (it.get_local_id(0) == 0)
                outBuf[it.get_group(0)] = scratch[0];
            });
        });
      };
      float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
      if (us > 0.0f)
        test.emit("int_local", ((float)globalThreads * (float)ATOMIC_REPS) / us / 1e3f);
      else
        test.skip("int_local", ResultStatus::Error, "kernel launch failed");

      sycl::free(outBuf, dev.stream);
    }
  }

  return 0;
}

#endif // ENABLE_ONEAPI
