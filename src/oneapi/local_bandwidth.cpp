#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>

// Shared-local-memory streaming benchmark.  Each work-group ping-pongs
// LMEM_REPS times through a per-group scratch array, alternating writes and
// reads with a barrier between phases (matches local_bandwidth_kernels.cl).
//
// SLM bytes accessed per WI per iter = 2 (read+write) * sizeof(vector)
// Total bytes = LMEM_REPS * 2 * width * sizeof(float) * globalThreads.

template <int W> class local_bw_kernel;

template <int W>
static float runLocalVariant(OneapiPeak &peak, OneapiDevice &dev,
                             float *outBuf,
                             uint32_t numBlocks, uint32_t blockSize,
                             unsigned int targetTimeUs, unsigned int forced)
{
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;
  using VecT = sycl::vec<float, W>;

  auto submit = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      sycl::local_accessor<VecT, 1> scratch{sycl::range<1>(blockSize), h};
      h.parallel_for<local_bw_kernel<W>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          uint32_t lid   = (uint32_t)it.get_local_id(0);
          uint32_t lsize = (uint32_t)it.get_local_range(0);
          uint32_t gid   = (uint32_t)it.get_global_id(0);
          uint32_t next  = (lid + 1u) % lsize;

          VecT sum{(float)gid};
          // Initialise additional lanes to distinct values so the compiler
          // can't fold the vector load to a broadcast.
          if (W >= 2) sum[(W >= 2 ? 1 : 0)] = (float)(gid + 1u);
          if (W >= 4) { sum[2] = (float)(gid + 2u); sum[3] = (float)(gid + 3u); }

          #pragma unroll 1
          for (int i = 0; i < (int)LMEM_REPS; i++)
          {
            scratch[lid] = sum;
            sycl::group_barrier(it.get_group());
            sum = scratch[next];
            sycl::group_barrier(it.get_group());
          }
          float acc = 0.0f;
          #pragma unroll
          for (int k = 0; k < W; k++) acc += sum[k];
          outBuf[gid] = acc;
        });
    });
  };
  return peak.runKernel(dev, submit, targetTimeUs, forced);
}

int OneapiPeak::runLocalBandwidth(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"local_memory_bandwidth", "Local memory bandwidth", "gbps"});

  const uint32_t blockSize = 256;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);
  uint32_t numBlocks = (uint32_t)(globalThreads / blockSize);

  float *outBuf = sycl::malloc_device<float>(globalThreads, dev.stream);
  if (!outBuf)
  {
    test.skip("float",  ResultStatus::Error, "Buffer alloc failed");
    test.skip("float2", ResultStatus::Error, "Buffer alloc failed");
    test.skip("float4", ResultStatus::Error, "Buffer alloc failed");
    return -1;
  }

  struct V { const char *key; int W; };
  const V vs[] = { {"float", 1}, {"float2", 2}, {"float4", 4} };

  for (const auto &v : vs)
  {
    float us = (v.W == 1)
      ? runLocalVariant<1>(*this, dev, outBuf, numBlocks, blockSize, cfg.targetTimeUs, forceIters ? specifiedIters : 0)
      : (v.W == 2)
        ? runLocalVariant<2>(*this, dev, outBuf, numBlocks, blockSize, cfg.targetTimeUs, forceIters ? specifiedIters : 0)
        : runLocalVariant<4>(*this, dev, outBuf, numBlocks, blockSize, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
    {
      test.skip(v.key, ResultStatus::Error, "kernel launch failed");
      continue;
    }
    uint64_t bytes = (uint64_t)LMEM_REPS * 2 * v.W * sizeof(float) * globalThreads;
    test.emit(v.key, (float)bytes / us / 1e3f);
  }

  sycl::free(outBuf, dev.stream);
  return 0;
}

#endif // ENABLE_ONEAPI
