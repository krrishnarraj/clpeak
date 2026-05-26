#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>

// Read FETCH_PER_WI elements per work-item, strided by local-size so adjacent
// lanes coalesce.  Matches global_bandwidth.hip / .cl byte-for-byte.

template <int W> class global_bw_kernel;

template <int W>
static float runGlobalVariant(OneapiPeak &peak, OneapiDevice &dev,
                              const float *inBuf, float *outBuf,
                              uint32_t numBlocks, uint32_t blockSize,
                              unsigned int targetTimeUs, unsigned int forced)
{
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;
  using VecT = sycl::vec<float, W>;
  const VecT *inVec = reinterpret_cast<const VecT *>(inBuf);

  auto submit = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<global_bw_kernel<W>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          uint32_t gid = (uint32_t)it.get_global_id(0);
          uint32_t lid = (uint32_t)it.get_local_id(0);
          uint32_t wid = (uint32_t)it.get_group(0);
          uint32_t lsz = (uint32_t)it.get_local_range(0);
          uint32_t offset = wid * lsz * FETCH_PER_WI + lid;

          VecT sum{0.0f};
          #pragma unroll
          for (int i = 0; i < (int)FETCH_PER_WI; i++) {
            sum += inVec[offset];
            offset += lsz;
          }
          // Reduce vec lanes so the output is scalar (matches reference kernels).
          float acc = 0.0f;
          #pragma unroll
          for (int k = 0; k < W; k++) acc += sum[k];
          outBuf[gid] = acc;
        });
    });
  };
  return peak.runKernel(dev, submit, targetTimeUs, forced);
}

int OneapiPeak::runGlobalBandwidth(OneapiDevice &dev, benchmark_config_t &cfg)
{
  const uint32_t blockSize = 256;

  uint64_t maxItems = dev.info.totalGlobalMem / sizeof(float) / 4;
  uint64_t numItems = (maxItems / (blockSize * FETCH_PER_WI)) * (blockSize * FETCH_PER_WI);
  if (numItems > cfg.globalBWMaxSize / sizeof(float))
    numItems = (cfg.globalBWMaxSize / sizeof(float) / (blockSize * FETCH_PER_WI)) * (blockSize * FETCH_PER_WI);

  uint32_t numBlocks = (uint32_t)(numItems / FETCH_PER_WI / blockSize);
  if (numBlocks == 0)
    numBlocks = 1;

  auto test = currentDeviceScope->beginTest(
    {"global_memory_bandwidth", "Global memory bandwidth", "gbps"});

  float *inBuf  = sycl::malloc_device<float>(numItems, dev.stream);
  float *outBuf = sycl::malloc_device<float>(numItems, dev.stream);
  if (!inBuf || !outBuf)
  {
    const char *labels[] = {"float", "float2", "float4"};
    for (int i = 0; i < 3; i++)
      test.skip(labels[i], ResultStatus::Error, "Failed to allocate buffers");
    if (inBuf)  sycl::free(inBuf,  dev.stream);
    if (outBuf) sycl::free(outBuf, dev.stream);
    return -1;
  }

  // Populate with xorshift32 random bytes via host staging (defeats memory
  // compression — same trick the other backends use).
  {
    float *hInput = new float[numItems];
    populate(hInput, numItems);
    try { dev.stream.memcpy(inBuf, hInput, numItems * sizeof(float)).wait(); }
    catch (const sycl::exception &e)
    {
      const char *labels[] = {"float", "float2", "float4"};
      for (int i = 0; i < 3; i++)
        test.skip(labels[i], ResultStatus::Error,
                  std::string("Failed to upload input buffer: ") + e.what());
      delete[] hInput;
      sycl::free(inBuf,  dev.stream);
      sycl::free(outBuf, dev.stream);
      return -1;
    }
    delete[] hInput;
  }

  struct V { const char *key; int W; };
  const V vs[] = { {"float", 1}, {"float2", 2}, {"float4", 4} };

  for (const auto &v : vs)
  {
    uint64_t blocks = numItems / FETCH_PER_WI / v.W / blockSize;
    if (blocks == 0) blocks = 1;
    uint32_t blocksU = (uint32_t)blocks;

    float us = (v.W == 1)
      ? runGlobalVariant<1>(*this, dev, inBuf, outBuf, blocksU, blockSize, cfg.targetTimeUs, forceIters ? specifiedIters : 0)
      : (v.W == 2)
        ? runGlobalVariant<2>(*this, dev, inBuf, outBuf, blocksU, blockSize, cfg.targetTimeUs, forceIters ? specifiedIters : 0)
        : runGlobalVariant<4>(*this, dev, inBuf, outBuf, blocksU, blockSize, cfg.targetTimeUs, forceIters ? specifiedIters : 0);

    if (us <= 0.0f)
    {
      test.skip(v.key, ResultStatus::Error, "kernel launch failed");
      continue;
    }
    double bytes = (double)blocksU * blockSize * FETCH_PER_WI * v.W * sizeof(float);
    test.emit(v.key, (float)(bytes / us / 1e3));
  }

  sycl::free(inBuf,  dev.stream);
  sycl::free(outBuf, dev.stream);
  return 0;
}

#endif // ENABLE_ONEAPI
