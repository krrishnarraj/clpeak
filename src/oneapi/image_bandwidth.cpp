#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>

class image_bw_kernel;

int OneapiPeak::runImageBandwidth(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"image_memory_bandwidth", "Image memory bandwidth", "gbps"});

  if (!dev.dev.has(sycl::aspect::image))
  {
    test.skip("float4", ResultStatus::Unsupported, "device does not advertise sycl::aspect::image");
    return 0;
  }

  const int imgW = 4096, imgH = 4096;
  const uint32_t blockSize = 256;
  uint64_t groups = ((uint64_t)imgW * (uint64_t)imgH) / IMAGE_FETCH_PER_WI / blockSize;
  if (groups == 0) groups = 1;
  uint64_t globalThreads = groups * blockSize;
  uint32_t numBlocks = (uint32_t)groups;

  // Staging buffer populated with xorshift bytes; uploaded via sycl::image
  // host_ptr on creation.  We use sycl::buffer image semantics (SYCL 2020
  // sampled_image) via the legacy unsampled_image type for portability.
  const size_t numFloats = (size_t)imgW * (size_t)imgH * 4;
  float *staging = new float[numFloats];
  populate(staging, numFloats);

  float *outBuf = sycl::malloc_device<float>(globalThreads, dev.stream);
  if (!outBuf)
  {
    delete[] staging;
    test.skip("float4", ResultStatus::Error, "Output buffer alloc failed");
    return -1;
  }

  try
  {
    sycl::image<2> img(staging,
                       sycl::image_channel_order::rgba,
                       sycl::image_channel_type::fp32,
                       sycl::range<2>(imgW, imgH));

    auto submit = [&](sycl::queue &q) -> sycl::event {
      return q.submit([&](sycl::handler &h) {
        // Unsampled image accessor: read coordinates as int2, get a float4 back.
        sycl::accessor<sycl::float4, 2, sycl::access::mode::read,
                       sycl::access::target::image>
            acc(img, h);

        h.parallel_for<image_bw_kernel>(
          sycl::nd_range<1>(globalThreads, blockSize),
          [=](sycl::nd_item<1> it) {
            uint32_t gid = (uint32_t)it.get_global_id(0);
            // Walk the image in scanline order, IMAGE_FETCH_PER_WI samples
            // per WI.  Coordinates wrap to keep all lanes in-bounds.
            uint32_t base = gid * IMAGE_FETCH_PER_WI;
            sycl::float4 sum{0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int i = 0; i < (int)IMAGE_FETCH_PER_WI; i++)
            {
              uint32_t idx = base + i;
              int x = (int)(idx % (uint32_t)imgW);
              int y = (int)((idx / (uint32_t)imgW) % (uint32_t)imgH);
              sum += acc.read(sycl::int2{x, y});
            }
            outBuf[gid] = sum.x() + sum.y() + sum.z() + sum.w();
          });
      });
    };

    float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (us <= 0.0f)
      test.skip("float4", ResultStatus::Error, "kernel launch failed");
    else
    {
      uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads;
      test.emit("float4", (float)bytes / us / 1e3f);
    }
  }
  catch (const sycl::exception &e)
  {
    test.skip("float4", ResultStatus::Error,
              std::string("image creation/dispatch failed: ") + e.what());
  }

  delete[] staging;
  sycl::free(outBuf, dev.stream);
  return 0;
}

#endif // ENABLE_ONEAPI
