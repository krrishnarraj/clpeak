#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <algorithm>
#include <type_traits>
#include <sycl/sycl.hpp>

template <int WALK> class image_bw_kernel;

int OneapiPeak::runImageBandwidth(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"image_memory_bandwidth", "Image memory bandwidth", "gbps",
     Category::Unknown,
     "How many bytes per second the device reads through its texture units, "
     "which take a different path to memory than plain buffer reads.  Each "
     "pixel of the image is read exactly once, so caching cannot flatter the "
     "number."});

  // RGBA float image, so one fetch returns a whole pixel: four 32-bit values,
  // hence the metric name.
  const char *fetchNote = "Each fetch returns one whole pixel -- four 32-bit "
                          "colour values, 16 bytes.";

  if (!dev.dev.has(sycl::aspect::ext_intel_legacy_image))
  {
    test.skip("float4", ResultStatus::Unsupported, "device does not advertise ext_intel_legacy_image", fetchNote);
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
    test.skip("float4", ResultStatus::Error, "Output buffer alloc failed", fetchNote);
    return -1;
  }

  try
  {
    sycl::image<2> img(staging,
                       sycl::image_channel_order::rgba,
                       sycl::image_channel_type::fp32,
                       sycl::range<2>(imgW, imgH));

    // walk == 0 decomposes the linear index row-major (x fastest); walk == 1
    // transposes it (y fastest).  Both cover every pixel exactly once, so the
    // byte count is identical and the two rates are directly comparable; the
    // host races them and reports the faster.  No single walk suits every
    // image layout -- reading 32 texels along x is ideal for a linear surface
    // but hits scattered chunks of a tiled one, and the transpose is the mirror
    // image.  Same reasoning as the raced MAD-chain shapes; see
    // include/common/common.h, and the Vulkan and CUDA image_bandwidth.cpp for
    // the NVIDIA measurements that motivated it.
    // Generic lambda + integral_constant rather than a templated lambda: the
    // project builds as C++17, where the latter is not available.
    auto submit = [&](auto walkTag) {
      constexpr int WALK = decltype(walkTag)::value;
      return [&](sycl::queue &q) -> sycl::event {
        return q.submit([&](sycl::handler &h) {
          // Unsampled image accessor: read coordinates as int2, get a float4 back.
          sycl::accessor<sycl::float4, 2, sycl::access::mode::read,
                         sycl::access::target::image>
              acc(img, h);

          h.parallel_for<image_bw_kernel<WALK>>(
            sycl::nd_range<1>(globalThreads, blockSize),
            [=](sycl::nd_item<1> it) {
              uint32_t gid = (uint32_t)it.get_global_id(0);
              // IMAGE_FETCH_PER_WI samples per WI.  Coordinates wrap to keep
              // all lanes in-bounds.
              uint32_t base = gid * IMAGE_FETCH_PER_WI;
              sycl::float4 sum{0.0f, 0.0f, 0.0f, 0.0f};
              #pragma unroll
              for (int i = 0; i < (int)IMAGE_FETCH_PER_WI; i++)
              {
                uint32_t idx = base + i;
                int x, y;
                if constexpr (WALK == 0)
                {
                  x = (int)(idx % (uint32_t)imgW);
                  y = (int)((idx / (uint32_t)imgW) % (uint32_t)imgH);
                }
                else
                {
                  y = (int)(idx % (uint32_t)imgH);
                  x = (int)((idx / (uint32_t)imgH) % (uint32_t)imgW);
                }
                sum += acc.read(sycl::int2{x, y});
              }
              outBuf[gid] = sum.x() + sum.y() + sum.z() + sum.w();
            });
        });
      };
    };

    unsigned int forced = forceIters ? specifiedIters : 0;
    float rowUs = runKernel(dev, submit(std::integral_constant<int, 0>{}),
                            cfg.targetTimeUs, forced);
    float colUs = runKernel(dev, submit(std::integral_constant<int, 1>{}),
                            cfg.targetTimeUs, forced);
    const uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads;
    float rowGbps = rowUs > 0.0f ? (float)bytes / rowUs / 1e3f : 0.0f;
    float colGbps = colUs > 0.0f ? (float)bytes / colUs / 1e3f : 0.0f;
    CLPEAK_VLOG("image_memory_bandwidth: row-major %.1f, column-major %.1f gbps\n",
                rowGbps, colGbps);

    if (rowGbps <= 0.0f && colGbps <= 0.0f)
      test.skip("float4", ResultStatus::Error, "kernel launch failed", fetchNote);
    else
      test.emit("float4", std::max(rowGbps, colGbps), fetchNote);
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
