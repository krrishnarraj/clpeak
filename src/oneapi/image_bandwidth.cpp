#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <algorithm>
#include <sycl/sycl.hpp>

template <int TW, int TH, bool TRANSPOSE> class image_bw_kernel;

// One walk shape: TW x TH blocks walked row-major, row-major within a block
// (TW = the width with TH = 1 is the plain row-major sweep), or with
// TRANSPOSE the whole image transposed.  The Vulkan backend's shapes -- see
// the image-bandwidth block in include/common/common.h.
template <int TW, int TH, bool TRANSPOSE> struct ImageWalk {};
template <typename T> struct WalkOf;
template <int TW, int TH, bool TRANSPOSE> struct WalkOf<ImageWalk<TW, TH, TRANSPOSE>>
{
  static constexpr int tw = TW, th = TH;
  static constexpr bool transpose = TRANSPOSE;
};

int OneapiPeak::runImageBandwidth(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"image_memory_bandwidth", "Image memory bandwidth", "bps",
     Category::Unknown,
     "Image read bandwidth, reading each pixel once.",
     TestShape::Homogeneous});

  // RGBA float image, so one fetch returns a whole pixel: four 32-bit values,
  // hence the metric name.
  const char *fetchNote = "One RGBA fp32 pixel (16 bytes) per fetch.";

  if (!dev.dev.has(sycl::aspect::ext_intel_legacy_image))
  {
    test.skip("float4", ResultStatus::Unsupported, "device does not advertise ext_intel_legacy_image", fetchNote);
    return 0;
  }

  // Compile-time extents, so every coordinate in the walk is a shift and a
  // mask: a divide by a run-time value is a long emulated sequence on GPUs
  // without an integer divider, Intel's among them.
  constexpr int imgW = 4096, imgH = 4096;
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

    // The walk shapes are raced and the fastest reported -- why, and why
    // none can flatter the result: the image-bandwidth block in
    // include/common/common.h.  Every shape covers every pixel exactly once.
    // Generic lambda over an ImageWalk tag rather than a templated lambda:
    // the project builds as C++17, where the latter is not available.
    auto submit = [&](auto walkTag) {
      return [&, walkTag](sycl::queue &q) -> sycl::event {
        return q.submit([&](sycl::handler &h) {
          // Unsampled image accessor: read coordinates as int2, get a float4 back.
          sycl::accessor<sycl::float4, 2, sycl::access::mode::read,
                         sycl::access::target::image>
              acc(img, h);

          h.parallel_for<image_bw_kernel<WalkOf<decltype(walkTag)>::tw,
                                         WalkOf<decltype(walkTag)>::th,
                                         WalkOf<decltype(walkTag)>::transpose>>(
            sycl::nd_range<1>(globalThreads, blockSize),
            [=](sycl::nd_item<1> it) {
              using W = WalkOf<decltype(walkTag)>;
              uint32_t gid   = (uint32_t)it.get_global_id(0);
              uint32_t gsize = (uint32_t)globalThreads;
              // Stride the IMAGE_FETCH_PER_WI samples by the global size, so
              // adjacent work-items touch adjacent pixels of the walk.  This is
              // the pattern every other backend uses; reading a contiguous run
              // per work-item instead made this row incomparable with them.
              // No wrap: globalThreads * IMAGE_FETCH_PER_WI <= imgW * imgH.
              sycl::float4 sum{0.0f, 0.0f, 0.0f, 0.0f};
              #pragma unroll
              for (int i = 0; i < (int)IMAGE_FETCH_PER_WI; i++)
              {
                uint32_t pixel = gid + (uint32_t)i * gsize;
                int x, y;
                if constexpr (W::transpose)
                {
                  y = (int)(pixel % (uint32_t)imgH);
                  x = (int)(pixel / (uint32_t)imgH);
                }
                else
                {
                  constexpr uint32_t tilesX = (uint32_t)(imgW / W::tw);
                  constexpr uint32_t perTile = (uint32_t)(W::tw * W::th);
                  uint32_t t = pixel / perTile, in = pixel % perTile;
                  x = (int)((t % tilesX) * W::tw + in % W::tw);
                  y = (int)((t / tilesX) * W::th + in / W::tw);
                }
                sum += acc.read(sycl::int2{x, y});
              }
              outBuf[gid] = sum.x() + sum.y() + sum.z() + sum.w();
            });
        });
      };
    };

    unsigned int forced = forceIters ? specifiedIters : 0;
    const uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads;
    float best = 0.0f;
    auto timeWalk = [&](auto walkTag, const char *label) {
      float us = runKernel(dev, submit(walkTag), cfg.targetTimeUs, forced);
      float bps = us > 0.0f ? (float)bytes / us * 1e6f : 0.0f;
      CLPEAK_VLOG("image_memory_bandwidth: %-16s %.1f B/s\n", label, bps);
      best = std::max(best, bps);
    };
    timeWalk(ImageWalk<imgW, 1, false>{}, "row");
    timeWalk(ImageWalk<imgW, 1, true>{}, "col");
    timeWalk(ImageWalk<2, 2, false>{}, "tile2x2");
    timeWalk(ImageWalk<16, 16, false>{}, "tile16x16");

    if (best <= 0.0f)
      test.skip("float4", ResultStatus::Error, "kernel launch failed", fetchNote);
    else
      test.emit("float4", best, fetchNote);
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
