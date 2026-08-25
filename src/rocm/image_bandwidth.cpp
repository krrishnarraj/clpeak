#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>
#include <algorithm>

int RocmPeak::runImageBandwidth(RocmDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"image_memory_bandwidth", "Image memory bandwidth", "gbps",
     Category::Unknown,
     "How many bytes per second the GPU reads through its texture units, "
     "which take a different path to memory than plain buffer reads.  Each "
     "pixel of the image is read exactly once, so caching cannot flatter the "
     "number."});

  // RGBA float image, so one fetch returns a whole pixel: four 32-bit values,
  // hence the metric name.
  const char *fetchNote = "Each fetch returns one whole pixel -- four 32-bit "
                          "colour values, 16 bytes.";

  const int imgW = 4096, imgH = 4096;
  const uint32_t blockSize = 256;
  uint64_t groups = ((uint64_t)imgW * (uint64_t)imgH) / IMAGE_FETCH_PER_WI / blockSize;
  if (groups == 0) groups = 1;
  uint64_t globalThreads = groups * blockSize;
  uint32_t numBlocks = (uint32_t)groups;

  hipChannelFormatDesc desc = hipCreateChannelDesc<float4>();
  hipArray_t arr = nullptr;
  if (hipMallocArray(&arr, &desc, imgW, imgH) != hipSuccess)
  {
    // CDNA data-center GPUs (gfx9xx / MI-series) have no texture/image
    // hardware, so the array allocation legitimately fails -- this is a
    // device capability gap, not a benchmark error. (The OpenCL backend
    // reports the same device as "Device has no image support".)
    test.skip("float4", ResultStatus::Unsupported,
               "Device has no image/texture support", fetchNote);
    return 0;
  }

  {
    size_t numFloats = (size_t)imgW * (size_t)imgH * 4;
    float *staging = new float[numFloats];
    populate(staging, numFloats);
    hipError_t copyStatus = hipMemcpy2DToArray(
        arr, 0, 0, staging, (size_t)imgW * 4 * sizeof(float),
        (size_t)imgW * 4 * sizeof(float), (size_t)imgH,
        hipMemcpyHostToDevice);
    delete[] staging;
    if (copyStatus != hipSuccess)
    {
      (void)hipFreeArray(arr);
      test.skip("float4", ResultStatus::Error,
               "Image upload failed", fetchNote);
      return -1;
    }
  }

  hipResourceDesc rd = {};
  rd.resType = hipResourceTypeArray;
  rd.res.array.array = arr;

  hipTextureDesc td = {};
  td.addressMode[0] = hipAddressModeClamp;
  td.addressMode[1] = hipAddressModeClamp;
  td.filterMode = hipFilterModePoint;
  td.readMode = hipReadModeElementType;
  td.normalizedCoords = 0;

  hipTextureObject_t tex = 0;
  if (hipCreateTextureObject(&tex, &rd, &td, nullptr) != hipSuccess)
  {
    (void)hipFreeArray(arr);
    test.skip("float4", ResultStatus::Error,
               "Texture object create failed", fetchNote);
    return -1;
  }

  void *outBuf = nullptr;
  if (hipMalloc(&outBuf, globalThreads * sizeof(float)) != hipSuccess)
  {
    (void)hipDestroyTextureObject(tex);
    (void)hipFreeArray(arr);
    test.skip("float4", ResultStatus::Error,
               "Output buffer alloc failed", fetchNote);
    return -1;
  }

  hipFunction_t fn;
  if (!dev.getKernel(rocm_kernels::image_bandwidth,
                     "image_bandwidth", fn))
  {
    (void)hipFree(outBuf);
    (void)hipDestroyTextureObject(tex);
    (void)hipFreeArray(arr);
    test.skip("float4", ResultStatus::Error,
               "Kernel compile failed", fetchNote);
    return -1;
  }

  int w = imgW, h = imgH;
  int walk = 0;
  void *args[5] = {&tex, &outBuf, &w, &h, &walk};
  const uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads;

  // Two walk orders are raced and the faster reported.  Neither can flatter the
  // result: both read every pixel exactly once, so the byte count is identical
  // and the only difference is how the reads land in the image's memory layout.
  // No single walk suits every layout -- a warp reading 32 texels along x is
  // ideal for a linear surface but hits 8 scattered chunks of a block-linear
  // one, and the transposed walk is the mirror image.  Measured on an RTX 5060:
  // CUDA reads 270 GBPS row-major and 419 transposed while Vulkan gets ~415
  // either way, so a row-major-only test made the CUDA texture path look 1.5x
  // slower when both in fact reach the card's full memory rate.  Same reasoning
  // as the raced MAD-chain shapes -- see include/common/common.h.
  auto timeWalk = [&](int w_) {
    walk = w_;
    return runKernel(dev, fn, numBlocks, blockSize, args,
                     cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  };
  float rowUs = timeWalk(0);
  float colUs = timeWalk(1);
  float rowGbps = rowUs > 0.0f ? (float)bytes / rowUs / 1e3f : 0.0f;
  float colGbps = colUs > 0.0f ? (float)bytes / colUs / 1e3f : 0.0f;
  CLPEAK_VLOG("image_memory_bandwidth: row-major %.1f, column-major %.1f gbps\n",
              rowGbps, colGbps);

  if (rowGbps <= 0.0f && colGbps <= 0.0f)
    test.skip("float4", ResultStatus::Error, "kernel launch failed", fetchNote);
  else
    test.emit("float4", std::max(rowGbps, colGbps), fetchNote);

  (void)hipFree(outBuf);
  (void)hipDestroyTextureObject(tex);
  (void)hipFreeArray(arr);
  return 0;
}

#endif // ENABLE_ROCM
