#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>

int CudaPeak::runImageBandwidth(CudaDevice &dev, benchmark_config_t &cfg)
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
  // Size the dispatch so each pixel is read exactly once per launch,
  // eliminating cache reuse that inflates apparent bandwidth.
  uint64_t groups = ((uint64_t)imgW * (uint64_t)imgH) / IMAGE_FETCH_PER_WI / blockSize;
  if (groups == 0) groups = 1;
  uint64_t globalThreads = groups * blockSize;
  uint32_t numBlocks = (uint32_t)groups;

  // Create CUarray (RGBA float).
  CUDA_ARRAY_DESCRIPTOR adesc = {};
  adesc.Width = imgW;
  adesc.Height = imgH;
  adesc.Format = CU_AD_FORMAT_FLOAT;
  adesc.NumChannels = 4;
  CUarray arr;
  if (cuArrayCreate(&arr, &adesc) != CUDA_SUCCESS)
  {
    test.skip("float4", ResultStatus::Error, "Image array create failed", fetchNote);
    return -1;
  }

  // Fill image with pseudo-random data to defeat hardware memory compression.
  {
    size_t numFloats = (size_t)imgW * (size_t)imgH * 4;
    float *staging = new float[numFloats];
    populate(staging, numFloats);
    CUDA_MEMCPY2D copy = {};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost       = staging;
    copy.srcPitch      = (size_t)imgW * 4 * sizeof(float);
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray      = arr;
    copy.WidthInBytes  = (size_t)imgW * 4 * sizeof(float);
    copy.Height        = (size_t)imgH;
    cuMemcpy2D(&copy);
    delete[] staging;
  }

  CUDA_RESOURCE_DESC rd = {};
  rd.resType = CU_RESOURCE_TYPE_ARRAY;
  rd.res.array.hArray = arr;
  CUDA_TEXTURE_DESC td = {};
  td.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  td.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  td.filterMode = CU_TR_FILTER_MODE_POINT;
  td.flags = 0; // no normalization needed for float textures
  CUtexObject tex = 0;
  if (cuTexObjectCreate(&tex, &rd, &td, nullptr) != CUDA_SUCCESS)
  {
    cuArrayDestroy(arr);
    test.skip("float4", ResultStatus::Error, "Texture object create failed", fetchNote);
    return -1;
  }

  CUdeviceptr outBuf = 0;
  cuMemAlloc(&outBuf, globalThreads * sizeof(float));

  CUfunction fn;
  if (!dev.getKernel(cuda_kernels::image_bandwidth,
                     "image_bandwidth", fn))
  {
    test.skip("float4", ResultStatus::Error, "Kernel compile failed", fetchNote);
    cuTexObjectDestroy(tex);
    cuArrayDestroy(arr);
    cuMemFree(outBuf);
    return -1;
  }

  int w = imgW, h = imgH;
  int walk = 0;   // row-major: this is the reported reading
  void *args[5] = {&tex, &outBuf, &w, &h, &walk};
  float us = runKernel(dev, fn, numBlocks, blockSize, args,
                       cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads;
  float gbps = (float)bytes / us / 1e3f;

  // --verbose-only layout probe.  Re-run the identical fetch count with the walk
  // transposed, so a warp reads 32 texels down a column instead of along a row.
  // A CUarray is always block-linear, and a GOB is several rows tall, so the
  // column stays inside it and the rate should only dip; a pitch-linear image
  // falls off a cliff.  The point of comparison is the same probe in the Vulkan
  // backend, which reports ~1.5x this test's figure for the same image on the
  // same card -- there the layout is the driver's choice and invisible to us.
  if (::clpeak::verboseEnabled() && us > 0.0f)
  {
    walk = 1;
    float colUs = runKernel(dev, fn, numBlocks, blockSize, args,
                            cfg.targetTimeUs, forceIters ? specifiedIters : 0);
    if (colUs > 0.0f)
      CLPEAK_VLOG("image_memory_bandwidth: row-major %.1f, column-major %.1f "
                  "gbps (%.2fx slower)\n", gbps, (float)bytes / colUs / 1e3f,
                  colUs / us);
  }

  test.emit("float4", gbps, fetchNote);

  cuTexObjectDestroy(tex);
  cuArrayDestroy(arr);
  cuMemFree(outBuf);
  return 0;
}

#endif // ENABLE_CUDA
