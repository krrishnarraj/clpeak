#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runImageBandwidth(RocmDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"image_memory_bandwidth", "Image memory bandwidth", "gbps"});

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
    test.skip("float4", ResultStatus::Error, "Image array create failed");
    return -1;
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
      hipFreeArray(arr);
      test.skip("float4", ResultStatus::Error, "Image upload failed");
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
    hipFreeArray(arr);
    test.skip("float4", ResultStatus::Error, "Texture object create failed");
    return -1;
  }

  void *outBuf = nullptr;
  if (hipMalloc(&outBuf, globalThreads * sizeof(float)) != hipSuccess)
  {
    hipDestroyTextureObject(tex);
    hipFreeArray(arr);
    test.skip("float4", ResultStatus::Error, "Output buffer alloc failed");
    return -1;
  }

  hipFunction_t fn;
  if (!dev.getKernel(rocm_kernels::image_bandwidth_src,
                     rocm_kernels::image_bandwidth_name,
                     "image_bandwidth", fn))
  {
    hipFree(outBuf);
    hipDestroyTextureObject(tex);
    hipFreeArray(arr);
    test.skip("float4", ResultStatus::Error, "Kernel compile failed");
    return -1;
  }

  int w = imgW, h = imgH;
  void *args[4] = {&tex, &outBuf, &w, &h};
  float us = runKernel(dev, fn, numBlocks, blockSize, args,
                       cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
  {
    test.skip("float4", ResultStatus::Error, "kernel launch failed");
  }
  else
  {
    uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads;
    test.emit("float4", (float)bytes / us / 1e3f);
  }

  hipFree(outBuf);
  hipDestroyTextureObject(tex);
  hipFreeArray(arr);
  return 0;
}

#endif // ENABLE_ROCM
