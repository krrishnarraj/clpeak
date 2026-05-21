#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>

int CudaPeak::runImageBandwidth(CudaDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"image_memory_bandwidth", "Image memory bandwidth", "gbps"});

  const int imgW = 4096, imgH = 4096;
  const uint32_t blockSize = 256;
  // Match OpenCL: scale to CU count without a floor so we don't
  // oversubscribe a fixed-size image and inflate cache reuse.
  uint64_t globalThreads = (uint64_t)dev.info.numSMs * cfg.computeWgsPerCU * blockSize;
  uint32_t numBlocks = (uint32_t)(globalThreads / blockSize);

  // Create CUarray (RGBA float).
  CUDA_ARRAY_DESCRIPTOR adesc = {};
  adesc.Width = imgW;
  adesc.Height = imgH;
  adesc.Format = CU_AD_FORMAT_FLOAT;
  adesc.NumChannels = 4;
  CUarray arr;
  if (cuArrayCreate(&arr, &adesc) != CUDA_SUCCESS)
  {
    test.skip("float4", ResultStatus::Error, "Image array create failed");
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
  td.flags = CU_TRSF_READ_AS_INTEGER; // we want raw float bits, no normalization
  CUtexObject tex = 0;
  if (cuTexObjectCreate(&tex, &rd, &td, nullptr) != CUDA_SUCCESS)
  {
    cuArrayDestroy(arr);
    test.skip("float4", ResultStatus::Error, "Texture object create failed");
    return -1;
  }

  CUdeviceptr outBuf = 0;
  cuMemAlloc(&outBuf, globalThreads * sizeof(float));

  CUfunction fn;
  if (!dev.getKernel(cuda_kernels::image_bandwidth_src,
                     cuda_kernels::image_bandwidth_name,
                     "image_bandwidth", fn))
  {
    test.skip("float4", ResultStatus::Error, "Kernel compile failed");
    cuTexObjectDestroy(tex);
    cuArrayDestroy(arr);
    cuMemFree(outBuf);
    return -1;
  }

  int w = imgW, h = imgH;
  void *args[4] = {&tex, &outBuf, &w, &h};
  float us = runKernel(dev, fn, numBlocks, blockSize, args,
                       cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  uint64_t bytes = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads;
  float gbps = (float)bytes / us / 1e3f;
  test.emit("float4", gbps);

  cuTexObjectDestroy(tex);
  cuArrayDestroy(arr);
  cuMemFree(outBuf);
  return 0;
}

// ---------------------------------------------------------------------------
// Atomic throughput (CUDA -- global + local atomics)
// ---------------------------------------------------------------------------


#endif // ENABLE_CUDA
