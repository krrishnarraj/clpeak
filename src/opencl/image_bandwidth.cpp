#include <opencl/cl_peak.h>
#include <algorithm>

int clPeak::runImageBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  float timed, gbps;
  cl::NDRange globalSize, localSize;

  if (!isAllowed(Benchmark::ImageBW))
    return 0;

  auto test = currentDeviceScope->beginTest(
    {"image_memory_bandwidth", "Image memory bandwidth", "gbps"});

  if (!devInfo.imageSupported)
  {
    test.skip("float4", ResultStatus::Unsupported,
               "Device has no image support");
    return 0;
  }

  unsigned int forced = forceIters ? specifiedIters : 0;

  // Choose image dimensions: up to 4096x4096, bounded by device limits and maxAllocSize
  uint64_t imgW = std::min((uint64_t)4096, devInfo.image2dMaxWidth);
  uint64_t imgH = std::min((uint64_t)4096, devInfo.image2dMaxHeight);
  uint64_t bytesPerPixel = 4 * sizeof(cl_float); // RGBA float
  uint64_t imgBytes = imgW * imgH * bytesPerPixel;
  if (imgBytes > devInfo.maxAllocSize / 2)
  {
    imgH = (devInfo.maxAllocSize / 2) / (imgW * bytesPerPixel);
    if (imgH == 0)
      imgH = 1;
  }

  // Size the dispatch so each pixel is read exactly once per launch,
  // eliminating cache reuse that inflates apparent bandwidth.
  uint64_t groups = ((uint64_t)imgW * (uint64_t)imgH) / IMAGE_FETCH_PER_WI / devInfo.maxWGSize;
  if (groups == 0) groups = 1;
  uint64_t globalWIs = groups * devInfo.maxWGSize;

  try
  {
    cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

    cl::ImageFormat imgFmt(CL_RGBA, CL_FLOAT);
    cl::Image2D img(ctx, CL_MEM_READ_ONLY, imgFmt, (size_t)imgW, (size_t)imgH);

    // Fill image with pseudo-random data to defeat hardware memory compression.
    {
      size_t numFloats = (size_t)imgW * (size_t)imgH * 4;
      float *staging = new float[numFloats];
      populate(staging, numFloats);
      cl::array<cl::size_type, 3> origin = {0, 0, 0};
      cl::array<cl::size_type, 3> region = {(size_t)imgW, (size_t)imgH, 1};
      queue.enqueueWriteImage(img, CL_TRUE, origin, region, 0, 0, staging);
      delete[] staging;
    }

    cl::Buffer outputBuf = cl::Buffer(ctx, CL_MEM_WRITE_ONLY, globalWIs * sizeof(cl_float));

    globalSize = globalWIs;
    localSize  = devInfo.maxWGSize;

    ///////////////////////////////////////////////////////////////////////////
    // float4 -- read_imagef always returns float4 (RGBA)
    {
      cl::Kernel kernel_v1(prog, "image_bandwidth_v1");
      kernel_v1.setArg(0, img);
      kernel_v1.setArg(1, outputBuf);

      timed = run_kernel(queue, kernel_v1, globalSize, localSize, cfg.targetTimeUs, forced);

      // Each WI reads IMAGE_FETCH_PER_WI float4 pixels = IMAGE_FETCH_PER_WI * 4 * sizeof(float) bytes
      uint64_t bytesPerCall = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(cl_float) * globalWIs;
      gbps = (float)bytesPerCall / timed / 1e3f;

      test.emit("float4", gbps);
    }
    ///////////////////////////////////////////////////////////////////////////
  }
  catch (cl::Error &error)
  {
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    test.skip("float4", ResultStatus::Error, reason);
    return -1;
  }

  return 0;
}
