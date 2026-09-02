#include <opencl/cl_peak.h>
#include <algorithm>

int clPeak::runImageBandwidthTest(cl::CommandQueue &queue, cl::Program &prog, device_info_t &devInfo, benchmark_config_t &cfg)
{
  float timed, bps;
  cl::NDRange globalSize, localSize;

  if (!isAllowed(Benchmark::ImageBW))
    return 0;

  auto test = currentDeviceScope->beginTest(
    {"image_memory_bandwidth", "Image memory bandwidth", "bps",
     Category::Unknown,
     "How many bytes per second the device reads through its texture units, "
     "which take a different path to memory than plain buffer reads.  Each "
     "pixel of the image is read exactly once, so caching cannot flatter the "
     "number.",
     TestShape::Homogeneous});

  // The image is RGBA float, so one fetch returns a whole pixel: four 32-bit
  // values, hence the metric name.
  const char *fetchNote = "Each fetch returns one whole pixel -- four 32-bit "
                          "colour values, 16 bytes.";

  if (!devInfo.imageSupported)
  {
    test.skip("float4", ResultStatus::Unsupported,
               "Device has no image support", fetchNote);
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

      // Two walk orders are raced and the faster reported -- why, and why
      // neither can flatter the result: the image-bandwidth block in
      // include/common/common.h.
      auto timeWalk = [&](cl_int walk) -> float {
        kernel_v1.setArg(2, walk);
        return run_kernel(queue, kernel_v1, globalSize, localSize,
                          cfg.targetTimeUs, forced);
      };
      float rowUs = timeWalk(0);
      float colUs = timeWalk(1);

      // Each WI reads IMAGE_FETCH_PER_WI float4 pixels = IMAGE_FETCH_PER_WI * 4 * sizeof(float) bytes
      uint64_t bytesPerCall = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(cl_float) * ndRangeTotal(globalSize);
      float rowBps = rowUs > 0.0f ? (float)bytesPerCall / rowUs * 1e6f : 0.0f;
      float colBps = colUs > 0.0f ? (float)bytesPerCall / colUs * 1e6f : 0.0f;
      CLPEAK_VLOG("image_memory_bandwidth: row-major %.1f, column-major %.1f B/s\n",
                  rowBps, colBps);
      bps = std::max(rowBps, colBps);
      (void)timed;

      test.emit("float4", bps, fetchNote);
    }
    ///////////////////////////////////////////////////////////////////////////
  }
  catch (cl::Error &error)
  {
    std::string reason = std::string(error.what()) + " (" + std::to_string(error.err()) + ")";
    test.skip("float4", ResultStatus::Error, reason, fetchNote);
    return -1;
  }

  return 0;
}
