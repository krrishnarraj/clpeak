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

  // Choose image dimensions: up to 4096x4096, bounded by device limits and
  // maxAllocSize, and each a power of two -- the kernel walks the image with
  // shifts and masks (see kernels/image_bandwidth_kernels.cl).
  auto floorPow2 = [](uint64_t v) {
    uint64_t p = 1;
    while (p * 2 <= v)
      p *= 2;
    return p;
  };
  auto log2Of = [](uint64_t pow2) {
    cl_int l = 0;
    while ((1ull << l) < pow2)
      l++;
    return l;
  };
  uint64_t imgW = floorPow2(std::min((uint64_t)4096, devInfo.image2dMaxWidth));
  uint64_t imgH = floorPow2(std::min((uint64_t)4096, devInfo.image2dMaxHeight));
  uint64_t bytesPerPixel = 4 * sizeof(cl_float); // RGBA float
  uint64_t imgBytes = imgW * imgH * bytesPerPixel;
  if (imgBytes > devInfo.maxAllocSize / 2)
    imgH = floorPow2(std::max<uint64_t>(1, (devInfo.maxAllocSize / 2) / (imgW * bytesPerPixel)));

  // Size the dispatch so each pixel is read exactly once per launch,
  // eliminating cache reuse that inflates apparent bandwidth.
  uint64_t groups = ((uint64_t)imgW * (uint64_t)imgH) / IMAGE_FETCH_PER_WI / devInfo.maxWGSize;
  if (groups == 0) groups = 1;
  uint64_t globalWIs = groups * devInfo.maxWGSize;

  try
  {
    // A program that failed to build is a null handle, and an older ICD
    // loader dereferences one rather than returning CL_INVALID_PROGRAM.
    if (!prog())
      throw cl::Error(CL_INVALID_PROGRAM, "clCreateKernel");
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

      // The Vulkan backend's walk shapes, raced and the fastest reported --
      // why, and why none can flatter the result: the image-bandwidth block
      // in include/common/common.h.  A block larger than the image is skipped.
      struct Shape { const char *label; cl_int walk; uint64_t tileW, tileH; };
      const Shape shapes[] = {
        { "row",       0, imgW, 1 },
        { "col",       1, imgW, 1 },
        { "tile2x2",   2,    2, 2 },
        { "tile16x16", 2,   16, 16 },
      };
      const cl_int logW = log2Of(imgW), logH = log2Of(imgH);

      // Each WI reads IMAGE_FETCH_PER_WI float4 pixels = IMAGE_FETCH_PER_WI * 4 * sizeof(float) bytes
      uint64_t bytesPerCall = (uint64_t)IMAGE_FETCH_PER_WI * 4 * sizeof(cl_float) * ndRangeTotal(globalSize);
      bps = 0.0f;
      for (const Shape &s : shapes)
      {
        if (s.tileW > imgW || s.tileH > imgH)
          continue;
        kernel_v1.setArg(2, s.walk);
        kernel_v1.setArg(3, logW);
        kernel_v1.setArg(4, logH);
        kernel_v1.setArg(5, log2Of(s.tileW));
        kernel_v1.setArg(6, log2Of(s.tileH));
        timed = run_kernel(queue, kernel_v1, globalSize, localSize,
                           cfg.targetTimeUs, forced);
        float shapeBps = timed > 0.0f ? (float)bytesPerCall / timed * 1e6f : 0.0f;
        CLPEAK_VLOG("image_memory_bandwidth: %-16s %.1f B/s\n", s.label, shapeBps);
        bps = std::max(bps, shapeBps);
      }

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
