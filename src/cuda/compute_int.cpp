#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// Integer compute benchmarks
// ---------------------------------------------------------------------------

int CudaPeak::runComputeInt32(CudaDevice &dev, benchmark_config_t &cfg)
{
  // Scalar 32-bit integer IMAD chain throughput.  Distinct shader-core
  // path from __dp4a (compute_int8_dp) and the int4 emulation; reported
  // in GOPS.
  int A = 3;
  cuda_compute_desc_t d = {};
  d.title = "Integer compute (32-bit IMAD)";
  d.resultTag = "integer_compute";
  d.unit = "gops";
  d.description = "Peak speed on 32-bit whole numbers -- the arithmetic behind "
                  "indexing, addressing and bit manipulation, which shaders do "
                  "alongside their fractional maths.";
  d.metricLabel = "int";
  d.kernelName = "compute_int32";
  d.blob = &cuda_kernels::compute_int32;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI; // 4096 ops/thread (same scaling)
  d.elemSize = sizeof(int);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeInt8DP(CudaDevice &dev, benchmark_config_t &cfg)
{
  // These variants are independent chains, not wider vectors -- so they carry
  // their own notes rather than cudaWidthNote().
  static const cuda_compute_variant_t variants[] = {
      {"int8_dp", "compute_int8_dp", &cuda_kernels::compute_int8_dp,
       "One chain of dot products, each waiting on the one before it."},
      {"int8_dp2", "compute_int8_dp2", &cuda_kernels::compute_int8_dp,
       "Two independent chains, so the GPU has a second dot product to get on "
       "with while the first is still finishing."},
      {"int8_dp4", "compute_int8_dp4", &cuda_kernels::compute_int8_dp,
       "Four independent chains."},
      {"int8_dp8", "compute_int8_dp8", &cuda_kernels::compute_int8_dp,
       "Eight independent chains.  Where this stops improving on four, the "
       "hardware itself is the limit, not the waiting."},
  };
  int A = 4;
  cuda_compute_desc_t d = {};
  d.title = "INT8 dot-product compute (__dp4a)";
  d.resultTag = "integer_compute_int8_dp";
  d.unit = "gops";
  d.description = "Peak speed of the 8-bit dot-product instruction, which multiplies "
                  "four pairs of small whole numbers and sums them in one step -- the "
                  "shader-core path for quantized (compressed) neural networks.";
  d.variants = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI = COMPUTE_INT8_DP_WORK_PER_WI;
  d.elemSize = sizeof(int);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  d.skip = !dev.info.dp4aSupported;
  d.skipMsg = "__dp4a requires sm_61 or newer (Pascal+)! Skipped";
  return runComputeKernel(dev, cfg, d);
}

// ---------------------------------------------------------------------------
// WMMA + FP8 mma.sync umbrella -- mirrors vkPeak::runCoopMatrix.
// ---------------------------------------------------------------------------

#endif // ENABLE_CUDA
