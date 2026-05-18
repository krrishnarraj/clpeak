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
  d.title = "Integer compute (32-bit IMAD) (GOPS)";
  d.resultTag = "integer_compute";
  d.unit = "gops";
  d.metricLabel = "int";
  d.kernelName = "compute_int32";
  d.src = cuda_kernels::compute_int32_src;
  d.srcName = cuda_kernels::compute_int32_name;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI; // 4096 ops/thread (same scaling)
  d.elemSize = sizeof(int);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeInt8DP(CudaDevice &dev, benchmark_config_t &cfg)
{
  static const cuda_compute_variant_t variants[] = {
      {"int8_dp", "compute_int8_dp", cuda_kernels::compute_int8_dp_src, cuda_kernels::compute_int8_dp_name},
      {"int8_dp2", "compute_int8_dp2", cuda_kernels::compute_int8_dp_src, cuda_kernels::compute_int8_dp_name},
      {"int8_dp4", "compute_int8_dp4", cuda_kernels::compute_int8_dp_src, cuda_kernels::compute_int8_dp_name},
      {"int8_dp8", "compute_int8_dp8", cuda_kernels::compute_int8_dp_src, cuda_kernels::compute_int8_dp_name},
  };
  int A = 4;
  cuda_compute_desc_t d = {};
  d.title = "INT8 dot-product compute (__dp4a) (GOPS)";
  d.resultTag = "integer_compute_int8_dp";
  d.unit = "gops";
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

int CudaPeak::runComputeInt4Packed(CudaDevice &dev, benchmark_config_t &cfg)
{
  int A = 3;
  cuda_compute_desc_t d = {};
  d.title = "Packed INT4 compute (emulated) (GOPS)";
  d.resultTag = "int4_packed_compute";
  d.unit = "gops";
  d.metricLabel = "int4_packed";
  d.kernelName = "compute_int4_packed";
  d.src = cuda_kernels::compute_int4_packed_src;
  d.srcName = cuda_kernels::compute_int4_packed_name;
  d.workPerWI = COMPUTE_INT4_PACKED_WORK_PER_WI;
  d.elemSize = sizeof(int);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

// ---------------------------------------------------------------------------
// WMMA + FP8 mma.sync umbrella -- mirrors vkPeak::runCoopMatrix.
// ---------------------------------------------------------------------------

#endif // ENABLE_CUDA
