#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runComputeInt32(RocmDevice &dev, benchmark_config_t &cfg)
{
  int A = 3;
  rocm_compute_desc_t d = {};
  d.title = "Integer compute (32-bit IMAD)";
  d.resultTag = "integer_compute";
  d.unit = "gops";
  d.metricLabel = "int";
  d.kernelName = "compute_int32";
  d.src = rocm_kernels::compute_int32_src;
  d.srcName = rocm_kernels::compute_int32_name;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(int);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int RocmPeak::runComputeInt8DP(RocmDevice &dev, benchmark_config_t &cfg)
{
  // INT8 DP4a (v_dot4_i32_i8) vector-shader path -- distinct from the matrix
  // INT8 MFMA peak (runMfma). All four variants do 8192 ops/thread, so the
  // numbers are directly comparable; they differ only in ILP (chain count).
  static const rocm_compute_variant_t variants[] = {
      {"int8_dp", "compute_int8_dp", rocm_kernels::compute_int8_dp_src, rocm_kernels::compute_int8_dp_name},
      {"int8_dp2", "compute_int8_dp2", rocm_kernels::compute_int8_dp_src, rocm_kernels::compute_int8_dp_name},
      {"int8_dp4", "compute_int8_dp4", rocm_kernels::compute_int8_dp_src, rocm_kernels::compute_int8_dp_name},
      {"int8_dp8", "compute_int8_dp8", rocm_kernels::compute_int8_dp_src, rocm_kernels::compute_int8_dp_name},
  };
  int A = 4;
  rocm_compute_desc_t d = {};
  d.title = "INT8 dot-product compute (DP4a)";
  d.resultTag = "integer_compute_int8_dp";
  d.unit = "gops";
  d.variants = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI = COMPUTE_INT8_DP_WORK_PER_WI;
  d.elemSize = sizeof(int);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int RocmPeak::runComputeInt4Packed(RocmDevice &dev, benchmark_config_t &cfg)
{
  int A = 3;
  rocm_compute_desc_t d = {};
  d.title = "Packed INT4 compute (emulated)";
  d.resultTag = "int4_packed_compute";
  d.unit = "gops";
  d.metricLabel = "int4_packed";
  d.kernelName = "compute_int4_packed";
  d.src = rocm_kernels::compute_int4_packed_src;
  d.srcName = rocm_kernels::compute_int4_packed_name;
  d.workPerWI = COMPUTE_INT4_PACKED_WORK_PER_WI;
  d.elemSize = sizeof(int);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

#endif // ENABLE_ROCM
