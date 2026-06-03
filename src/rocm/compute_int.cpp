#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runComputeInt32(RocmDevice &dev, benchmark_config_t &cfg)
{
  // Native HIP SDK vector widths: int, int2, int4. Each variant does the same
  // 4096 ops/thread (loop count divided by the vector width).
  static const rocm_compute_variant_t variants[] = {
      {"int", "compute_int32", rocm_kernels::compute_int32_src, rocm_kernels::compute_int32_name},
      {"int2", "compute_int32_v2", rocm_kernels::compute_int32_src, rocm_kernels::compute_int32_name},
      {"int4", "compute_int32_v4", rocm_kernels::compute_int32_src, rocm_kernels::compute_int32_name},
  };
  int A = 3;
  rocm_compute_desc_t d = {};
  d.title = "Integer compute (32-bit IMAD)";
  d.resultTag = "integer_compute";
  d.unit = "gops";
  d.variants = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
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
  // The DP4a builtin is absent on some archs (e.g. gfx12 lacks dot1-insts when
  // built against the legacy sdot4 path); suppress the multi-page HIPRTC log so
  // a failure shows just "[error] compile/load failed".
  d.quietCompile = true;
  return runComputeKernel(dev, cfg, d);
}

#endif // ENABLE_ROCM
