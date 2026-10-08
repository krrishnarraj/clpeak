#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runComputeInt32(RocmDevice &dev, benchmark_config_t &cfg)
{
  // Native HIP SDK vector widths: int, int2, int4. Each variant does the same
  // 4096 ops/thread (loop count divided by the vector width).
  static const rocm_compute_variant_t variants[] = {
      {"int", "compute_int32", &rocm_kernels::compute_int32, rocmWidthNote(1)},
      {"int2", "compute_int32_v2", &rocm_kernels::compute_int32, rocmWidthNote(2)},
      {"int4", "compute_int32_v4", &rocm_kernels::compute_int32, rocmWidthNote(4)},
  };
  int A = 3;
  rocm_compute_desc_t d = {};
  d.title = "Integer compute (32-bit IMAD)";
  d.resultTag = "integer_compute";
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "ops";
  d.description = "Peak 32-bit integer arithmetic rate.";
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
  // Where dp8 stops gaining on dp4, the issue rate is the limit, not latency.
  static const rocm_compute_variant_t variants[] = {
      {"int8_dp", "compute_int8_dp", &rocm_kernels::compute_int8_dp,
       "One dependent chain of dot products."},
      {"int8_dp2", "compute_int8_dp2", &rocm_kernels::compute_int8_dp,
       "2 independent chains."},
      {"int8_dp4", "compute_int8_dp4", &rocm_kernels::compute_int8_dp,
       "4 independent chains."},
      {"int8_dp8", "compute_int8_dp8", &rocm_kernels::compute_int8_dp,
       "8 independent chains."},
  };
  int A = 4;
  rocm_compute_desc_t d = {};
  d.title = "INT8 dot-product compute (DP4a)";
  d.resultTag = "integer_compute_int8_dp";
  d.shape = TestShape::Homogeneous;
  // Independent DP4a chains, not wider vectors: each reading gives the
  // hardware more dot products to have in flight at once.
  d.axis = "chains in flight";
  d.unit = "ops";
  d.description = "Peak rate of the 4-way int8 dot-product instruction, "
                  "without the matrix engine.";
  d.variants = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI = COMPUTE_INT8_DP_WORK_PER_WI;
  d.elemSize = sizeof(int);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  // Vega 10, the GCN5 APUs, gfx1010 and gfx1013 have no packed int8 dot
  // instruction; the kernel's arch group leaves them out.
  d.notBuilt = "No 8-bit dot-product instruction on this GPU";
  return runComputeKernel(dev, cfg, d);
}

#endif // ENABLE_ROCM
