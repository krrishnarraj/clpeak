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
  d.unit = "gops";
  d.description = "Peak speed on 32-bit whole numbers -- the arithmetic behind "
                  "indexing, addressing and bit manipulation, which shaders do "
                  "alongside their fractional maths.";
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
      {"int8_dp", "compute_int8_dp", &rocm_kernels::compute_int8_dp,
       "One chain of dot products, each waiting on the one before it."},
      {"int8_dp2", "compute_int8_dp2", &rocm_kernels::compute_int8_dp,
       "Two independent chains, so the GPU has a second dot product to get on "
       "with while the first is still finishing."},
      {"int8_dp4", "compute_int8_dp4", &rocm_kernels::compute_int8_dp,
       "Four independent chains."},
      {"int8_dp8", "compute_int8_dp8", &rocm_kernels::compute_int8_dp,
       "Eight independent chains.  Where this stops improving on four, the "
       "hardware itself is the limit, not the waiting."},
  };
  int A = 4;
  rocm_compute_desc_t d = {};
  d.title = "INT8 dot-product compute (DP4a)";
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
  // The DP4a builtin is absent on some archs (e.g. gfx12 lacks dot1-insts when
  // built against the legacy sdot4 path); such a compile failure shows just
  // "[error] compile/load failed" (the multi-page HIPRTC log is --verbose-only).
  return runComputeKernel(dev, cfg, d);
}

#endif // ENABLE_ROCM
