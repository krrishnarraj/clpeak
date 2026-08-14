#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runComputeSP(RocmDevice &dev, benchmark_config_t &cfg)
{
  // Native HIP SDK vector widths: float, float2, float4 (hip_vector_types.h
  // has no float8/float16). Each variant does the same 4096 flops/thread.
  static const rocm_compute_variant_t variants[] = {
      {"float", "compute_sp", &rocm_kernels::compute_sp, rocmWidthNote(1)},
      {"float2", "compute_sp_v2", &rocm_kernels::compute_sp, rocmWidthNote(2)},
      {"float4", "compute_sp_v4", &rocm_kernels::compute_sp, rocmWidthNote(4)},
  };
  float A = 1.3f;
  rocm_compute_desc_t d = {};
  d.title = "Single-precision compute";
  d.resultTag = "single_precision_compute";
  d.unit = "gflops";
  d.description = "Peak arithmetic speed of the GPU's shader cores on 32-bit "
                  "fractional numbers -- the ordinary float type.  Nothing touches "
                  "memory, so only the arithmetic units limit the rate.";
  d.variants = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int RocmPeak::runComputeHP(RocmDevice &dev, benchmark_config_t &cfg)
{
  static const rocm_compute_variant_t variants[] = {
      {"half", "compute_hp", &rocm_kernels::compute_hp, rocmWidthNote(1)},
      {"half2", "compute_hp2", &rocm_kernels::compute_hp, rocmWidthNote(2)},
  };
  float A = 1.3f;
  rocm_compute_desc_t d = {};
  d.title = "Half-precision compute";
  d.resultTag = "half_precision_compute";
  d.unit = "gflops";
  d.description = "Peak arithmetic speed on 16-bit fractional numbers -- half the "
                  "size of a normal float, and what graphics and on-device AI mostly "
                  "run on.  AMD shader cores reach full speed only on the packed "
                  "form that does two at a time.";
  d.variants = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  d.skip = !dev.info.fp16Supported;
  d.skipMsg = "fp16 not supported by this ROCm device! Skipped";
  return runComputeKernel(dev, cfg, d);
}

int RocmPeak::runComputeDP(RocmDevice &dev, benchmark_config_t &cfg)
{
  // Native HIP SDK vector widths: double, double2, double4. Each variant does
  // the same 512 flops/thread (loop count divided by the vector width).
  static const rocm_compute_variant_t variants[] = {
      {"double", "compute_dp", &rocm_kernels::compute_dp, rocmWidthNote(1)},
      {"double2", "compute_dp_v2", &rocm_kernels::compute_dp, rocmWidthNote(2)},
      {"double4", "compute_dp_v4", &rocm_kernels::compute_dp, rocmWidthNote(4)},
  };
  double A = 1.3;
  rocm_compute_desc_t d = {};
  d.title = "Double-precision compute";
  d.resultTag = "double_precision_compute";
  d.unit = "gflops";
  d.description = "Peak arithmetic speed on 64-bit fractional numbers, the "
                  "high-accuracy type scientific computing relies on.  Radeon "
                  "gaming cards run these far slower than 32-bit; the Instinct "
                  "compute cards do not.";
  d.variants = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI = COMPUTE_DP_WORK_PER_WI;
  d.elemSize = sizeof(double);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int RocmPeak::runComputeMP(RocmDevice &dev, benchmark_config_t &cfg)
{
  float A = 1.3f;
  rocm_compute_desc_t d = {};
  d.title = "Mixed-precision compute fp16xfp16+fp32";
  d.resultTag = "mixed_precision_compute";
  d.unit = "gflops";
  d.description = "Peak speed when the GPU multiplies 16-bit numbers but keeps the "
                  "running total in 32 bits -- the accuracy-preserving pattern AI "
                  "code uses.  This is the shader cores, not the matrix cores.";
  d.metricLabel = "mp";
  d.kernelName = "compute_mp";
  d.blob = &rocm_kernels::compute_mp;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  d.skip = !dev.info.fp16Supported;
  d.skipMsg = "fp16 not supported by this ROCm device! Skipped";
  return runComputeKernel(dev, cfg, d);
}

int RocmPeak::runComputeBF16(RocmDevice &dev, benchmark_config_t &cfg)
{
  float A = 1.3f;
  rocm_compute_desc_t d = {};
  d.title = "BF16 compute bf16xbf16+fp32";
  d.resultTag = "bfloat16_compute";
  d.unit = "gflops";
  d.description = "Peak speed on bfloat16 -- 16 bits arranged for AI work, trading "
                  "digits of accuracy for the number range of a full float.  Again "
                  "the shader cores; the matrix-core figure is in the WMMA or MFMA "
                  "rows.";
  d.metricLabel = "bf16";
  d.kernelName = "compute_bf16";
  d.blob = &rocm_kernels::compute_bf16;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  d.skip = !dev.info.bf16Supported;
  d.skipMsg = "bf16 not supported by this ROCm device! Skipped";
  return runComputeKernel(dev, cfg, d);
}

#endif // ENABLE_ROCM
