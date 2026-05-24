#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

int RocmPeak::runComputeSP(RocmDevice &dev, benchmark_config_t &cfg)
{
  float A = 1.3f;
  rocm_compute_desc_t d = {};
  d.title = "Single-precision compute";
  d.resultTag = "single_precision_compute";
  d.unit = "gflops";
  d.metricLabel = "float";
  d.kernelName = "compute_sp";
  d.src = rocm_kernels::compute_sp_src;
  d.srcName = rocm_kernels::compute_sp_name;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int RocmPeak::runComputeHP(RocmDevice &dev, benchmark_config_t &cfg)
{
  static const rocm_compute_variant_t variants[] = {
      {"half", "compute_hp", rocm_kernels::compute_hp_src, rocm_kernels::compute_hp_name},
      {"half2", "compute_hp2", rocm_kernels::compute_hp_src, rocm_kernels::compute_hp_name},
  };
  float A = 1.3f;
  rocm_compute_desc_t d = {};
  d.title = "Half-precision compute";
  d.resultTag = "half_precision_compute";
  d.unit = "gflops";
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
  double A = 1.3;
  rocm_compute_desc_t d = {};
  d.title = "Double-precision compute";
  d.resultTag = "double_precision_compute";
  d.unit = "gflops";
  d.metricLabel = "double";
  d.kernelName = "compute_dp";
  d.src = rocm_kernels::compute_dp_src;
  d.srcName = rocm_kernels::compute_dp_name;
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
  d.metricLabel = "mp";
  d.kernelName = "compute_mp";
  d.src = rocm_kernels::compute_mp_src;
  d.srcName = rocm_kernels::compute_mp_name;
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
  d.metricLabel = "bf16";
  d.kernelName = "compute_bf16";
  d.src = rocm_kernels::compute_bf16_src;
  d.srcName = rocm_kernels::compute_bf16_name;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  d.skip = !dev.info.bf16Supported;
  d.skipMsg = "bf16 not supported by this ROCm device! Skipped";
  return runComputeKernel(dev, cfg, d);
}

#endif // ENABLE_ROCM
