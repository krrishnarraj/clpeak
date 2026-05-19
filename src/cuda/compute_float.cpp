#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// Floating-point compute benchmarks
// ---------------------------------------------------------------------------

int CudaPeak::runComputeSP(CudaDevice &dev, benchmark_config_t &cfg)
{
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title = "Single-precision compute";
  d.resultTag = "single_precision_compute";
  d.unit = "gflops";
  d.metricLabel = "float";
  d.kernelName = "compute_sp";
  d.src = cuda_kernels::compute_sp_src;
  d.srcName = cuda_kernels::compute_sp_name;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeHP(CudaDevice &dev, benchmark_config_t &cfg)
{
  static const cuda_compute_variant_t variants[] = {
      {"half", "compute_hp", cuda_kernels::compute_hp_src, cuda_kernels::compute_hp_name},
      {"half2", "compute_hp2", cuda_kernels::compute_hp_src, cuda_kernels::compute_hp_name},
  };
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title = "Half-precision compute";
  d.resultTag = "half_precision_compute";
  d.unit = "gflops";
  d.variants = variants;
  d.numVariants = sizeof(variants) / sizeof(variants[0]);
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float); // 32-bit slot per thread; we store the reduced fp32 result
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  d.skip = !dev.info.fp16Supported;
  d.skipMsg = "fp16 not supported on this compute capability! Skipped";
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeDP(CudaDevice &dev, benchmark_config_t &cfg)
{
  double A = 1.3;
  cuda_compute_desc_t d = {};
  d.title = "Double-precision compute";
  d.resultTag = "double_precision_compute";
  d.unit = "gflops";
  d.metricLabel = "double";
  d.kernelName = "compute_dp";
  d.src = cuda_kernels::compute_dp_src;
  d.srcName = cuda_kernels::compute_dp_name;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(double);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeMP(CudaDevice &dev, benchmark_config_t &cfg)
{
  // Single variant: NVIDIA shader-core fp16xfp16+fp32 issues at FP32 rate.
  // The packed (HFMA2) path is fp16xfp16+fp16 -- that's compute_hp2, not MP.
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title = "Mixed-precision compute fp16xfp16+fp32";
  d.resultTag = "mixed_precision_compute";
  d.unit = "gflops";
  d.metricLabel = "mp";
  d.kernelName = "compute_mp";
  d.src = cuda_kernels::compute_mp_src;
  d.srcName = cuda_kernels::compute_mp_name;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  d.skip = !dev.info.fp16Supported;
  d.skipMsg = "fp16 not supported on this compute capability! Skipped";
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeBF16(CudaDevice &dev, benchmark_config_t &cfg)
{
  // Single variant: shader-core bf16xbf16+fp32 issues at FP32 rate on
  // Ampere+.  Packed BF16 is reachable through tensor cores (wmma), not
  // an SFU-style packed shader instruction, so a bf16_2 variant wouldn't
  // be a different code path.
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title = "BF16 compute bf16xbf16+fp32";
  d.resultTag = "bfloat16_compute";
  d.unit = "gflops";
  d.metricLabel = "bf16";
  d.kernelName = "compute_bf16";
  d.src = cuda_kernels::compute_bf16_src;
  d.srcName = cuda_kernels::compute_bf16_name;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  d.skip = !dev.info.bf16Supported;
  d.skipMsg = "bf16 requires sm_80 or newer (Ampere+)! Skipped";
  return runComputeKernel(dev, cfg, d);
}

#endif // ENABLE_CUDA
