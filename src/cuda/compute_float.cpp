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
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak arithmetic speed of the GPU's shader cores on 32-bit "
                  "fractional numbers -- the ordinary float type.  Nothing touches "
                  "memory, so only the arithmetic units limit the rate.";
  d.metricLabel = "float";
  d.kernelName = "compute_sp";
  d.blob = &cuda_kernels::compute_sp;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  return runComputeKernel(dev, cfg, d);
}

int CudaPeak::runComputeHP(CudaDevice &dev, benchmark_config_t &cfg)
{
  static const cuda_compute_variant_t variants[] = {
      {"half", "compute_hp", &cuda_kernels::compute_hp, cudaWidthNote(1)},
      {"half2", "compute_hp2", &cuda_kernels::compute_hp, cudaWidthNote(2)},
  };
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title = "Half-precision compute";
  d.resultTag = "half_precision_compute";
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak arithmetic speed on 16-bit fractional numbers -- half the "
                  "size of a normal float, and what graphics and on-device AI mostly "
                  "run on.  NVIDIA shader cores reach full speed only on the packed "
                  "form that does two at a time.";
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
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak arithmetic speed on 64-bit fractional numbers, the "
                  "high-accuracy type scientific computing relies on.  Consumer "
                  "GeForce cards run these dozens of times slower than 32-bit; the "
                  "datacenter parts do not.";
  d.metricLabel = "double";
  d.kernelName = "compute_dp";
  d.blob = &cuda_kernels::compute_dp;
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
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak speed when the GPU multiplies 16-bit numbers but keeps the "
                  "running total in 32 bits -- the accuracy-preserving pattern AI "
                  "code uses.  This is the shader cores, not the tensor cores.";
  d.metricLabel = "mp";
  d.kernelName = "compute_mp";
  d.blob = &cuda_kernels::compute_mp;
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
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak speed on bfloat16 -- 16 bits arranged for AI work, trading "
                  "digits of accuracy for the number range of a full float.  Again "
                  "the shader cores, with the tensor-core figure in the WMMA rows.";
  d.metricLabel = "bf16";
  d.kernelName = "compute_bf16";
  d.blob = &cuda_kernels::compute_bf16;
  d.workPerWI = COMPUTE_FP_WORK_PER_WI;
  d.elemSize = sizeof(float);
  d.scalarArg = &A;
  d.scalarSize = sizeof(A);
  d.skip = !dev.info.bf16Supported;
  d.skipMsg = "bf16 requires sm_80 or newer (Ampere+)! Skipped";
  return runComputeKernel(dev, cfg, d);
}

#endif // ENABLE_CUDA
