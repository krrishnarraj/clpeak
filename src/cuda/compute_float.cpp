#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>

using xpupeak::Dim; using xpupeak::DType;

// ---------------------------------------------------------------------------
// Floating-point compute benchmarks
// ---------------------------------------------------------------------------

int CudaPeak::runComputeSP(CudaDevice &dev, benchmark_config_t &cfg)
{
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title = "Single-precision compute";
  d.resultTag = "single_precision_compute";
  // 2: its MAD chain went from two ping-ponged registers to one accumulator
  // (2.1.0) after it first shipped, which moved its numbers.
  d.revision = 2;
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak fp32 arithmetic rate of the GPU's ALUs, with no memory "
                  "traffic.";
  d.dims = {{Dim::Dtype, DType::Fp32}};
  d.metricLabel = "float";
  d.metricDims = {{Dim::Width, 1}};
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
      {"half", "compute_hp", &cuda_kernels::compute_hp, cudaWidthNote(1),
       {{Dim::Width, 1}}},
      {"half2", "compute_hp2", &cuda_kernels::compute_hp,
       "Two fp16 values packed in one instruction.", {{Dim::Width, 2}}},
  };
  float A = 1.3f;
  cuda_compute_desc_t d = {};
  d.title = "Half-precision compute";
  d.resultTag = "half_precision_compute";
  // 2: its MAD chain went from two ping-ponged registers to one accumulator
  // (2.1.0) after it first shipped, which moved its numbers.
  d.revision = 2;
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak fp16 arithmetic rate, with fp16 inputs and accumulator.";
  d.dims = {{Dim::Dtype, DType::Fp16}};
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
  // 2: its MAD chain went from two ping-ponged registers to one accumulator
  // (2.1.0) after it first shipped, which moved its numbers.
  d.revision = 2;
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak fp64 arithmetic rate.  Consumer GPUs run it at a small "
                  "fraction of their fp32 rate.";
  d.dims = {{Dim::Dtype, DType::Fp64}};
  d.metricLabel = "double";
  d.metricDims = {{Dim::Width, 1}};
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
  // 2: its MAD chain went to one accumulator (2.1.0) and its fp16 round trip
  // out of the chain (2.1.4) after it first shipped, which moved its numbers.
  d.revision = 2;
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak rate of fp16 multiplies accumulated in fp32, "
                  "without the matrix engine.";
  d.dims = {{Dim::Dtype, DType::Fp16}};   // the family's fp32 accumulator
  d.metricLabel = "mp";
  d.metricDims = {{Dim::Width, 1}};
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
  // 2: its MAD chain went from two ping-ponged registers to one accumulator
  // (2.1.0) after it first shipped, which moved its numbers.
  d.revision = 2;
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak rate of bf16 multiplies accumulated in fp32, "
                  "without the matrix engine.";
  // bfloat16-compute's own accumulator is bf16: this one is fp32.
  d.dims = {{Dim::Dtype, DType::Bf16}, {Dim::Acc, DType::Fp32}};
  d.metricLabel = "bf16";
  d.metricDims = {{Dim::Width, 1}};
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
