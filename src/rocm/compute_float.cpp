#ifdef ENABLE_ROCM

#include <rocm/rocm_peak.h>
#include <common/common.h>

using xpupeak::Dim; using xpupeak::DType;

int RocmPeak::runComputeSP(RocmDevice &dev, benchmark_config_t &cfg)
{
  // Native HIP SDK vector widths: float, float2, float4 (hip_vector_types.h
  // has no float8/float16). Each variant does the same 4096 flops/thread.
  static const rocm_compute_variant_t variants[] = {
      {"float", "compute_sp", &rocm_kernels::compute_sp, rocmWidthNote(1), {{Dim::Width, 1}}},
      {"float2", "compute_sp_v2", &rocm_kernels::compute_sp, rocmWidthNote(2), {{Dim::Width, 2}}},
      {"float4", "compute_sp_v4", &rocm_kernels::compute_sp, rocmWidthNote(4), {{Dim::Width, 4}}},
  };
  float A = 1.3f;
  rocm_compute_desc_t d = {};
  d.title = "Single-precision compute";
  d.resultTag = "single_precision_compute";
  // 2: its MAD chain went from two ping-ponged registers to one accumulator
  // (2.1.0) after it first shipped, which moved its numbers.
  d.revision = 2;
  d.dims = {{Dim::Dtype, DType::Fp32}};
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak fp32 arithmetic rate of the GPU's ALUs, with no memory "
                  "traffic.";
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
      {"half", "compute_hp", &rocm_kernels::compute_hp, rocmWidthNote(1), {{Dim::Width, 1}}},
      {"half2", "compute_hp2", &rocm_kernels::compute_hp,
       "Two fp16 values packed in one instruction.", {{Dim::Width, 2}}},
  };
  float A = 1.3f;
  rocm_compute_desc_t d = {};
  d.title = "Half-precision compute";
  d.resultTag = "half_precision_compute";
  // 2: its MAD chain went from two ping-ponged registers to one accumulator
  // (2.1.0) after it first shipped, which moved its numbers.
  d.revision = 2;
  d.dims = {{Dim::Dtype, DType::Fp16}};
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak fp16 arithmetic rate, with fp16 inputs and accumulator.  "
                  "Only the packed half2 form reaches full rate.";
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
      {"double", "compute_dp", &rocm_kernels::compute_dp, rocmWidthNote(1), {{Dim::Width, 1}}},
      {"double2", "compute_dp_v2", &rocm_kernels::compute_dp, rocmWidthNote(2), {{Dim::Width, 2}}},
      {"double4", "compute_dp_v4", &rocm_kernels::compute_dp, rocmWidthNote(4), {{Dim::Width, 4}}},
  };
  double A = 1.3;
  rocm_compute_desc_t d = {};
  d.title = "Double-precision compute";
  d.resultTag = "double_precision_compute";
  // 2: its MAD chain went from two ping-ponged registers to one accumulator
  // (2.1.0) after it first shipped, which moved its numbers.
  d.revision = 2;
  d.dims = {{Dim::Dtype, DType::Fp64}};
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak fp64 arithmetic rate.  Consumer GPUs run it at a small "
                  "fraction of their fp32 rate.";
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
  // 2: its MAD chain went to one accumulator (2.1.0) and its fp16 round trip
  // out of the chain (2.1.4) after it first shipped, which moved its numbers.
  d.revision = 2;
  // The family's own accumulator is fp32, so the data type says it all.
  d.dims = {{Dim::Dtype, DType::Fp16}};
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak rate of fp16 multiplies accumulated in fp32, "
                  "without the matrix engine.";
  d.metricLabel = "mp";
  d.metricDims = {{Dim::Width, 1}};
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
  // 2: its MAD chain went from two ping-ponged registers to one accumulator
  // (2.1.0) after it first shipped, which moved its numbers.
  d.revision = 2;
  // bfloat16-compute's own accumulator is bf16: this one is fp32.
  d.dims = {{Dim::Dtype, DType::Bf16}, {Dim::Acc, DType::Fp32}};
  d.shape = TestShape::Homogeneous;
  d.axis = "vector width";
  d.unit = "flops";
  d.description = "Peak rate of bf16 multiplies accumulated in fp32, "
                  "without the matrix engine.";
  d.metricLabel = "bf16";
  d.metricDims = {{Dim::Width, 1}};
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
