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

int RocmPeak::runComputeInt8DP(RocmDevice &, benchmark_config_t &)
{
  auto test = currentDeviceScope->beginTest(
    {"integer_compute_int8_dp", "INT8 dot-product compute", "gops"});
  test.skip("int8_dp", ResultStatus::Unsupported,
            "native AMD DP4a/MFMA dot-product path not implemented in ROCm backend yet");
  test.skip("int8_dp2", ResultStatus::Unsupported,
            "native AMD DP4a/MFMA dot-product path not implemented in ROCm backend yet");
  test.skip("int8_dp4", ResultStatus::Unsupported,
            "native AMD DP4a/MFMA dot-product path not implemented in ROCm backend yet");
  test.skip("int8_dp8", ResultStatus::Unsupported,
            "native AMD DP4a/MFMA dot-product path not implemented in ROCm backend yet");
  return 0;
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
