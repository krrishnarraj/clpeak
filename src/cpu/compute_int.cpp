#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

#include <string>

using clpeak_cpu::kernels;

int CpuPeak::runComputeInt32(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest({"integer_compute", "Integer compute", "gops"});
  const auto &v = kernels().int32;
  if (v.fn) emitCompute(*this, test, "int", v.opsPerIter, v.fn, cfg);
  else      test.skip("int", ResultStatus::Unsupported, "no SIMD int32 path for this CPU");
  return 0;
}

int CpuPeak::runComputeInt8DP(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"int8_dot_product_compute", "INT8 dot-product compute", "gops"});
  const auto &v = kernels().int8dp;
  if (v.fn) emitCompute(*this, test, "int8_dp", v.opsPerIter, v.fn, cfg);
  else      test.skip("int8_dp", ResultStatus::Unsupported, "no int8 dot instruction on this CPU");
  return 0;
}

#endif // ENABLE_CPU
