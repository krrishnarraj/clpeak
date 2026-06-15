#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

using clpeak_cpu::kernelMenu;

int CpuPeak::runComputeInt32(benchmark_config_t &cfg)
{
  emitVariants(*this, {"integer_compute", "Integer compute", "gops"},
               "int", kernelMenu().int32, "no SIMD int32 path for this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeInt8DP(benchmark_config_t &cfg)
{
  emitVariants(*this, {"int8_dot_product_compute", "INT8 dot-product compute", "gops"},
               "int8_dp", kernelMenu().int8dp, "no int8 dot instruction on this CPU", cfg);
  return 0;
}

#endif // ENABLE_CPU
