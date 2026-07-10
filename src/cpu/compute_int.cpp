#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

using clpeak_cpu::kernelMenu;
using clpeak_cpu::kernels;

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

int CpuPeak::runComputeInt16DP(benchmark_config_t &cfg)
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  // int16 dot is x86-only today (VPDPWSSD in every VNNI CPU since Cascade Lake;
  // AVX-VNNI-INT16 adds the mixed-sign forms on Diamond Rapids / Nova Lake).
  // NEON/SVE have no 16-bit dot instruction, so skip the row on ARM builds.
  emitVariants(*this, {"int16_dot_product_compute", "INT16 dot-product compute", "gops"},
               "int16_dp", kernelMenu().int16dp, "no int16 dot instruction on this CPU", cfg);
#else
  (void)cfg;
#endif
  return 0;
}

int CpuPeak::runComputeIntDiv(benchmark_config_t &cfg)
{
  // Scalar u64 divide: there is no SIMD integer divide on x86 or NEON, so this
  // is a single test with no per-ISA variants -- the kernel is identical in
  // every TU and read straight from kernels().intdiv (always present via the
  // generic floor).  The number is the scalar DIV unit's throughput in Gops.
  const auto &v = kernels().intdiv;
  logger::TestSpec spec{"integer_divide_compute", "Integer divide compute u64", "gops"};
  auto test = currentDeviceScope->beginTest(spec);
  if (v.fn)
    emitCompute(*this, test, "u64", v.opsPerIter, v.fn, cfg);
  else
    test.skip("u64 ST", ResultStatus::Unsupported, "no integer divide kernel");
  return 0;
}

#endif // ENABLE_CPU
