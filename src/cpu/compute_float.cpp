#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

// FP compute tests.  The actual SIMD kernels are compiled per-ISA in
// cpu_kernels_tu.cpp and selected at runtime; here we just look up the chosen
// variant and emit (or record Unsupported when no variant exists for this CPU).

using clpeak_cpu::kernels;

static void runChain(CpuPeak &peak, const logger::TestSpec &spec, const char *metric,
                     const clpeak_cpu::ChainVariant &v, const char *unsupReason,
                     benchmark_config_t &cfg)
{
  auto test = peak.currentDeviceScope->beginTest(spec);
  if (v.fn) emitCompute(peak, test, metric, v.opsPerIter, v.fn, cfg);
  else      test.skip(metric, ResultStatus::Unsupported, unsupReason);
}

int CpuPeak::runComputeSP(benchmark_config_t &cfg)
{
  runChain(*this, {"single_precision_compute", "Single-precision compute", "gflops"},
           "float", kernels().fp32, "no SIMD fp32 path for this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeDP(benchmark_config_t &cfg)
{
  runChain(*this, {"double_precision_compute", "Double-precision compute", "gflops"},
           "double", kernels().fp64, "no SIMD fp64 path for this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeHP(benchmark_config_t &cfg)
{
  runChain(*this, {"half_precision_compute", "Half-precision compute", "gflops"},
           "half", kernels().fp16, "no native fp16 arithmetic on this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeBF16(benchmark_config_t &cfg)
{
  runChain(*this, {"bfloat16_compute", "BF16 compute bf16xbf16+fp32", "gflops"},
           "bf16", kernels().bf16, "no bf16 dot instruction on this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeMP(benchmark_config_t &cfg)
{
  runChain(*this, {"mixed_precision_compute", "Mixed-precision compute fp16xfp16+fp32", "gflops"},
           "mp", kernels().mp, "no conversion-free fp16xfp16+fp32 widening FMA on this CPU", cfg);
  return 0;
}

#endif // ENABLE_CPU
