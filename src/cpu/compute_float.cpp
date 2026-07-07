#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

// FP compute tests.  The actual SIMD kernels are compiled per-ISA in
// cpu_kernels_tu.cpp and selected at runtime; here we just look up the chosen
// variant and emit (or record Unsupported when no variant exists for this CPU).

using clpeak_cpu::kernelMenu;

int CpuPeak::runComputeSP(benchmark_config_t &cfg)
{
  emitVariants(*this, {"single_precision_compute", "Single-precision compute", "gflops"},
               "float", kernelMenu().fp32, "no SIMD fp32 path for this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeDP(benchmark_config_t &cfg)
{
  emitVariants(*this, {"double_precision_compute", "Double-precision compute", "gflops"},
               "double", kernelMenu().fp64, "no SIMD fp64 path for this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeHP(benchmark_config_t &cfg)
{
  emitVariants(*this, {"half_precision_compute", "Half-precision compute", "gflops"},
               "half", kernelMenu().fp16, "no native fp16 arithmetic on this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeBF16(benchmark_config_t &cfg)
{
  emitVariants(*this, {"bfloat16_compute", "BF16 compute bf16xbf16+fp32", "gflops"},
               "bf16", kernelMenu().bf16, "no bf16 dot instruction on this CPU", cfg);
  // Native full-rate bf16 vector FMA (AVX10.2) -- a genuinely different peak from
  // the bf16 dot above (real bf16 multiply-add, no fp32-accumulate widening).
  emitVariants(*this, {"bfloat16_fma_compute", "BF16 FMA compute bf16xbf16+bf16", "gflops"},
               "bf16_fma", kernelMenu().bf16fma,
               "no native bf16 vector FMA (AVX10.2) on this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeMP(benchmark_config_t &cfg)
{
  emitVariants(*this, {"mixed_precision_compute", "Mixed-precision compute fp16xfp16+fp32", "gflops"},
               "mp", kernelMenu().mp, "no conversion-free fp16xfp16+fp32 widening FMA on this CPU", cfg);
  return 0;
}

#endif // ENABLE_CPU
