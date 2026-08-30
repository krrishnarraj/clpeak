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
  emitVariants(*this, {"single_precision_compute", "Single-precision compute", "gflops",
                       Category::Unknown,
                       "Peak arithmetic speed on 32-bit fractional numbers -- the "
                       "ordinary float type most programs use.  The data stays in "
                       "registers, so only the CPU's vector units limit the rate."},
               "float", kernelMenu().fp32, "no SIMD fp32 path for this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeDP(benchmark_config_t &cfg)
{
  emitVariants(*this, {"double_precision_compute", "Double-precision compute", "gflops",
                       Category::Unknown,
                       "Peak arithmetic speed on 64-bit fractional numbers, the "
                       "high-accuracy type scientific and engineering code relies on."},
               "double", kernelMenu().fp64, "no SIMD fp64 path for this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeHP(benchmark_config_t &cfg)
{
  emitVariants(*this, {"half_precision_compute", "Half-precision compute", "gflops",
                       Category::Unknown,
                       "Peak arithmetic speed on 16-bit fractional numbers -- half the "
                       "size of a normal float, and common in graphics and AI.  Needs "
                       "real 16-bit instructions, not conversion to 32-bit and back."},
               "half", kernelMenu().fp16, "no native fp16 arithmetic on this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeBF16(benchmark_config_t &cfg)
{
  emitVariants(*this, {"bfloat16_compute", "BF16 compute bf16xbf16+fp32", "gflops",
                       Category::Unknown,
                       "Peak speed of the bfloat16 dot-product instruction, which "
                       "multiplies pairs of 16-bit AI-format numbers and adds the "
                       "results into a 32-bit running total."},
               "bf16", kernelMenu().bf16, "no bf16 dot instruction on this CPU", cfg);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  // Native full-rate bf16 vector FMA is AVX10.2-only (x86 exclusive) -- a
  // genuinely different peak from the bf16 dot above (real bf16 multiply-add, no
  // fp32-accumulate widening).  Only emit it on an x86 build; on ARM there is no
  // native-bf16-FMA instruction, so the row would be meaningless noise.
  emitVariants(*this, {"bfloat16_fma_compute", "BF16 FMA compute bf16xbf16+bf16", "gflops",
                       Category::Unknown,
                       "Peak speed of true bfloat16 multiply-add, which keeps the "
                       "answer in 16 bits instead of widening it to 32.  Only the "
                       "newest x86 chips (AVX10.2) have the instruction."},
               "bf16_fma", kernelMenu().bf16fma,
               "no native bf16 vector FMA (AVX10.2) on this CPU", cfg);
#endif
  return 0;
}

int CpuPeak::runComputeDivSqrt(benchmark_config_t &cfg)
{
  // Divider/sqrt-unit throughput -- ops are divides (or sqrts) per second, so
  // the numbers are far below the FMA rows by design (the units are narrow,
  // partially pipelined, and this is where CPU generations differ 5-10x).
  //
  // One test, four readings: divide and square root run on the same narrow
  // unit, and its profile across the two operations and both precisions is
  // one fact about the CPU, not four.
  emitFamily(*this,
             {"divide_sqrt", "Divide and square-root throughput", "gflops",
              Category::Unknown,
              "How many divisions and square roots per second the CPU sustains. "
              "Both run on their own narrow unit rather than the wide "
              "multiply-add pipeline, so these land far below the compute rows "
              "-- and the gap between CPU generations here is much larger.",
              TestShape::Heterogeneous, "operation"},
             {{ "fdiv fp32", "Division on 32-bit floats.",
                "no SIMD fp32 divide path for this CPU", &kernelMenu().div32 },
              { "fdiv fp64", "Division on 64-bit floats, usually around half "
                             "the 32-bit rate.",
                "no SIMD fp64 divide path for this CPU", &kernelMenu().div64 },
              { "fsqrt fp32", "Square root on 32-bit floats.",
                "no SIMD fp32 sqrt path for this CPU", &kernelMenu().sqrt32 },
              { "fsqrt fp64", "Square root on 64-bit floats.",
                "no SIMD fp64 sqrt path for this CPU", &kernelMenu().sqrt64 }},
             cfg);
  return 0;
}

int CpuPeak::runComputeMP(benchmark_config_t &cfg)
{
  emitVariants(*this, {"mixed_precision_compute", "Mixed-precision compute fp16xfp16+fp32", "gflops",
                       Category::Unknown,
                       "Peak speed when the CPU multiplies 16-bit numbers but keeps "
                       "the running total in 32 bits -- the accuracy-preserving "
                       "pattern AI code uses, done in one instruction with no "
                       "conversion step."},
               "mp", kernelMenu().mp, "no conversion-free fp16xfp16+fp32 widening FMA on this CPU", cfg);
  return 0;
}

int CpuPeak::runComputeFP8DP(benchmark_config_t &cfg)
{
#if defined(__aarch64__) || defined(_M_ARM64)
  // Native fp8 vector dot is ARM-only today (FEAT_FP8DOT4 -- NVIDIA Vera first;
  // x86 has no fp8 *vector* instruction, only the AMX-FP8 tile path, which is
  // its own matrix row).  Only emit on an arm64 build, mirroring how the
  // x86-only bf16-FMA row is handled.
  emitVariants(*this, {"fp8_dot_product_compute", "FP8 dot-product compute fp8xfp8+fp32", "gflops",
                       Category::Unknown,
                       "Peak speed of the 8-bit float dot product, the narrowest "
                       "number format a CPU can multiply directly.  Only the newest "
                       "Arm cores have the instruction."},
               "fp8_dp", kernelMenu().fp8dp, "no fp8 dot instruction (FEAT_FP8DOT4) on this CPU", cfg);
#else
  (void)cfg;
#endif
  return 0;
}

#endif // ENABLE_CPU
