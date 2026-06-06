#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_simd.h"
#include "compute_common.h"

using namespace cpu_simd;

// ---------------------------------------------------------------------------
// FMA dependency chains.  Each kernel keeps F*_NACC independent accumulators
// (so the FMA latency*throughput product is hidden) updated by a loop-carried
// recurrence acc = acc*b + c with b<1 so the values converge to a finite fixed
// point (no inf / denormal that would perturb throughput).  The runtime trip
// count (outer) is what stops -O3/-ffast-math from constant-folding the chain.
// ---------------------------------------------------------------------------
static double runFp32Chain(uint64_t outer)
{
  f32v acc[F32_NACC];
  const f32v b = f32_set(0.999999f);
  const f32v c = f32_set(0.000001f);
  for (int j = 0; j < F32_NACC; j++) acc[j] = f32_set(0.1f * (float)(j + 1));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < F32_NACC; j++)
        acc[j] = f32_fma(acc[j], b, c);
    }
  f32v s = acc[0];
  for (int j = 1; j < F32_NACC; j++) s = f32_add(s, acc[j]);
  return (double)f32_hsum(s);
}

static double runFp64Chain(uint64_t outer)
{
  f64v acc[F64_NACC];
  const f64v b = f64_set(0.999999);
  const f64v c = f64_set(0.000001);
  for (int j = 0; j < F64_NACC; j++) acc[j] = f64_set(0.1 * (double)(j + 1));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < F64_NACC; j++)
        acc[j] = f64_fma(acc[j], b, c);
    }
  f64v s = acc[0];
  for (int j = 1; j < F64_NACC; j++) s = f64_add(s, acc[j]);
  return (double)f64_hsum(s);
}

int CpuPeak::runComputeSP(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"single_precision_compute", "Single-precision compute", "gflops"});
  double ops = (double)INNER * F32_NACC * F32_LANES * 2.0;
  emitCompute(*this, test, "float", ops, runFp32Chain, cfg);
  return 0;
}

int CpuPeak::runComputeDP(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"double_precision_compute", "Double-precision compute", "gflops"});
  double ops = (double)INNER * F64_NACC * F64_LANES * 2.0;
  emitCompute(*this, test, "double", ops, runFp64Chain, cfg);
  return 0;
}

// ---------------------------------------------------------------------------
// Half precision — native fp16 FMA where the ISA has it (ARM FEAT_FP16 vector
// arithmetic / x86 AVX512-FP16).  F16C (x86) is convert-only and does not
// count.
// ---------------------------------------------------------------------------
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define CPU_HAS_FP16_KERNEL 1
static constexpr int FP16_LANES = 8, FP16_NACC = 16;
static double runFp16Chain(uint64_t outer)
{
  float16x8_t acc[FP16_NACC];
  // Constants must stay distinct from 1.0 / 0.0 *after* fp16 rounding, or the
  // recurrence acc=acc*b+c degenerates to acc=acc and the loop is deleted.
  const float16x8_t b = vdupq_n_f16((float16_t)0.9995f);
  const float16x8_t c = vdupq_n_f16((float16_t)0.001f);
  for (int j = 0; j < FP16_NACC; j++) acc[j] = vdupq_n_f16((float16_t)(0.1f * (j + 1)));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < FP16_NACC; j++)
        acc[j] = vfmaq_f16(c, acc[j], b);
    }
  float16_t tmp[8]; vst1q_f16(tmp, acc[0]);
  return (double)tmp[0];
}
#elif defined(__AVX512FP16__)
#define CPU_HAS_FP16_KERNEL 1
static constexpr int FP16_LANES = 32, FP16_NACC = 16;
static double runFp16Chain(uint64_t outer)
{
  __m512h acc[FP16_NACC];
  // fp16-representable constants distinct from 1.0 / 0.0 (see ARM note above).
  const __m512h b = _mm512_set1_ph((_Float16)0.9995f);
  const __m512h c = _mm512_set1_ph((_Float16)0.001f);
  for (int j = 0; j < FP16_NACC; j++) acc[j] = _mm512_set1_ph((_Float16)(0.1f * (j + 1)));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < FP16_NACC; j++)
        acc[j] = _mm512_fmadd_ph(acc[j], b, c);
    }
  _Float16 tmp[32]; _mm512_storeu_ph(tmp, acc[0]);
  return (double)tmp[0];
}
#endif

int CpuPeak::runComputeHP(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"half_precision_compute", "Half-precision compute", "gflops"});
#ifdef CPU_HAS_FP16_KERNEL
  if (!info.hasFP16)
  {
    test.skip("half", ResultStatus::Unsupported, "no native fp16 arithmetic on this CPU");
    return 0;
  }
  double ops = (double)INNER * FP16_NACC * FP16_LANES * 2.0;
  emitCompute(*this, test, "half", ops, runFp16Chain, cfg);
#else
  test.skip("half", ResultStatus::Unsupported,
            "fp16 arithmetic not compiled for this target ISA");
#endif
  return 0;
}

// ---------------------------------------------------------------------------
// BF16 compute — the CPU bf16 throughput instruction is the dot-product
// (bf16xbf16 -> fp32 accumulate): ARM bfdot / x86 AVX512-BF16 dpbf16.
// ---------------------------------------------------------------------------
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC) || defined(__ARM_FEATURE_BF16)
#define CPU_HAS_BF16_KERNEL 1
// vbfdotq_f32: 4 fp32 lanes each += sum of 2 bf16 products = 16 flops / instr.
static constexpr int BF16_NACC = 16, BF16_FLOPS_PER_INSTR = 16;
static double runBf16Chain(uint64_t outer)
{
  float32x4_t acc[BF16_NACC];
  const bfloat16x4_t lo = vcvt_bf16_f32(vdupq_n_f32(1.0001f));
  const bfloat16x8_t a  = vcombine_bf16(lo, lo);
  const bfloat16x8_t b  = vcombine_bf16(lo, lo);
  for (int j = 0; j < BF16_NACC; j++) acc[j] = vdupq_n_f32(0.0f);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < BF16_NACC; j++)
        acc[j] = vbfdotq_f32(acc[j], a, b);
    }
  return (double)vaddvq_f32(acc[0]);
}
#elif defined(__AVX512BF16__)
#define CPU_HAS_BF16_KERNEL 1
// dpbf16_ps: 16 fp32 lanes each += 2 bf16 products = 64 flops / instr.
static constexpr int BF16_NACC = 16, BF16_FLOPS_PER_INSTR = 64;
static double runBf16Chain(uint64_t outer)
{
  __m512 acc[BF16_NACC];
  const __m512bh a = _mm512_cvtne2ps_pbh(_mm512_set1_ps(1.0001f), _mm512_set1_ps(1.0001f));
  const __m512bh b = a;
  for (int j = 0; j < BF16_NACC; j++) acc[j] = _mm512_setzero_ps();
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < BF16_NACC; j++)
        acc[j] = _mm512_dpbf16_ps(acc[j], a, b);
    }
  return (double)_mm512_reduce_add_ps(acc[0]);
}
#endif

int CpuPeak::runComputeBF16(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"bfloat16_compute", "BF16 compute bf16xbf16+fp32", "gflops"});
#ifdef CPU_HAS_BF16_KERNEL
  if (!info.hasBF16)
  {
    test.skip("bf16", ResultStatus::Unsupported, "no bf16 dot instruction on this CPU");
    return 0;
  }
  double ops = (double)INNER * BF16_NACC * BF16_FLOPS_PER_INSTR;
  emitCompute(*this, test, "bf16", ops, runBf16Chain, cfg);
#else
  test.skip("bf16", ResultStatus::Unsupported,
            "bf16 dot not compiled for this target ISA");
#endif
  return 0;
}

// ---------------------------------------------------------------------------
// Mixed precision: fp16 multiply -> fp32 accumulate (ARM FEAT_FP16 path).
// ---------------------------------------------------------------------------
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#define CPU_HAS_MP_KERNEL 1
static constexpr int MP_NACC = 16;
static double runMpChain(uint64_t outer)
{
  // Honest mixed kernel: each step narrows the fp32 accumulator to fp16,
  // multiplies in fp16 (so the multiply *depends* on acc and can't be hoisted),
  // then widens the product back and accumulates in fp32.  b<0 with |b| small
  // makes acc decay (no inf), while the fp16 multiply still executes every step.
  float32x4_t acc[MP_NACC];
  const float16x4_t b = vdup_n_f16((float16_t)(-0.0005f));
  for (int j = 0; j < MP_NACC; j++) acc[j] = vdupq_n_f32(1.0f + 0.01f * j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < MP_NACC; j++)
      {
        float16x4_t h = vcvt_f16_f32(acc[j]);          // 4 fp32->fp16 narrow
        float16x4_t p = vmul_f16(h, b);                // 4 fp16 multiplies
        acc[j] = vaddq_f32(acc[j], vcvt_f32_f16(p));   // 4 fp32 adds
      }
    }
  return (double)vaddvq_f32(acc[0]);
}
#endif

int CpuPeak::runComputeMP(benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"mixed_precision_compute", "Mixed-precision compute fp16xfp16+fp32", "gflops"});
#ifdef CPU_HAS_MP_KERNEL
  if (!info.hasFP16)
  {
    test.skip("mp", ResultStatus::Unsupported, "no native fp16 arithmetic on this CPU");
    return 0;
  }
  // 4 fp16 mults + 4 fp32 adds per accumulator update = 8 ops.
  double ops = (double)INNER * MP_NACC * 8.0;
  emitCompute(*this, test, "mp", ops, runMpChain, cfg);
#else
  test.skip("mp", ResultStatus::Unsupported,
            "mixed fp16/fp32 not compiled for this target ISA");
#endif
  return 0;
}

#endif // ENABLE_CPU
