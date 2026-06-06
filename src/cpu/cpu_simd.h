#ifndef CPU_SIMD_H
#define CPU_SIMD_H

// Backend-internal SIMD abstraction for the CPU compute kernels.  The backend
// is compiled with -march=native (see src/cpu/CMakeLists.txt), so the widest
// vector ISA the build host supports is selected here at compile time, with a
// scalar fallback that always works.  Each "Vec" exposes set1 / fma / hsum and
// a lane count + a per-ISA accumulator count (NACC) chosen to (a) hide the FMA
// latency*throughput product and (b) stay within the architectural register
// file so the independent chains don't spill.

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#elif defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

// Force full unrolling of the accumulator loop so the NACC independent chains
// live in distinct vector registers; otherwise they stay on the stack and the
// kernel becomes load/store bound at roughly half the FMA throughput.
#if defined(__clang__)
#define CPU_UNROLL_FULL _Pragma("clang loop unroll(full)")
// Unroll the hot iteration loop a few times so the loop-control branch isn't a
// scheduling bubble between back-to-back FMA groups (~13% on Firestorm).
#define CPU_UNROLL_K    _Pragma("clang loop unroll_count(4)")
#elif defined(__GNUC__)
#define CPU_UNROLL_FULL _Pragma("GCC unroll 32")
#define CPU_UNROLL_K    _Pragma("GCC unroll 4")
#else
#define CPU_UNROLL_FULL
#define CPU_UNROLL_K
#endif

namespace cpu_simd {

// Groups of NACC FMAs executed per outer iteration.  Large enough that one
// probe (iters=1) is a measurable slice; runWorkload's pickIters scales the
// timed batch to the target budget.
constexpr int INNER = 4096;

// ===========================================================================
// float32
// ===========================================================================
#if defined(__AVX512F__)
using f32v = __m512;
constexpr int F32_LANES = 16;
constexpr int F32_NACC  = 16;
static inline f32v f32_set(float a)            { return _mm512_set1_ps(a); }
static inline f32v f32_load(const float *p)    { return _mm512_loadu_ps(p); }
static inline f32v f32_fma(f32v a, f32v b, f32v c) { return _mm512_fmadd_ps(a, b, c); }
static inline f32v f32_add(f32v a, f32v b)     { return _mm512_add_ps(a, b); }
static inline float f32_hsum(f32v a)           { return _mm512_reduce_add_ps(a); }
#elif defined(__AVX2__) && defined(__FMA__)
using f32v = __m256;
constexpr int F32_LANES = 8;
constexpr int F32_NACC  = 12;
static inline f32v f32_set(float a)            { return _mm256_set1_ps(a); }
static inline f32v f32_load(const float *p)    { return _mm256_loadu_ps(p); }
static inline f32v f32_fma(f32v a, f32v b, f32v c) { return _mm256_fmadd_ps(a, b, c); }
static inline f32v f32_add(f32v a, f32v b)     { return _mm256_add_ps(a, b); }
static inline float f32_hsum(f32v a)
{
  __m128 lo = _mm256_castps256_ps128(a);
  __m128 hi = _mm256_extractf128_ps(a, 1);
  lo = _mm_add_ps(lo, hi);
  lo = _mm_hadd_ps(lo, lo);
  lo = _mm_hadd_ps(lo, lo);
  return _mm_cvtss_f32(lo);
}
#elif defined(__ARM_NEON) || defined(__aarch64__)
using f32v = float32x4_t;
constexpr int F32_LANES = 4;
constexpr int F32_NACC  = 16;
static inline f32v f32_set(float a)            { return vdupq_n_f32(a); }
static inline f32v f32_load(const float *p)    { return vld1q_f32(p); }
static inline f32v f32_fma(f32v a, f32v b, f32v c) { return vfmaq_f32(c, a, b); }
static inline f32v f32_add(f32v a, f32v b)     { return vaddq_f32(a, b); }
static inline float f32_hsum(f32v a)           { return vaddvq_f32(a); }
#else
using f32v = float;
constexpr int F32_LANES = 1;
constexpr int F32_NACC  = 8;
static inline f32v f32_set(float a)            { return a; }
static inline f32v f32_load(const float *p)    { return *p; }
static inline f32v f32_fma(f32v a, f32v b, f32v c) { return a * b + c; }
static inline f32v f32_add(f32v a, f32v b)     { return a + b; }
static inline float f32_hsum(f32v a)           { return a; }
#endif

// ===========================================================================
// float64
// ===========================================================================
#if defined(__AVX512F__)
using f64v = __m512d;
constexpr int F64_LANES = 8;
constexpr int F64_NACC  = 16;
static inline f64v f64_set(double a)           { return _mm512_set1_pd(a); }
static inline f64v f64_fma(f64v a, f64v b, f64v c) { return _mm512_fmadd_pd(a, b, c); }
static inline f64v f64_add(f64v a, f64v b)     { return _mm512_add_pd(a, b); }
static inline double f64_hsum(f64v a)          { return _mm512_reduce_add_pd(a); }
#elif defined(__AVX2__) && defined(__FMA__)
using f64v = __m256d;
constexpr int F64_LANES = 4;
constexpr int F64_NACC  = 12;
static inline f64v f64_set(double a)           { return _mm256_set1_pd(a); }
static inline f64v f64_fma(f64v a, f64v b, f64v c) { return _mm256_fmadd_pd(a, b, c); }
static inline f64v f64_add(f64v a, f64v b)     { return _mm256_add_pd(a, b); }
static inline double f64_hsum(f64v a)
{
  __m128d lo = _mm256_castpd256_pd128(a);
  __m128d hi = _mm256_extractf128_pd(a, 1);
  lo = _mm_add_pd(lo, hi);
  return _mm_cvtsd_f64(_mm_hadd_pd(lo, lo));
}
#elif defined(__aarch64__)
using f64v = float64x2_t;
constexpr int F64_LANES = 2;
constexpr int F64_NACC  = 16;
static inline f64v f64_set(double a)           { return vdupq_n_f64(a); }
static inline f64v f64_fma(f64v a, f64v b, f64v c) { return vfmaq_f64(c, a, b); }
static inline f64v f64_add(f64v a, f64v b)     { return vaddq_f64(a, b); }
static inline double f64_hsum(f64v a)          { return vaddvq_f64(a); }
#else
using f64v = double;
constexpr int F64_LANES = 1;
constexpr int F64_NACC  = 8;
static inline f64v f64_set(double a)           { return a; }
static inline f64v f64_fma(f64v a, f64v b, f64v c) { return a * b + c; }
static inline f64v f64_add(f64v a, f64v b)     { return a + b; }
static inline double f64_hsum(f64v a)          { return a; }
#endif

// ===========================================================================
// int32 (multiply-add; no integer FMA on x86 so it is mul + add = 2 ops)
// ===========================================================================
#if defined(__AVX512F__)
using i32v = __m512i;
constexpr int I32_LANES = 16;
constexpr int I32_NACC  = 16;
static inline i32v i32_set(int a)              { return _mm512_set1_epi32(a); }
static inline i32v i32_madd(i32v a, i32v b, i32v c) { return _mm512_add_epi32(_mm512_mullo_epi32(a, b), c); }
static inline i32v i32_add(i32v a, i32v b)     { return _mm512_add_epi32(a, b); }
static inline int   i32_hsum(i32v a)           { return _mm512_reduce_add_epi32(a); }
#elif defined(__AVX2__)
using i32v = __m256i;
constexpr int I32_LANES = 8;
constexpr int I32_NACC  = 12;
static inline i32v i32_set(int a)              { return _mm256_set1_epi32(a); }
static inline i32v i32_madd(i32v a, i32v b, i32v c) { return _mm256_add_epi32(_mm256_mullo_epi32(a, b), c); }
static inline i32v i32_add(i32v a, i32v b)     { return _mm256_add_epi32(a, b); }
static inline int   i32_hsum(i32v a)
{
  __m128i lo = _mm256_castsi256_si128(a);
  __m128i hi = _mm256_extracti128_si256(a, 1);
  lo = _mm_add_epi32(lo, hi);
  lo = _mm_hadd_epi32(lo, lo);
  lo = _mm_hadd_epi32(lo, lo);
  return _mm_cvtsi128_si32(lo);
}
#elif defined(__ARM_NEON) || defined(__aarch64__)
using i32v = int32x4_t;
constexpr int I32_LANES = 4;
constexpr int I32_NACC  = 16;
static inline i32v i32_set(int a)              { return vdupq_n_s32(a); }
static inline i32v i32_madd(i32v a, i32v b, i32v c) { return vmlaq_s32(c, a, b); }
static inline i32v i32_add(i32v a, i32v b)     { return vaddq_s32(a, b); }
static inline int   i32_hsum(i32v a)           { return vaddvq_s32(a); }
#else
using i32v = int;
constexpr int I32_LANES = 1;
constexpr int I32_NACC  = 8;
static inline i32v i32_set(int a)              { return a; }
// Use unsigned arithmetic so wrap is well-defined (signed overflow is UB).
static inline i32v i32_madd(i32v a, i32v b, i32v c)
{ return (int)((unsigned)a * (unsigned)b + (unsigned)c); }
static inline i32v i32_add(i32v a, i32v b)     { return (int)((unsigned)a + (unsigned)b); }
static inline int   i32_hsum(i32v a)           { return a; }
#endif

} // namespace cpu_simd

#endif // CPU_SIMD_H
