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
#elif defined(__aarch64__) || defined(_M_ARM64)
// Only AArch64 uses the NEON kernels; 32-bit ARMv7 lacks fused-FMA / horizontal
// reduce intrinsics, so it uses the scalar fallback and needs no arm_neon.h.
// MSVC ARM64 defines _M_ARM64 (never __aarch64__) and ships the full AArch64
// ACLE in its arm_neon.h.
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
// 32 ZMM registers give headroom to hide FMA latency (same rationale as NEON);
// AVX2 below stays at 12 (only 16 YMM regs, and fp64 confirms 12 suffices).
constexpr int F32_NACC  = 24;
static inline f32v f32_set(float a)            { return _mm512_set1_ps(a); }
static inline f32v f32_load(const float *p)    { return _mm512_loadu_ps(p); }
static inline f32v f32_fma(f32v a, f32v b, f32v c) { return _mm512_fmadd_ps(a, b, c); }
static inline f32v f32_add(f32v a, f32v b)     { return _mm512_add_ps(a, b); }
static inline float f32_hsum(f32v a)           { return _mm512_reduce_add_ps(a); }
#elif defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
// MSVC enables the FMA intrinsics under /arch:AVX2 but never defines __FMA__
// (and every AVX2 CPU has FMA3), so accept AVX2 alone on MSVC.
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
#elif defined(__SSE2__) || defined(_M_X64)
// MSVC never defines __SSE2__, but x86-64 always has SSE2 (this is the MSVC
// baseline/sse42 TU, which has no /arch:AVX2 so __AVX2__ is undefined above).
using f32v = __m128;
constexpr int F32_LANES = 4;
constexpr int F32_NACC  = 12;   // 16 XMM registers on x86-64
static inline f32v f32_set(float a)            { return _mm_set1_ps(a); }
static inline f32v f32_load(const float *p)    { return _mm_loadu_ps(p); }
static inline f32v f32_fma(f32v a, f32v b, f32v c) { return _mm_add_ps(_mm_mul_ps(a, b), c); } // no FMA on SSE
static inline f32v f32_add(f32v a, f32v b)     { return _mm_add_ps(a, b); }
static inline float f32_hsum(f32v a)
{
  __m128 t = _mm_add_ps(a, _mm_movehl_ps(a, a));
  t = _mm_add_ss(t, _mm_shuffle_ps(t, t, 0x55));
  return _mm_cvtss_f32(t);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
// AArch64 only: 32-bit ARMv7 NEON has no fused FMA (vfmaq_f32) and no
// horizontal reduce (vaddvq_f32), so armeabi-v7a falls through to scalar below.
using f32v = float32x4_t;
constexpr int F32_LANES = 4;
// Firestorm fp32 FMLA latency is ~6 cycles across 4 FP pipes, so ~24 in-flight
// accumulators are needed to saturate (NACC=16 left fp32 at ~62% of peak).
// NB: this assumes a compiler that schedules 24 independent chains well (clang
// does; GCC<=14 serialises them -- the root CMake prefers clang on Linux).
constexpr int F32_NACC  = 24;
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
constexpr int F64_NACC  = 24;   // 32 ZMM regs; hide FMA latency (see fp32)
static inline f64v f64_set(double a)           { return _mm512_set1_pd(a); }
static inline f64v f64_fma(f64v a, f64v b, f64v c) { return _mm512_fmadd_pd(a, b, c); }
static inline f64v f64_add(f64v a, f64v b)     { return _mm512_add_pd(a, b); }
static inline double f64_hsum(f64v a)          { return _mm512_reduce_add_pd(a); }
#elif defined(__AVX2__) && (defined(__FMA__) || defined(_MSC_VER))
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
#elif defined(__SSE2__) || defined(_M_X64)
using f64v = __m128d;
constexpr int F64_LANES = 2;
constexpr int F64_NACC  = 12;
static inline f64v f64_set(double a)           { return _mm_set1_pd(a); }
static inline f64v f64_fma(f64v a, f64v b, f64v c) { return _mm_add_pd(_mm_mul_pd(a, b), c); }
static inline f64v f64_add(f64v a, f64v b)     { return _mm_add_pd(a, b); }
static inline double f64_hsum(f64v a)          { return _mm_cvtsd_f64(_mm_add_sd(a, _mm_unpackhi_pd(a, a))); }
#elif defined(__aarch64__) || defined(_M_ARM64)
using f64v = float64x2_t;
constexpr int F64_LANES = 2;
constexpr int F64_NACC  = 24;   // same latency-hiding rationale as fp32
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
constexpr int I32_NACC  = 24;   // vpmulld has high latency; extra accumulators help
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
#elif defined(__SSE4_1__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86)))
// MSVC never defines __SSE4_1__ but exposes _mm_mullo_epi32 unconditionally on
// x86; the MSVC sse42 TU runs only when CPUID reports SSE4.2 (⊇ SSE4.1) at
// runtime.  Gate on _M_X64/_M_IX86, not bare _MSC_VER: MSVC ARM64 must fall
// through to the NEON branch below.
using i32v = __m128i;
constexpr int I32_LANES = 4;
constexpr int I32_NACC  = 12;
static inline i32v i32_set(int a)              { return _mm_set1_epi32(a); }
static inline i32v i32_madd(i32v a, i32v b, i32v c) { return _mm_add_epi32(_mm_mullo_epi32(a, b), c); } // mullo = SSE4.1
static inline i32v i32_add(i32v a, i32v b)     { return _mm_add_epi32(a, b); }
static inline int   i32_hsum(i32v a)
{
  __m128i t = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4E));
  t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1));
  return _mm_cvtsi128_si32(t);
}
#elif defined(__aarch64__) || defined(_M_ARM64)
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
