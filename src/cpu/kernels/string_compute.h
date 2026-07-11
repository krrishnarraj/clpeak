#ifndef CPU_KERN_STRING_COMPUTE_H
#define CPU_KERN_STRING_COMPUTE_H

#ifdef ENABLE_CPU

// ===========================================================================
// String-processing throughput kernels (Category::String), reported in GB/s:
//
//   strscan  memchr-style byte search: compare a whole buffer against a
//            needle byte, 4 vectors per step with an OR tree and ONE
//            found-test per step (the glibc/aarch64-optimized memchr shape).
//            The needle is absent, so the found-branch is never taken and the
//            number is the peak scan rate of the compare+test machinery.
//            Variants: SSE2 PCMPEQB+PMOVMSKB, AVX2, AVX-512BW VPCMPB+KOR,
//            NEON CMEQ+UMAXV, and the historical SSE4.2 PCMPISTRI string
//            instruction as its own row (famously slower than plain
//            compare+movemask on modern cores -- that IS the result).
//            An SVE variant lives in sve_compute.h (same buffer helpers).
//
//   utf8     UTF-8 validation via the lookup-shuffle algorithm (Keiser &
//            Lemire, "Validating UTF-8 In Less Than One Instruction Per
//            Byte" -- the simdjson/simdutf stage): three 16-entry nibble
//            tables classify (prev1-high, prev1-low, curr-high); a nonzero
//            AND of the three classes marks a structural error, plus a
//            saturating-subtract pass checks 3/4-byte leads are followed by
//            the right number of continuations.  No ASCII fast path: every
//            block runs the full classifier, so the number is the
//            data-independent classification rate on mixed text.
//            Variants: SSSE3/AVX2/AVX-512BW PSHUFB, NEON TBL.
//
// Both kernels read a 16 KB thread-local buffer -- L1-resident on every
// current core -- so they measure the instruction machinery, not memory
// bandwidth (a DRAM-sized input would just re-measure the bandwidth tests).
// Each pool worker touches its own copy; the first (warmup) call pays the
// init.  opsPerIter counts BYTES per outer iteration so the shared
// emitCompute() math lands in GB/s.
//
// Chain rules differ from the compute chains: the work here is a pure
// function of a read-only buffer, so a loop-carried accumulator alone would
// not stop LICM from hoisting an entire pass out of the outer loop.  A
// compiler memory barrier at the top of every pass makes the buffer contents
// opaque per pass, forcing every scan to be re-executed.  Guarded by
// CLPEAK_CORE_ONLY like crypto_compute.h (cl.exe TUs are core-only; every
// other toolchain supports the GNU inline-asm barrier).
// ===========================================================================

#include "cpu_kernels.h"
#include "cpu_simd.h"

#include <cstdint>

#ifndef CLPEAK_CORE_ONLY

namespace clpeak_cpu {
namespace {

// Compiler-only fence: tells the optimizer memory may have changed, so buffer
// loads can't be reused across passes (google-benchmark ClobberMemory).  Also
// used by the SVE strscan kernel in sve_compute.h.
#define CPU_STR_BARRIER() asm volatile("" ::: "memory")

// ---- Thread-local input buffers ---------------------------------------------
static constexpr int SCAN_BYTES = 16384;   // L1-resident on every current core
// Haystack bytes are 1..254: excludes 0x00 (the cmpeq-kernel needle, and the
// PCMPISTRI implicit string terminator) and 0xFF (the PCMPISTRI-kernel needle
// -- that instruction can't search for 0, it means end-of-string).
static constexpr uint8_t SCAN_NEEDLE = 0x00, SCAN_NEEDLE_ISTRI = 0xFF;

struct ScanBuf { alignas(64) uint8_t b[SCAN_BYTES]; bool init = false; };
[[maybe_unused]] static inline const uint8_t *scanBuf()
{
  static thread_local ScanBuf s;
  if (!s.init)
  {
    uint32_t x = 0x9E3779B9u;   // xorshift32; runtime data, nothing to const-fold
    for (int i = 0; i < SCAN_BYTES; i++)
    {
      x ^= x << 13; x ^= x >> 17; x ^= x << 5;
      s.b[i] = (uint8_t)((x % 254u) + 1u);
    }
    s.init = true;
  }
  return s.b;
}

static constexpr int UTF8_BYTES = 16384;
struct Utf8Buf { alignas(64) uint8_t b[UTF8_BYTES]; bool init = false; };
[[maybe_unused]] static inline const uint8_t *utf8Buf()
{
  static thread_local Utf8Buf u;
  if (!u.init)
  {
    // Valid mixed text, web-like ratio: repeating unit of 12 pseudo-random
    // ASCII bytes + one 2-byte (U+00E9), one 3-byte (U+20AC) and one 4-byte
    // (U+1F600) sequence (21 bytes, ~57% ASCII).  Content doesn't affect
    // speed (no data-dependent branches in the kernels), but a valid buffer
    // lets the sink double as a self-check: the reduced error must be 0.
    uint32_t x = 0x243F6A88u;
    int i = 0;
    while (i < UTF8_BYTES)
    {
      for (int k = 0; k < 12 && i < UTF8_BYTES; k++)
      {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        u.b[i++] = (uint8_t)(0x20 + (x % 95u));
      }
      if (i + 9 > UTF8_BYTES)   // never end mid-sequence: pad the tail ASCII
      {
        while (i < UTF8_BYTES) u.b[i++] = 'a';
        break;
      }
      u.b[i++] = 0xC3; u.b[i++] = 0xA9;                                  // U+00E9
      u.b[i++] = 0xE2; u.b[i++] = 0x82; u.b[i++] = 0xAC;                 // U+20AC
      u.b[i++] = 0xF0; u.b[i++] = 0x9F; u.b[i++] = 0x98; u.b[i++] = 0x80; // U+1F600
    }
    u.init = true;
  }
  return u.b;
}

// ---- UTF-8 nibble classification tables (Keiser-Lemire lookup-shuffle) ------
// Bit names follow simdjson's utf8_lookup4: a byte-pair error class is flagged
// when the same bit survives the AND of all three table lookups.  TWO_CONTS
// (0x80) marks continuation bytes; the must-be-2/3-continuation check XORs it
// away when a continuation is correctly forced by an E0..FF lead 2/3 back.
[[maybe_unused]] static constexpr uint8_t
  U8_TOO_SHORT = 1u << 0, U8_TOO_LONG = 1u << 1, U8_OVERLONG_3 = 1u << 2,
  U8_TOO_LARGE = 1u << 3, U8_SURROGATE = 1u << 4, U8_OVERLONG_2 = 1u << 5,
  U8_TOO_LARGE_1000 = 1u << 6, U8_OVERLONG_4 = 1u << 6, U8_TWO_CONTS = 1u << 7,
  U8_CARRY = U8_TOO_SHORT | U8_TOO_LONG | U8_TWO_CONTS;

alignas(16) [[maybe_unused]] static constexpr uint8_t U8_TBL1[16] = {
  // index = high nibble of the PREVIOUS byte (prev1)
  U8_TOO_LONG, U8_TOO_LONG, U8_TOO_LONG, U8_TOO_LONG,          // 0..3: ASCII
  U8_TOO_LONG, U8_TOO_LONG, U8_TOO_LONG, U8_TOO_LONG,          // 4..7: ASCII
  U8_TWO_CONTS, U8_TWO_CONTS, U8_TWO_CONTS, U8_TWO_CONTS,      // 8..B: continuation
  U8_TOO_SHORT | U8_OVERLONG_2,                                // C: 2-byte lead
  U8_TOO_SHORT,                                                // D: 2-byte lead
  U8_TOO_SHORT | U8_OVERLONG_3 | U8_SURROGATE,                 // E: 3-byte lead
  U8_TOO_SHORT | U8_TOO_LARGE | U8_TOO_LARGE_1000 | U8_OVERLONG_4, // F: 4-byte lead
};
alignas(16) [[maybe_unused]] static constexpr uint8_t U8_TBL2[16] = {
  // index = low nibble of the PREVIOUS byte (prev1)
  U8_CARRY | U8_OVERLONG_3 | U8_OVERLONG_2 | U8_OVERLONG_4,    // 0
  U8_CARRY | U8_OVERLONG_2,                                    // 1
  U8_CARRY, U8_CARRY,                                          // 2, 3
  U8_CARRY | U8_TOO_LARGE,                                     // 4
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000,                 // 5
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000,                 // 6
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000,                 // 7
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000,                 // 8
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000,                 // 9
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000,                 // A
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000,                 // B
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000,                 // C
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000 | U8_SURROGATE,  // D (ED = surrogate)
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000,                 // E
  U8_CARRY | U8_TOO_LARGE | U8_TOO_LARGE_1000,                 // F
};
alignas(16) [[maybe_unused]] static constexpr uint8_t U8_TBL3[16] = {
  // index = high nibble of the CURRENT byte
  U8_TOO_SHORT, U8_TOO_SHORT, U8_TOO_SHORT, U8_TOO_SHORT,      // 0..3: ASCII
  U8_TOO_SHORT, U8_TOO_SHORT, U8_TOO_SHORT, U8_TOO_SHORT,      // 4..7: ASCII
  U8_TOO_LONG | U8_OVERLONG_2 | U8_TWO_CONTS | U8_OVERLONG_3
      | U8_TOO_LARGE_1000 | U8_OVERLONG_4,                     // 8: cont 80..8F
  U8_TOO_LONG | U8_OVERLONG_2 | U8_TWO_CONTS | U8_OVERLONG_3
      | U8_TOO_LARGE,                                          // 9: cont 90..9F
  U8_TOO_LONG | U8_OVERLONG_2 | U8_TWO_CONTS | U8_SURROGATE
      | U8_TOO_LARGE,                                          // A: cont A0..AF
  U8_TOO_LONG | U8_OVERLONG_2 | U8_TWO_CONTS | U8_SURROGATE
      | U8_TOO_LARGE,                                          // B: cont B0..BF
  U8_TOO_SHORT, U8_TOO_SHORT, U8_TOO_SHORT, U8_TOO_SHORT,      // C..F: lead
};

// =============================================================================
// strscan -- memchr-style byte search, widest ISA this TU was compiled for.
// The found-arm is cold (the needle is absent); its coarse position math is
// fine, it exists so the compare results are consumed by a real test+branch.
// =============================================================================
#if defined(__AVX512BW__)

#define CPU_HAS_STRSCAN_KERNEL 1
static double runStrScanChain(uint64_t outer)
{
  const uint8_t *buf = scanBuf();
  volatile uint8_t vneedle = SCAN_NEEDLE;
  const __m512i needle = _mm512_set1_epi8((char)vneedle);
  uint64_t found = 0;
  for (uint64_t o = 0; o < outer; o++)
  {
    CPU_STR_BARRIER();
    CPU_UNROLL_K
    for (int i = 0; i < SCAN_BYTES; i += 256)
    {
      __m512i v0 = _mm512_load_si512((const void *)(buf + i));
      __m512i v1 = _mm512_load_si512((const void *)(buf + i + 64));
      __m512i v2 = _mm512_load_si512((const void *)(buf + i + 128));
      __m512i v3 = _mm512_load_si512((const void *)(buf + i + 192));
      __mmask64 k0 = _mm512_cmpeq_epi8_mask(v0, needle);
      __mmask64 k1 = _mm512_cmpeq_epi8_mask(v1, needle);
      __mmask64 k2 = _mm512_cmpeq_epi8_mask(v2, needle);
      __mmask64 k3 = _mm512_cmpeq_epi8_mask(v3, needle);
      uint64_t m = (uint64_t)(k0 | k1 | k2 | k3);
      if (m) found += (uint64_t)i + (unsigned)__builtin_ctzll(m);  // never taken
    }
  }
  return (double)found;
}

#elif defined(__AVX2__)

#define CPU_HAS_STRSCAN_KERNEL 1
static double runStrScanChain(uint64_t outer)
{
  const uint8_t *buf = scanBuf();
  volatile uint8_t vneedle = SCAN_NEEDLE;
  const __m256i needle = _mm256_set1_epi8((char)vneedle);
  uint64_t found = 0;
  for (uint64_t o = 0; o < outer; o++)
  {
    CPU_STR_BARRIER();
    CPU_UNROLL_K
    for (int i = 0; i < SCAN_BYTES; i += 128)
    {
      __m256i v0 = _mm256_load_si256((const __m256i *)(buf + i));
      __m256i v1 = _mm256_load_si256((const __m256i *)(buf + i + 32));
      __m256i v2 = _mm256_load_si256((const __m256i *)(buf + i + 64));
      __m256i v3 = _mm256_load_si256((const __m256i *)(buf + i + 96));
      __m256i c01 = _mm256_or_si256(_mm256_cmpeq_epi8(v0, needle),
                                    _mm256_cmpeq_epi8(v1, needle));
      __m256i c23 = _mm256_or_si256(_mm256_cmpeq_epi8(v2, needle),
                                    _mm256_cmpeq_epi8(v3, needle));
      unsigned m = (unsigned)_mm256_movemask_epi8(_mm256_or_si256(c01, c23));
      if (m) found += (uint64_t)i + (unsigned)__builtin_ctz(m);    // never taken
    }
  }
  return (double)found;
}

#elif defined(__SSE2__)

#define CPU_HAS_STRSCAN_KERNEL 1
static double runStrScanChain(uint64_t outer)
{
  const uint8_t *buf = scanBuf();
  volatile uint8_t vneedle = SCAN_NEEDLE;
  const __m128i needle = _mm_set1_epi8((char)vneedle);
  uint64_t found = 0;
  for (uint64_t o = 0; o < outer; o++)
  {
    CPU_STR_BARRIER();
    CPU_UNROLL_K
    for (int i = 0; i < SCAN_BYTES; i += 64)
    {
      __m128i v0 = _mm_load_si128((const __m128i *)(buf + i));
      __m128i v1 = _mm_load_si128((const __m128i *)(buf + i + 16));
      __m128i v2 = _mm_load_si128((const __m128i *)(buf + i + 32));
      __m128i v3 = _mm_load_si128((const __m128i *)(buf + i + 48));
      __m128i c01 = _mm_or_si128(_mm_cmpeq_epi8(v0, needle),
                                 _mm_cmpeq_epi8(v1, needle));
      __m128i c23 = _mm_or_si128(_mm_cmpeq_epi8(v2, needle),
                                 _mm_cmpeq_epi8(v3, needle));
      unsigned m = (unsigned)_mm_movemask_epi8(_mm_or_si128(c01, c23));
      if (m) found += (uint64_t)i + (unsigned)__builtin_ctz(m);    // never taken
    }
  }
  return (double)found;
}

#elif defined(__aarch64__) || defined(_M_ARM64)

#define CPU_HAS_STRSCAN_KERNEL 1
static double runStrScanChain(uint64_t outer)
{
  const uint8_t *buf = scanBuf();
  volatile uint8_t vneedle = SCAN_NEEDLE;
  const uint8x16_t needle = vdupq_n_u8(vneedle);
  uint64_t found = 0;
  for (uint64_t o = 0; o < outer; o++)
  {
    CPU_STR_BARRIER();
    CPU_UNROLL_K
    for (int i = 0; i < SCAN_BYTES; i += 64)
    {
      uint8x16_t v0 = vld1q_u8(buf + i);
      uint8x16_t v1 = vld1q_u8(buf + i + 16);
      uint8x16_t v2 = vld1q_u8(buf + i + 32);
      uint8x16_t v3 = vld1q_u8(buf + i + 48);
      uint8x16_t c01 = vorrq_u8(vceqq_u8(v0, needle), vceqq_u8(v1, needle));
      uint8x16_t c23 = vorrq_u8(vceqq_u8(v2, needle), vceqq_u8(v3, needle));
      if (vmaxvq_u8(vorrq_u8(c01, c23)))                           // never taken
        found += (uint64_t)i;
    }
  }
  return (double)found;
}

#endif // strscan ISA chain

// ---- SSE4.2 PCMPISTRI string-instruction scan (its own menu row) ------------
// One PCMPISTRI per 16-byte chunk (implicit-length EQUAL_ANY search for one
// character).  Kept as a separate row because it is the literal "string
// instruction" -- and measurably slower than compare+movemask on modern cores.
#if defined(__SSE4_2__) && (defined(__x86_64__) || defined(_M_X64))

#define CPU_HAS_STRSCAN_ISTRI_KERNEL 1
static double runStrScanIstriChain(uint64_t outer)
{
  const uint8_t *buf = scanBuf();
  volatile uint8_t vneedle = SCAN_NEEDLE_ISTRI;
  const __m128i set = _mm_insert_epi8(_mm_setzero_si128(), (char)vneedle, 0);
  uint64_t found = 0;
  for (uint64_t o = 0; o < outer; o++)
  {
    CPU_STR_BARRIER();
    CPU_UNROLL_K
    for (int i = 0; i < SCAN_BYTES; i += 16)
    {
      __m128i chunk = _mm_load_si128((const __m128i *)(buf + i));
      int idx = _mm_cmpistri(set, chunk,
                             _SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
                             _SIDD_POSITIVE_POLARITY | _SIDD_LEAST_SIGNIFICANT);
      if (idx != 16) found += (uint64_t)i + (unsigned)idx;         // never taken
    }
  }
  return (double)found;
}

#endif // PCMPISTRI

// =============================================================================
// utf8 -- lookup-shuffle validation, widest ISA this TU was compiled for.
// Per input vector: prev1/2/3 are the input shifted by 1..3 bytes (carrying
// bytes in from the previous vector); the three nibble-table lookups AND to
// the special-case classes, and the saturating-subtract pass turns "a byte
// >= 0xE0 (resp. 0xF0) sits 2 (resp. 3) back" into the 0x80 must-be-
// continuation bit that XORs against TWO_CONTS.  Errors OR into `err`.
// =============================================================================
#if defined(__AVX512BW__)

#define CPU_HAS_UTF8_KERNEL 1
static double runUtf8Chain(uint64_t outer)
{
  const uint8_t *buf = utf8Buf();
  const __m512i t1 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)U8_TBL1));
  const __m512i t2 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)U8_TBL2));
  const __m512i t3 = _mm512_broadcast_i32x4(_mm_load_si128((const __m128i *)U8_TBL3));
  const __m512i low4 = _mm512_set1_epi8(0x0F);
  __m512i err = _mm512_setzero_si512();
  __m512i prev = _mm512_setzero_si512();
  for (uint64_t o = 0; o < outer; o++)
  {
    CPU_STR_BARRIER();
    for (int i = 0; i < UTF8_BYTES; i += 64)
    {
      __m512i in = _mm512_load_si512((const void *)(buf + i));
      // Per-128-lane shift with the previous lane (lane0 <- prev's top lane).
      __m512i lsh = _mm512_alignr_epi64(in, prev, 6);
      __m512i p1 = _mm512_alignr_epi8(in, lsh, 15);
      __m512i p2 = _mm512_alignr_epi8(in, lsh, 14);
      __m512i p3 = _mm512_alignr_epi8(in, lsh, 13);
      __m512i sc = _mm512_and_si512(
          _mm512_and_si512(
              _mm512_shuffle_epi8(t1, _mm512_and_si512(_mm512_srli_epi16(p1, 4), low4)),
              _mm512_shuffle_epi8(t2, _mm512_and_si512(p1, low4))),
          _mm512_shuffle_epi8(t3, _mm512_and_si512(_mm512_srli_epi16(in, 4), low4)));
      __m512i m23 = _mm512_or_si512(_mm512_subs_epu8(p2, _mm512_set1_epi8(0x60)),
                                    _mm512_subs_epu8(p3, _mm512_set1_epi8(0x70)));
      m23 = _mm512_and_si512(m23, _mm512_set1_epi8((char)0x80));
      err = _mm512_or_si512(err, _mm512_xor_si512(sc, m23));
      prev = in;
    }
  }
  return (double)_mm512_reduce_add_epi64(err);   // 0 on our valid input
}

#elif defined(__AVX2__)

#define CPU_HAS_UTF8_KERNEL 1
static double runUtf8Chain(uint64_t outer)
{
  const uint8_t *buf = utf8Buf();
  const __m256i t1 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)U8_TBL1));
  const __m256i t2 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)U8_TBL2));
  const __m256i t3 = _mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)U8_TBL3));
  const __m256i low4 = _mm256_set1_epi8(0x0F);
  __m256i err = _mm256_setzero_si256();
  __m256i prev = _mm256_setzero_si256();
  for (uint64_t o = 0; o < outer; o++)
  {
    CPU_STR_BARRIER();
    for (int i = 0; i < UTF8_BYTES; i += 32)
    {
      __m256i in = _mm256_load_si256((const __m256i *)(buf + i));
      __m256i lsh = _mm256_permute2x128_si256(prev, in, 0x21);  // [prev.hi, in.lo]
      __m256i p1 = _mm256_alignr_epi8(in, lsh, 15);
      __m256i p2 = _mm256_alignr_epi8(in, lsh, 14);
      __m256i p3 = _mm256_alignr_epi8(in, lsh, 13);
      __m256i sc = _mm256_and_si256(
          _mm256_and_si256(
              _mm256_shuffle_epi8(t1, _mm256_and_si256(_mm256_srli_epi16(p1, 4), low4)),
              _mm256_shuffle_epi8(t2, _mm256_and_si256(p1, low4))),
          _mm256_shuffle_epi8(t3, _mm256_and_si256(_mm256_srli_epi16(in, 4), low4)));
      __m256i m23 = _mm256_or_si256(_mm256_subs_epu8(p2, _mm256_set1_epi8(0x60)),
                                    _mm256_subs_epu8(p3, _mm256_set1_epi8(0x70)));
      m23 = _mm256_and_si256(m23, _mm256_set1_epi8((char)0x80));
      err = _mm256_or_si256(err, _mm256_xor_si256(sc, m23));
      prev = in;
    }
  }
  alignas(32) uint8_t tmp[32];
  _mm256_store_si256((__m256i *)tmp, err);
  unsigned s = 0;
  for (int j = 0; j < 32; j++) s += tmp[j];
  return (double)s;   // 0 on our valid input
}

#elif defined(__SSSE3__)

#define CPU_HAS_UTF8_KERNEL 1
static double runUtf8Chain(uint64_t outer)
{
  const uint8_t *buf = utf8Buf();
  const __m128i t1 = _mm_load_si128((const __m128i *)U8_TBL1);
  const __m128i t2 = _mm_load_si128((const __m128i *)U8_TBL2);
  const __m128i t3 = _mm_load_si128((const __m128i *)U8_TBL3);
  const __m128i low4 = _mm_set1_epi8(0x0F);
  __m128i err = _mm_setzero_si128();
  __m128i prev = _mm_setzero_si128();
  for (uint64_t o = 0; o < outer; o++)
  {
    CPU_STR_BARRIER();
    for (int i = 0; i < UTF8_BYTES; i += 16)
    {
      __m128i in = _mm_load_si128((const __m128i *)(buf + i));
      __m128i p1 = _mm_alignr_epi8(in, prev, 15);
      __m128i p2 = _mm_alignr_epi8(in, prev, 14);
      __m128i p3 = _mm_alignr_epi8(in, prev, 13);
      __m128i sc = _mm_and_si128(
          _mm_and_si128(
              _mm_shuffle_epi8(t1, _mm_and_si128(_mm_srli_epi16(p1, 4), low4)),
              _mm_shuffle_epi8(t2, _mm_and_si128(p1, low4))),
          _mm_shuffle_epi8(t3, _mm_and_si128(_mm_srli_epi16(in, 4), low4)));
      __m128i m23 = _mm_or_si128(_mm_subs_epu8(p2, _mm_set1_epi8(0x60)),
                                 _mm_subs_epu8(p3, _mm_set1_epi8(0x70)));
      m23 = _mm_and_si128(m23, _mm_set1_epi8((char)0x80));
      err = _mm_or_si128(err, _mm_xor_si128(sc, m23));
      prev = in;
    }
  }
  alignas(16) uint8_t tmp[16];
  _mm_store_si128((__m128i *)tmp, err);
  unsigned s = 0;
  for (int j = 0; j < 16; j++) s += tmp[j];
  return (double)s;   // 0 on our valid input
}

#elif defined(__aarch64__) || defined(_M_ARM64)

#define CPU_HAS_UTF8_KERNEL 1
static double runUtf8Chain(uint64_t outer)
{
  const uint8_t *buf = utf8Buf();
  const uint8x16_t t1 = vld1q_u8(U8_TBL1);
  const uint8x16_t t2 = vld1q_u8(U8_TBL2);
  const uint8x16_t t3 = vld1q_u8(U8_TBL3);
  const uint8x16_t low4 = vdupq_n_u8(0x0F);
  uint8x16_t err = vdupq_n_u8(0);
  uint8x16_t prev = vdupq_n_u8(0);
  for (uint64_t o = 0; o < outer; o++)
  {
    CPU_STR_BARRIER();
    for (int i = 0; i < UTF8_BYTES; i += 16)
    {
      uint8x16_t in = vld1q_u8(buf + i);
      uint8x16_t p1 = vextq_u8(prev, in, 15);
      uint8x16_t p2 = vextq_u8(prev, in, 14);
      uint8x16_t p3 = vextq_u8(prev, in, 13);
      uint8x16_t sc = vandq_u8(
          vandq_u8(vqtbl1q_u8(t1, vshrq_n_u8(p1, 4)),
                   vqtbl1q_u8(t2, vandq_u8(p1, low4))),
          vqtbl1q_u8(t3, vshrq_n_u8(in, 4)));
      uint8x16_t m23 = vorrq_u8(vqsubq_u8(p2, vdupq_n_u8(0x60)),
                                vqsubq_u8(p3, vdupq_n_u8(0x70)));
      m23 = vandq_u8(m23, vdupq_n_u8(0x80));
      err = vorrq_u8(err, veorq_u8(sc, m23));
      prev = in;
    }
  }
  return (double)vaddlvq_u8(err);   // 0 on our valid input
}

#endif // utf8 ISA chain

} // anonymous namespace
} // namespace clpeak_cpu

#endif // CLPEAK_CORE_ONLY

#endif // ENABLE_CPU
#endif // CPU_KERN_STRING_COMPUTE_H
