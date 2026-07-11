#ifndef CPU_KERN_CRYPTO_COMPUTE_H
#define CPU_KERN_CRYPTO_COMPUTE_H

#ifdef ENABLE_CPU

// ===========================================================================
// Crypto / hash throughput chains: AES-128 encrypt, SHA-256 / SHA-512
// compression, and CRC32-C.  These
// measure the dedicated crypto pipes (every TLS stream, disk-encryption and
// checksum path runs through them), reported in GB/s: each kernel's
// opsPerIter is the BYTE count it processes per outer iteration, so the
// shared emitCompute() math lands directly in GB/s.
//
// Chain rules (same rationale as the compute chains):
//   - Every kernel is loop-carried (block/state/crc feeds the next step), so
//     LICM can't hoist the work; the crypto intrinsics are opaque target
//     intrinsics, so there is no closed form to scalar-evolve.
//   - NACC independent blocks/streams/chains provide the ILP that turns a
//     latency-bound single stream into a throughput measurement, and EVERY
//     chain is reduced into the sink so none is dead-code-eliminated.
//   - Keys / round constants / messages are synthetic: throughput is
//     data-independent, and we are measuring the pipes, not implementing
//     verified crypto.  The SHA kernels derive the message block from the
//     running state so the schedule instructions can't be hoisted either.
// Guarded by CLPEAK_CORE_ONLY like lowp_compute.h (cl.exe TUs are core-only;
// the crypto TUs are only built by clang/gcc/clang-cl anyway).
// ===========================================================================

#include "cpu_kernels.h"
#include "cpu_simd.h"

#include <cstring>

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

#ifndef CLPEAK_CORE_ONLY

namespace clpeak_cpu {
namespace {

// ---- AES-128 encrypt --------------------------------------------------------
// Full 10-round AES-128 on NACC independent 16-byte blocks; each block's
// ciphertext feeds back as the next plaintext (real dependency per block, ILP
// across blocks).  Round keys are arbitrary constants: 10 keys + NACC blocks
// exceed the 16-XMM file on x86, so clang reloads some keys from the stack in
// the hot loop -- that is how real AES round keys are fed, so it stays in.
#if defined(__AES__)
#define CPU_HAS_AES_KERNEL 1
static constexpr int AES_NACC = 8, AES_ROUNDS = 10, AES_BLOCK_BYTES = 16;
static double runAesChain(uint64_t outer)
{
  __m128i blk[AES_NACC], rk[AES_ROUNDS];
  volatile int vseed = 0x1234567;
  const int seed = vseed;
  for (int r = 0; r < AES_ROUNDS; r++) rk[r] = _mm_set1_epi32(seed * (r + 1) + r);
  for (int j = 0; j < AES_NACC; j++) blk[j] = _mm_set1_epi32(seed + j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < AES_NACC; j++)
      {
        __m128i b = blk[j];
        CPU_UNROLL_FULL
        for (int r = 0; r < AES_ROUNDS - 1; r++) b = _mm_aesenc_si128(b, rk[r]);
        blk[j] = _mm_aesenclast_si128(b, rk[AES_ROUNDS - 1]);
      }
    }
  __m128i s = blk[0];
  for (int j = 1; j < AES_NACC; j++) s = _mm_xor_si128(s, blk[j]);
  alignas(16) uint64_t tmp[2]; _mm_store_si128((__m128i *)tmp, s);
  return (double)(tmp[0] ^ tmp[1]);
}
#elif defined(__ARM_FEATURE_AES) || defined(__ARM_FEATURE_CRYPTO)
#define CPU_HAS_AES_KERNEL 1
// ARM AESE = AddRoundKey+SubBytes+ShiftRows, AESMC = MixColumns; the AESE+AESMC
// pair fuses on every big core.  32 vector regs fit 12 blocks + 11 keys, and
// the AES pipes are wider than x86 (Apple: 2+ per P-core), so NACC is higher.
static constexpr int AES_NACC = 12, AES_ROUNDS = 10, AES_BLOCK_BYTES = 16;
static double runAesChain(uint64_t outer)
{
  uint8x16_t blk[AES_NACC], rk[AES_ROUNDS + 1];
  volatile uint8_t vseed = 0x3C;
  const uint8_t seed = vseed;
  for (int r = 0; r <= AES_ROUNDS; r++) rk[r] = vdupq_n_u8((uint8_t)(seed + 17 * r));
  for (int j = 0; j < AES_NACC; j++) blk[j] = vdupq_n_u8((uint8_t)(seed ^ j));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < AES_NACC; j++)
      {
        uint8x16_t b = blk[j];
        CPU_UNROLL_FULL
        for (int r = 0; r < AES_ROUNDS - 1; r++) b = vaesmcq_u8(vaeseq_u8(b, rk[r]));
        blk[j] = veorq_u8(vaeseq_u8(b, rk[AES_ROUNDS - 1]), rk[AES_ROUNDS]);
      }
    }
  uint8x16_t s = blk[0];
  for (int j = 1; j < AES_NACC; j++) s = veorq_u8(s, blk[j]);
  return (double)(vgetq_lane_u64(vreinterpretq_u64_u8(s), 0) ^
                  vgetq_lane_u64(vreinterpretq_u64_u8(s), 1));
}
#endif

// ---- VAES-512 (x86 EVEX 512-bit AES: 4 blocks per instruction) --------------
#if defined(__VAES__) && defined(__AVX512F__)
#define CPU_HAS_VAES_KERNEL 1
static constexpr int VAES_NACC = 8, VAES_ROUNDS = 10, VAES_BLOCK_BYTES = 64;
static double runVaesChain(uint64_t outer)
{
  __m512i blk[VAES_NACC], rk[VAES_ROUNDS];
  volatile int vseed = 0x1234567;
  const int seed = vseed;
  for (int r = 0; r < VAES_ROUNDS; r++) rk[r] = _mm512_set1_epi32(seed * (r + 1) + r);
  for (int j = 0; j < VAES_NACC; j++) blk[j] = _mm512_set1_epi32(seed + j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < VAES_NACC; j++)
      {
        __m512i b = blk[j];
        CPU_UNROLL_FULL
        for (int r = 0; r < VAES_ROUNDS - 1; r++) b = _mm512_aesenc_epi128(b, rk[r]);
        blk[j] = _mm512_aesenclast_epi128(b, rk[VAES_ROUNDS - 1]);
      }
    }
  __m512i s = blk[0];
  for (int j = 1; j < VAES_NACC; j++) s = _mm512_xor_si512(s, blk[j]);
  return (double)_mm512_reduce_add_epi64(s);
}
#endif

// ---- SHA-256 compression -----------------------------------------------------
// Full 64-round compression (rounds + message schedule) on NSTR independent
// streams; the message block is derived from the running state, so neither the
// schedule nor the rounds are hoistable.  Round constants use one synthetic K
// vector (distinct real Ks change nothing about throughput).  Bytes = 64 per
// stream per compression.
#if defined(__SHA__)
#define CPU_HAS_SHA256_KERNEL 1
// x86 SHA-NI: 16 XMM regs cap this at 2 streams (2 state + 4 W each, plus
// temps); sha256rnds2 is the serial bottleneck per stream.
static constexpr int SHA256_NSTR = 2, SHA256_BLOCK_BYTES = 64;
static double runSha256Chain(uint64_t outer)
{
  __m128i s0[SHA256_NSTR], s1[SHA256_NSTR];
  volatile int vseed = 0x6a09e667;
  const int seed = vseed;
  const __m128i K = _mm_set_epi32(0xc67178f2, 0x8f1bbcdc, 0x5a827999, seed);
  for (int t = 0; t < SHA256_NSTR; t++)
  {
    s0[t] = _mm_set1_epi32(seed + t);
    s1[t] = _mm_set1_epi32(seed ^ (t + 1));
  }
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int t = 0; t < SHA256_NSTR; t++)
      {
        // Message derived from state -> the schedule can't be loop-hoisted.
        __m128i W[4];
        W[0] = s0[t];
        W[1] = s1[t];
        W[2] = _mm_xor_si128(s0[t], s1[t]);
        W[3] = _mm_add_epi32(s0[t], s1[t]);
        __m128i a = s0[t], b = s1[t];
        CPU_UNROLL_FULL
        for (int r = 0; r < 16; r++)   // 16 quads x 4 rounds = 64 rounds
        {
          __m128i msg = _mm_add_epi32(W[r & 3], K);
          b = _mm_sha256rnds2_epu32(b, a, msg);
          msg = _mm_shuffle_epi32(msg, 0x0E);
          a = _mm_sha256rnds2_epu32(a, b, msg);
          if (r < 12)                  // expand W[r] -> W[r+4]
          {
            __m128i tmp = _mm_alignr_epi8(W[(r + 3) & 3], W[(r + 2) & 3], 4);
            W[r & 3] = _mm_sha256msg2_epu32(
                _mm_add_epi32(_mm_sha256msg1_epu32(W[r & 3], W[(r + 1) & 3]), tmp),
                W[(r + 3) & 3]);
          }
        }
        s0[t] = _mm_add_epi32(s0[t], a);   // Davies-Meyer feedback
        s1[t] = _mm_add_epi32(s1[t], b);
      }
    }
  __m128i s = _mm_xor_si128(s0[0], s1[0]);
  for (int t = 1; t < SHA256_NSTR; t++)
    s = _mm_xor_si128(s, _mm_xor_si128(s0[t], s1[t]));
  alignas(16) uint64_t tmp[2]; _mm_store_si128((__m128i *)tmp, s);
  return (double)(tmp[0] ^ tmp[1]);
}
#elif defined(__ARM_FEATURE_SHA2) || defined(__ARM_FEATURE_CRYPTO)
#define CPU_HAS_SHA256_KERNEL 1
// ARM: 32 vector regs fit 4 streams (2 state + 4 W each).
static constexpr int SHA256_NSTR = 4, SHA256_BLOCK_BYTES = 64;
static double runSha256Chain(uint64_t outer)
{
  uint32x4_t s0[SHA256_NSTR], s1[SHA256_NSTR];
  volatile uint32_t vseed = 0x6a09e667u;
  const uint32_t seed = vseed;
  const uint32x4_t K = vdupq_n_u32(0x428a2f98u ^ seed);
  for (int t = 0; t < SHA256_NSTR; t++)
  {
    s0[t] = vdupq_n_u32(seed + (uint32_t)t);
    s1[t] = vdupq_n_u32(seed ^ (uint32_t)(t + 1));
  }
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int t = 0; t < SHA256_NSTR; t++)
      {
        uint32x4_t W[4];
        W[0] = s0[t];
        W[1] = s1[t];
        W[2] = veorq_u32(s0[t], s1[t]);
        W[3] = vaddq_u32(s0[t], s1[t]);
        uint32x4_t a = s0[t], b = s1[t];
        CPU_UNROLL_FULL
        for (int r = 0; r < 16; r++)   // 16 quads x 4 rounds = 64 rounds
        {
          uint32x4_t wk = vaddq_u32(W[r & 3], K);
          if (r < 12)
            W[r & 3] = vsha256su1q_u32(vsha256su0q_u32(W[r & 3], W[(r + 1) & 3]),
                                       W[(r + 2) & 3], W[(r + 3) & 3]);
          uint32x4_t ta = a;
          a = vsha256hq_u32(a, b, wk);
          b = vsha256h2q_u32(b, ta, wk);
        }
        s0[t] = vaddq_u32(s0[t], a);
        s1[t] = vaddq_u32(s1[t], b);
      }
    }
  uint32x4_t s = veorq_u32(s0[0], s1[0]);
  for (int t = 1; t < SHA256_NSTR; t++)
    s = veorq_u32(s, veorq_u32(s0[t], s1[t]));
  return (double)(vgetq_lane_u32(s, 0) ^ vgetq_lane_u32(s, 1) ^
                  vgetq_lane_u32(s, 2) ^ vgetq_lane_u32(s, 3));
}
#endif

// ---- SHA-512 compression (ARM FEAT_SHA512) -----------------------------------
// 80 rounds = 40 vsha512h/vsha512h2 pairs per 128-byte block, plus the su0/su1
// message schedule, following ARM's reference 2-round flow (state as a rotating
// ring of 4 u64x2 pairs).  The message is synthetic and state-derived; the
// dependency structure and instruction mix match a real implementation, which
// is what the throughput measures.  Register budget caps this at 2 streams
// (8 W + 4 state each).  x86 has a SHA512 EVEX extension (Arrow Lake) but no
// kernel here yet -- its row stays Unsupported on x86.
#if defined(__ARM_FEATURE_SHA512)
#define CPU_HAS_SHA512_KERNEL 1
static constexpr int SHA512_NSTR = 2, SHA512_BLOCK_BYTES = 128;
static double runSha512Chain(uint64_t outer)
{
  uint64x2_t st[SHA512_NSTR][4];   // {ab, cd, ef, gh} per stream
  volatile uint64_t vseed = 0x6a09e667f3bcc908ull;
  const uint64_t seed = vseed;
  const uint64x2_t K = vdupq_n_u64(0x428a2f98d728ae22ull ^ seed);
  for (int t = 0; t < SHA512_NSTR; t++)
    for (int i = 0; i < 4; i++)
      st[t][i] = vdupq_n_u64(seed + (uint64_t)(t * 4 + i + 1));
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int t = 0; t < SHA512_NSTR; t++)
      {
        uint64x2_t W[8];
        CPU_UNROLL_FULL
        for (int i = 0; i < 8; i++)
          W[i] = veorq_u64(st[t][i & 3], vdupq_n_u64((uint64_t)i * 0x9e3779b97f4a7c15ull));
        uint64x2_t s0 = st[t][0], s1 = st[t][1], s2 = st[t][2], s3 = st[t][3];
        CPU_UNROLL_FULL
        for (int r = 0; r < 40; r++)   // 40 pairs x 2 rounds = 80 rounds
        {
          if (r < 32)                  // expand W[r] -> W[r+8]
            W[r & 7] = vsha512su1q_u64(vsha512su0q_u64(W[r & 7], W[(r + 1) & 7]),
                                       W[(r + 7) & 7],
                                       vextq_u64(W[(r + 4) & 7], W[(r + 5) & 7], 1));
          uint64x2_t sum = vaddq_u64(W[r & 7], K);
          uint64x2_t inter = vaddq_u64(vextq_u64(sum, sum, 1), s3);
          inter = vsha512hq_u64(inter, vextq_u64(s2, s3, 1), vextq_u64(s1, s2, 1));
          uint64x2_t nh = vsha512h2q_u64(inter, s1, s0);
          // 2 rounds shift the working pairs: (ab,cd,ef,gh) <-
          //   (h2-result, old ab, cd+intermed, old ef)
          uint64x2_t oldab = s0, oldef = s2;
          s0 = nh;
          s2 = vaddq_u64(s1, inter);
          s1 = oldab;
          s3 = oldef;
        }
        st[t][0] = vaddq_u64(st[t][0], s0);
        st[t][1] = vaddq_u64(st[t][1], s1);
        st[t][2] = vaddq_u64(st[t][2], s2);
        st[t][3] = vaddq_u64(st[t][3], s3);
      }
    }
  uint64x2_t s = st[0][0];
  for (int t = 0; t < SHA512_NSTR; t++)
    for (int i = (t == 0 ? 1 : 0); i < 4; i++) s = veorq_u64(s, st[t][i]);
  return (double)(vgetq_lane_u64(s, 0) ^ vgetq_lane_u64(s, 1));
}
#endif

// ---- CRC32-C (Castagnoli) ----------------------------------------------------
// Single-instruction hardware CRC, 8 bytes per op.  The crc value is the
// loop-carried chain (latency ~3, throughput 1), so NACC independent chains
// make it a throughput number.  x86 SSE4.2 only has the Castagnoli polynomial;
// ARM FEAT_CRC32 has both -- CRC32-C is emitted on both for comparability.
#if defined(__SSE4_2__) && (defined(__x86_64__) || defined(_M_X64))
#define CPU_HAS_CRC32C_KERNEL 1
static constexpr int CRC_NACC = 8, CRC_OP_BYTES = 8;
static double runCrc32cChain(uint64_t outer)
{
  unsigned long long crc[CRC_NACC];
  volatile uint64_t vseed = 0x0123456789abcdefull;
  const uint64_t seed = vseed;
  for (int j = 0; j < CRC_NACC; j++) crc[j] = seed >> j;
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < CRC_NACC; j++)
        crc[j] = _mm_crc32_u64(crc[j], seed);
    }
  uint64_t s = 0;
  for (int j = 0; j < CRC_NACC; j++) s ^= crc[j];
  return (double)s;
}
#elif defined(__ARM_FEATURE_CRC32)
#define CPU_HAS_CRC32C_KERNEL 1
static constexpr int CRC_NACC = 8, CRC_OP_BYTES = 8;
static double runCrc32cChain(uint64_t outer)
{
  uint32_t crc[CRC_NACC];
  volatile uint64_t vseed = 0x0123456789abcdefull;
  const uint64_t seed = vseed;
  for (int j = 0; j < CRC_NACC; j++) crc[j] = (uint32_t)(seed >> j);
  for (uint64_t o = 0; o < outer; o++)
    CPU_UNROLL_K
    for (int k = 0; k < INNER; k++)
    {
      CPU_UNROLL_FULL
      for (int j = 0; j < CRC_NACC; j++)
        crc[j] = __crc32cd(crc[j], seed);
    }
  uint32_t s = 0;
  for (int j = 0; j < CRC_NACC; j++) s ^= crc[j];
  return (double)s;
}
#endif

} // anonymous namespace
} // namespace clpeak_cpu

#endif // CLPEAK_CORE_ONLY

#endif // ENABLE_CPU
#endif // CPU_KERN_CRYPTO_COMPUTE_H
