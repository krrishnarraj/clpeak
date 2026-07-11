#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

#ifdef ENABLE_CPU

#include <cstddef>
#include <cstdint>
#include <vector>

// ===========================================================================
// Runtime ISA dispatch.  The compute / read kernels are compiled once per
// feature TU (cpu_kernels_tu.cpp built N times with different -m/-arch flags;
// see CMakeLists.txt).  At runtime cpu_dispatch.cpp probes CPUID / HWCAP and
// assembles `kernels()` by picking, per kernel, the best variant whose full
// feature set the *running* CPU supports.  Taking the address of a higher-ISA
// function is safe at baseline; it is only ever *called* once guarded.
// ===========================================================================

namespace clpeak_cpu {

// Runtime CPU feature flags (filled by detectFeatures() in cpu_dispatch.cpp).
struct CpuFeatures {
  // x86
  bool sse42 = false, avx2 = false, fma = false;
  bool avx512f = false, avx512bw = false, avx512vl = false, avx512dq = false;
  bool avx512vnni = false, avx512bf16 = false, avx512fp16 = false;
  bool avxvnni = false;      // 256-bit VEX AVX-VNNI (Alder Lake+, Zen 5, Sierra Forest)
  bool avxvnniint8 = false;  // 256-bit AVX-VNNI-INT8 (signed×signed dpbssd; Zen 6, Lunar Lake)
  bool avxvnniint16 = false; // 256-bit AVX-VNNI-INT16 (mixed-sign int16 dot; Diamond Rapids, Nova Lake)
  bool avx10 = false, avx10_2_512 = false;  // AVX10.2 512-bit (Diamond Rapids): native bf16 FMA
  bool amx_tile = false, amx_int8 = false, amx_bf16 = false;
  bool amx_fp16 = false, amx_tf32 = false, amx_fp8 = false;  // Granite/Diamond Rapids AMX dtypes
  // ARM
  bool neon = false, fp16 = false, fp16fml = false, dotprod = false, bf16 = false, i8mm = false;
  // ARM SVE.  The compute kernels are vector-length-agnostic (one binary runs
  // 128-bit Oryon/Vera and 256-bit Graviton3); svebf16/svei8mm gate the SVE
  // BFDOT/BFMMLA and SMMLA paths independently (present on Neoverse V1 without
  // SVE2, and on every SVE2 core).
  bool sve = false, sve2 = false, svebf16 = false, svei8mm = false;
  // ARM SME (Scalable Matrix Extension).  `sme` gates the ZA outer-product
  // kernels + the streaming-SVE vector chains (Apple M4+, Snapdragon X2/8-Elite
  // Gen 5 clusters); `smeF64F64` gates the optional fp64 outer product.  `sme2`
  // is detection-only today (reported in the device header; the base-SME kernels
  // run on both SME1 and SME2 hardware).
  bool sme = false, sme2 = false, smeF64F64 = false;
  // ARM FP8 (2023 dpISA): FEAT_FP8DOT4 = native fp8 4-way dot -> fp32 in NEON
  // (and, with SVE2, in SVE).  First shipped by NVIDIA Vera (Olympus cores).
  bool fp8dot4 = false;
  // Crypto/hash silicon, unified across ISAs: `aes` = AES-NI (x86) / FEAT_AES
  // (ARM); `sha256` = SHA-NI / FEAT_SHA256; `sha512` = FEAT_SHA512 (x86 SHA512
  // is detection-only today -- no kernel yet); `crc32` = SSE4.2 CRC32 /
  // FEAT_CRC32.  `vaes` is the x86 512-bit EVEX form (Ice Lake+; needs the
  // AVX-512 OS grant, checked via avx512f at the menu push).
  bool aes = false, sha256 = false, sha512 = false, crc32 = false;
  bool vaes = false;
};

const CpuFeatures &cpuFeatures();   // cached runtime probe
const char       *isaName();        // widest active compute ISA, e.g. "AVX-512" / "NEON"
int               sveVLBytes();     // active SVE vector length in bytes (0 if no SVE)
int               smeSVLBytes();    // active SME streaming vector length in bytes (0 if no SME)

// A compute chain runs `outer` outer-iterations and returns a sink value.
using ChainFn = double (*)(uint64_t outer);
// A streaming read returns a checksum sink over `iters` passes of `M` floats.
using ReadFn = uint64_t (*)(const float *p, size_t M, uint64_t iters);

struct ChainVariant {
  ChainFn fn = nullptr;
  double  opsPerIter = 0.0;   // flops/ops one thread performs per outer-iteration
};

// One feature TU's offered kernels (null entries = not provided by that ISA).
struct CpuKernelTable {
  ChainVariant fp32, fp64, int32, fp16, bf16, mp, int8dp, mat_int8, mat_fp;
  // Newer x86 matrix/vector dtypes: AMX fp16 (Granite Rapids), AMX tf32 / AMX fp8
  // (Diamond Rapids), and native bf16 vector FMA (AVX10.2, full-rate, not a dot).
  ChainVariant mat_fp16, mat_tf32, mat_fp8, bf16fma;
  // int16 dot (x86 VPDPWSSD / VPDPWSUD) and ARM fp8 dot (FEAT_FP8DOT4 FDOT).
  ChainVariant int16dp, fp8dp;
  // ARM SME: fp32/fp64 ZA outer products (no x86 engine has these dtypes) and
  // the streaming-SVE vector chains (the only scalable-vector path on Apple,
  // which has SME but no non-streaming SVE).  The SME TU also fills mat_fp /
  // mat_fp16 / mat_int8 with its BFMOPA / FMOPA-f16 / SMOPA kernels.
  ChainVariant mat_fp32, mat_fp64, ssve_fp32, ssve_fp64;
  // Crypto/hash kernels (opsPerIter counts BYTES processed per outer iteration,
  // so emitCompute's giga() yields GB/s) and the fp divide/sqrt + scalar u64
  // integer-divide chains (opsPerIter counts divides/sqrts per outer iteration).
  ChainVariant aes, sha256, sha512, crc32c;
  // String kernels (opsPerIter counts BYTES per pass over the thread-local
  // L1-resident buffer -> GB/s): memchr-style byte scan (plus the historical
  // SSE4.2 PCMPISTRI form as its own variant) and UTF-8 lookup-shuffle
  // validation (PSHUFB / TBL).
  ChainVariant strscan, strscan_istri, utf8;
  ChainVariant div32, div64, sqrt32, sqrt64, intdiv;
  ReadFn       readsum = nullptr;
  const char  *isaName = "";
  // SVE vector length in bytes (svcntb()), set only by an SVE TU's table so the
  // dispatcher can report it in the device header.  0 on non-SVE tables.
  int          sveVLBytes = 0;
  // SME streaming vector length in bytes (svcntsb()), set only by an SME TU's
  // table.  0 on non-SME tables.
  int          smeSVLBytes = 0;
};

// The best kernels for THIS host (assembled once on first call).  Used by the
// bandwidth path (readsum), which is memory-bound and only needs one variant.
const CpuKernelTable &kernels();

// One compute variant tagged with its canonical ISA label (e.g. "AVX-512").
// The label lives here, not in ChainVariant, so it can differ per kernel slot
// for the same TU (a feature TU's int8dp is "AVX-512 VNNI" while its base fp32
// would be "AVX-512").
struct IsaVariant {
  ChainVariant v;
  const char  *isa = "";
};

// ALL supported variants of each compute kernel for THIS host, baseline-first
// (low ISA -> high ISA).  The compute tests run every entry so the user can
// compare instruction sets, rather than only the widest one (see kernels()).
struct CpuKernelMenu {
  std::vector<IsaVariant> fp32, fp64, int32, fp16, bf16, mp, int8dp, mat_int8, mat_fp;
  std::vector<IsaVariant> mat_fp16, mat_tf32, mat_fp8, bf16fma;
  std::vector<IsaVariant> int16dp, fp8dp, mat_fp32, mat_fp64;
  std::vector<IsaVariant> aes, sha256, sha512, crc32c;
  // String tests: the PCMPISTRI kernel rides in the strscan menu as its own
  // ISA row ("SSE4.2 PCMPISTRI"), next to the compare+movemask variants.
  std::vector<IsaVariant> strscan, utf8;
  std::vector<IsaVariant> div32, div64, sqrt32, sqrt64;
  // (the SME streaming-SVE fp32/fp64 chains ride in the fp32/fp64 menus above,
  //  labeled "SSVE" -- they are just another ISA variant of those tests.
  //  The scalar u64 intdiv has no per-ISA variants: it is read directly from
  //  kernels().intdiv, identical in every TU.)
};

// Assembled once on first call.
const CpuKernelMenu &kernelMenu();

} // namespace clpeak_cpu

#endif // ENABLE_CPU
#endif // CPU_KERNELS_H
