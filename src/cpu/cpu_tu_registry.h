#ifndef CPU_TU_REGISTRY_H
#define CPU_TU_REGISTRY_H

// ===========================================================================
// Single source of truth for the set of per-ISA feature TUs.  Each XPUPEAK_TU(tag)
// entry names one TU; CMake compiles cpu_kernels_tu.cpp once per tag (with that
// ISA's flags) into xpupeak_table_<tag>(), and defines XPUPEAK_TU_<tag>=1 for the
// tags it actually built (a flag may be unavailable on a given compiler/target).
//
// This list drives the forward declarations of every table accessor in
// cpu_dispatch.cpp.  The declarations are emitted unconditionally -- declaring a
// symbol that is never defined is harmless; only *calling* one requires the TU,
// and every call site is guarded by #if XPUPEAK_TU_<tag> (kernels()) or reached
// through TU(<tag>), which yields an empty table for an unbuilt TU (kernelMenu()).
//
// Adding a new TU is therefore three edits:
//   1. add XPUPEAK_TU(<tag>) below,
//   2. add a xpupeak_add_isa_tu(<tag> <flags>) call in CMakeLists.txt,
//   3. wire the merge (kernels(), bandwidth; under #if XPUPEAK_TU_<tag>) + push
//      (kernelMenu(), one row per supported ISA; through TU(<tag>), never under
//      #if, so its predicate runs in every build) in cpu_dispatch.cpp.
// The order below is baseline -> widest, matching the merge/menu ordering.
// ===========================================================================

#define XPUPEAK_TU_REGISTRY(XPUPEAK_TU)                                      \
  /* x86 tiers */                                                           \
  XPUPEAK_TU(generic)  /* SSE2 floor (x86) / NEON floor (arm) / scalar */   \
  XPUPEAK_TU(sse42)                                                         \
  XPUPEAK_TU(avx)      /* AVX but no AVX2 (Sandy/Ivy Bridge) */             \
  XPUPEAK_TU(avx2)                                                          \
  XPUPEAK_TU(avxvnni)      /* 256-bit AVX-VNNI int8 dot */                  \
  XPUPEAK_TU(avxvnniint8)  /* 256-bit AVX-VNNI-INT8 signed dot */           \
  XPUPEAK_TU(avxvnniint16) /* 256-bit AVX-VNNI-INT16 int16 dot */           \
  XPUPEAK_TU(avx512)                                                        \
  XPUPEAK_TU(avx512vnni)                                                    \
  XPUPEAK_TU(avx512bf16)                                                    \
  XPUPEAK_TU(avx512fp16)                                                    \
  XPUPEAK_TU(avx10bf16)    /* AVX10.2-512 native bf16 vector FMA */         \
  XPUPEAK_TU(amx)          /* AMX int8 + bf16 */                            \
  XPUPEAK_TU(amxfp16)                                                       \
  XPUPEAK_TU(amxfp8)                                                        \
  /* Crypto TUs: same tag on both arches (only one arch branch builds it) */ \
  XPUPEAK_TU(aes)          /* x86 AES-NI / ARM FEAT_AES */                  \
  XPUPEAK_TU(vaes)         /* x86 512-bit VAES (Ice Lake+) */               \
  XPUPEAK_TU(sha)          /* x86 SHA-NI / ARM FEAT_SHA256 */               \
  XPUPEAK_TU(sha512)       /* ARM FEAT_SHA512 (+sha3); x86 SHA512 TBD */    \
  XPUPEAK_TU(crc)          /* ARM FEAT_CRC32 (x86 CRC32C rides in sse42) */ \
  /* ARM feature TUs (NEON base comes from the generic floor) */            \
  XPUPEAK_TU(fp16)                                                          \
  XPUPEAK_TU(fp16fml)                                                       \
  XPUPEAK_TU(dotprod)                                                       \
  XPUPEAK_TU(bf16)                                                          \
  XPUPEAK_TU(i8mm)                                                          \
  XPUPEAK_TU(fp8dot)       /* NEON fp8 4-way dot (FEAT_FP8DOT4; Vera) */    \
  /* ARM SVE (vector-length-agnostic) */                                    \
  XPUPEAK_TU(sve)                                                           \
  XPUPEAK_TU(svebf16)                                                       \
  XPUPEAK_TU(svei8mm)                                                       \
  XPUPEAK_TU(svefp8dot)    /* SVE2 fp8 4-way dot */                         \
  /* ARM SME (streaming; menu-push only, never merged into kernels()) */    \
  XPUPEAK_TU(sme)          /* FMOPA/BFMOPA/SMOPA + streaming-SVE vectors */ \
  XPUPEAK_TU(smef64)       /* FMOPA fp64 (FEAT_SME_F64F64) */

#endif // CPU_TU_REGISTRY_H
