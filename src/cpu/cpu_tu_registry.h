#ifndef CPU_TU_REGISTRY_H
#define CPU_TU_REGISTRY_H

// ===========================================================================
// Single source of truth for the set of per-ISA feature TUs.  Each CLPEAK_TU(tag)
// entry names one TU; CMake compiles cpu_kernels_tu.cpp once per tag (with that
// ISA's flags) into clpeak_table_<tag>(), and defines CLPEAK_TU_<tag>=1 for the
// tags it actually built (a flag may be unavailable on a given compiler/target).
//
// This list drives the forward declarations of every table accessor in
// cpu_dispatch.cpp.  The declarations are emitted unconditionally -- declaring a
// symbol that is never defined is harmless; only *calling* one requires the TU,
// and every call site in kernels()/kernelMenu() is guarded by #if CLPEAK_TU_<tag>.
//
// Adding a new TU is therefore three edits:
//   1. add CLPEAK_TU(<tag>) below,
//   2. add a clpeak_add_isa_tu(<tag> <flags>) call in CMakeLists.txt,
//   3. wire the merge (kernels(), bandwidth) + push (kernelMenu(), one row per
//      supported ISA) under #if CLPEAK_TU_<tag> in cpu_dispatch.cpp.
// The order below is baseline -> widest, matching the merge/menu ordering.
// ===========================================================================

#define CLPEAK_TU_REGISTRY(CLPEAK_TU)                                        \
  /* x86 tiers */                                                           \
  CLPEAK_TU(generic)  /* SSE2 floor (x86) / NEON floor (arm) / scalar */    \
  CLPEAK_TU(sse42)                                                          \
  CLPEAK_TU(avx2)                                                           \
  CLPEAK_TU(avxvnni)      /* 256-bit AVX-VNNI int8 dot */                   \
  CLPEAK_TU(avxvnniint8)  /* 256-bit AVX-VNNI-INT8 signed dot */            \
  CLPEAK_TU(avx512)                                                         \
  CLPEAK_TU(avx512vnni)                                                     \
  CLPEAK_TU(avx512bf16)                                                     \
  CLPEAK_TU(avx512fp16)                                                     \
  CLPEAK_TU(avx10bf16)    /* AVX10.2-512 native bf16 vector FMA */          \
  CLPEAK_TU(amx)          /* AMX int8 + bf16 */                             \
  CLPEAK_TU(amxfp16)                                                        \
  CLPEAK_TU(amxtf32)                                                        \
  CLPEAK_TU(amxfp8)                                                         \
  /* ARM feature TUs (NEON base comes from the generic floor) */            \
  CLPEAK_TU(fp16)                                                           \
  CLPEAK_TU(fp16fml)                                                        \
  CLPEAK_TU(dotprod)                                                        \
  CLPEAK_TU(bf16)                                                           \
  CLPEAK_TU(i8mm)                                                           \
  /* ARM SVE (vector-length-agnostic) */                                    \
  CLPEAK_TU(sve)                                                            \
  CLPEAK_TU(svebf16)                                                        \
  CLPEAK_TU(svei8mm)

#endif // CPU_TU_REGISTRY_H
