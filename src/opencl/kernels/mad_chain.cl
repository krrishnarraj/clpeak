MSTRINGIFY(

// The alternate MAD chains, shared by every compute_*_alt_v* kernel.
//
// Each compute family defines its chain twice: the squaring recurrence
//
//     x = x*x + c                       (c a per-lane loop invariant)
//
// which the compute_*_v* kernels have always used, and a second shape with
// three distinct source registers, which the compute_*_alt_v* kernels add.
// runComputeTest times both and reports the faster.  Every macro here spells
// 16 chain instructions per _16, so the per-work-item op budget matches the
// kernel it is raced against and the two readings are comparable.
//
// Why two shapes.  No single recurrence reaches peak on every vendor:
//
//  - Intel Alchemist (Xe-HPG) halves a three-source mad unless all three
//    source operands are distinct registers.  x = x*x + c reads {x, x, c} and
//    lands at 0.496 instructions per lane per clock on an Arc A380, at every
//    vector width, every chain count and every work-group size.  So does
//    y = a*y + a, {a, y, a} -- it is the operand duplication, not the squaring.
//    Pre-Xe Intel (UHD 630) is unaffected.
//
//  - NVIDIA is the mirror image: one dependent chain reading three distinct
//    registers runs at half rate, which four independent chains restore.
//
//  - CPU OpenCL runtimes are latency-bound and need the independent chains
//    whatever the shape -- Intel's runtime and POCL both report a quarter of
//    what they can do on a single chain.
//
// AF* (affine, x_k = a*x_k + b) is the float families' second shape.  N is 4
// at vector width 1, 2 at width 2 and 1 from width 4 up, where the vector
// itself already supplies the instruction-level parallelism.
//
// RT* (rotating, x_k = x_k * x_(k+1) + c) is the integer families' second
// shape, and the reason they differ is not stylistic.  The affine form is
// foldable -- two steps compose into a*a*x + a*b + b -- and for integers that
// fold is a legal, ordinary optimisation, because integer multiply and add
// really are associative and distributive.  Apple's OpenCL compiler performs
// it: affine int16/char16/short16 came back 15.5x inflated.  Floating point is
// safe by comparison, since the same fold needs FP reassociation, which
// nothing in the toolchain set does by default.  RT keeps three distinct
// source registers but multiplies by another accumulator instead of a loop
// invariant, so the degree keeps rising and there is no closed form to fold
// to; x_k is rewritten only every N instructions, which is where its
// instruction-level parallelism comes from.  RT needs at least two
// accumulators to stay quadratic, so it never drops to one chain.
//
// M24_* is the same rotating shape as RT*, spelled with mad24 for the 24-bit
// fast-integer family.
//
// MP* is the mixed-precision family's second shape.  compute_mp measures
// fp16 x fp16 + fp32, so its chain has to keep the narrow type in the data
// path: x_k = narrow(m * wide(x_k) + b), with m an fp16 invariant and b an
// fp32 one.  That is three distinct source registers per mad, and exactly one
// narrowing conversion per mad -- the same ratio the squaring kernels carry,
// so adding chains does not quietly hand the alt build extra conversion work.
// The narrowing also blocks any fold outright, affine form or not.
//
// runComputeTest also refuses an alt reading more than MAX_ALT_CHAIN_RATIO
// times the squaring one, as a backstop against a fold nobody predicted.
//
// Live values per lane stay at ~5 in both shapes, which is what the MAD chain
// rules in include/common/common.h require.
//
// The chain is a plain contracted expression rather than fma()/mad() for the
// reason compute_sp_kernels.cl gives: some OpenCL frontends treat the builtin
// as a slow precise path.  Qualcomm's is 26x slower on it.

\n#undef CH_MAD
\n#undef AF4_DECL
\n#undef AF4_G
\n#undef AF4_16
\n#undef AF4_RES
\n#undef AF2_DECL
\n#undef AF2_G
\n#undef AF2_16
\n#undef AF2_RES
\n#undef AF1_DECL
\n#undef AF1_G
\n#undef AF1_16
\n#undef AF1_RES
\n#undef RT4_DECL
\n#undef RT4_G
\n#undef RT4_16
\n#undef RT4_RES
\n#undef RT2_DECL
\n#undef RT2_G
\n#undef RT2_16
\n#undef RT2_RES
\n#undef MP4_DECL
\n#undef MP4_16
\n#undef MP2_DECL
\n#undef MP2_16
\n#undef MP1_DECL
\n#undef MP1_16
\n#undef M24_4_DECL
\n#undef M24_4_16
\n#undef M24_2_DECL
\n#undef M24_2_16
\n
\n#define CH_MAD(d, m1, m2, ad)  d = (m1) * (m2) + (ad);
\n
\n#define AF4_DECL(T, seed, inv) T a = (inv); T b = a + (T)(2); T x0 = (seed); T x1 = (seed) + (T)(1); T x2 = (seed) + (T)(2); T x3 = (seed) + (T)(3);
\n#define AF4_G                  CH_MAD(x0, a, x0, b) CH_MAD(x1, a, x1, b) CH_MAD(x2, a, x2, b) CH_MAD(x3, a, x3, b)
\n#define AF4_16                 AF4_G AF4_G AF4_G AF4_G
\n#define AF4_RES                ((x0 + x1) + (x2 + x3))
\n
\n#define AF2_DECL(T, seed, inv) T a = (inv); T b = a + (T)(2); T x0 = (seed); T x1 = (seed) + (T)(1);
\n#define AF2_G                  CH_MAD(x0, a, x0, b) CH_MAD(x1, a, x1, b)
\n#define AF2_16                 AF2_G AF2_G AF2_G AF2_G AF2_G AF2_G AF2_G AF2_G
\n#define AF2_RES                (x0 + x1)
\n
\n#define AF1_DECL(T, seed, inv) T a = (inv); T b = a + (T)(2); T x0 = (seed);
\n#define AF1_G                  CH_MAD(x0, a, x0, b)
\n#define AF1_16                 AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G AF1_G
\n#define AF1_RES                (x0)
\n
\n#define RT4_DECL(T, seed, inv) T c = (inv); T x0 = (seed); T x1 = (seed) + (T)(1); T x2 = (seed) + (T)(2); T x3 = (seed) + (T)(3);
\n#define RT4_G                  CH_MAD(x0, x0, x1, c) CH_MAD(x1, x1, x2, c) CH_MAD(x2, x2, x3, c) CH_MAD(x3, x3, x0, c)
\n#define RT4_16                 RT4_G RT4_G RT4_G RT4_G
\n#define RT4_RES                ((x0 + x1) + (x2 + x3))
\n
\n#define RT2_DECL(T, seed, inv) T c = (inv); T x0 = (seed); T x1 = (seed) + (T)(1);
\n#define RT2_G                  CH_MAD(x0, x0, x1, c) CH_MAD(x1, x1, x0, c)
\n#define RT2_16                 RT2_G RT2_G RT2_G RT2_G RT2_G RT2_G RT2_G RT2_G
\n#define RT2_RES                (x0 + x1)
\n
\n#define MP_STEP(HT, FT, d, m, b)  d = convert_##HT(convert_##FT(m) * convert_##FT(d) + b);
\n
\n#define MP4_DECL(HT, FT, mseed, bseed, xseed) HT m = (mseed); FT b = (bseed); HT x0 = (xseed); HT x1 = (xseed) + (HT)(1); HT x2 = (xseed) + (HT)(2); HT x3 = (xseed) + (HT)(3);
\n#define MP4_G(HT, FT)          MP_STEP(HT,FT,x0,m,b) MP_STEP(HT,FT,x1,m,b) MP_STEP(HT,FT,x2,m,b) MP_STEP(HT,FT,x3,m,b)
\n#define MP4_16(HT, FT)         MP4_G(HT,FT) MP4_G(HT,FT) MP4_G(HT,FT) MP4_G(HT,FT)
\n#define MP4_RES(FT)            (convert_##FT(x0) + convert_##FT(x1) + convert_##FT(x2) + convert_##FT(x3))
\n
\n#define MP2_DECL(HT, FT, mseed, bseed, xseed) HT m = (mseed); FT b = (bseed); HT x0 = (xseed); HT x1 = (xseed) + (HT)(1);
\n#define MP2_G(HT, FT)          MP_STEP(HT,FT,x0,m,b) MP_STEP(HT,FT,x1,m,b)
\n#define MP2_16(HT, FT)         MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT)
\n#define MP2_RES(FT)            (convert_##FT(x0) + convert_##FT(x1))
\n
\n#define MP1_DECL(HT, FT, mseed, bseed, xseed) HT m = (mseed); FT b = (bseed); HT x0 = (xseed);
\n#define MP1_G(HT, FT)          MP_STEP(HT,FT,x0,m,b)
\n#define MP1_16(HT, FT)         MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT)
\n#define MP1_RES(FT)            (convert_##FT(x0))
\n
\n#define M24_MAD(d, m1, m2, ad)  d = mad24(m1, m2, ad);
\n
\n#define M24_4_DECL(T, seed, inv) T c = (inv); T x0 = (seed); T x1 = (seed) + (T)(1); T x2 = (seed) + (T)(2); T x3 = (seed) + (T)(3);
\n#define M24_4_G                  M24_MAD(x0, x0, x1, c) M24_MAD(x1, x1, x2, c) M24_MAD(x2, x2, x3, c) M24_MAD(x3, x3, x0, c)
\n#define M24_4_16                 M24_4_G M24_4_G M24_4_G M24_4_G
\n#define M24_4_RES                ((x0 + x1) + (x2 + x3))
\n
\n#define M24_2_DECL(T, seed, inv) T c = (inv); T x0 = (seed); T x1 = (seed) + (T)(1);
\n#define M24_2_G                  M24_MAD(x0, x0, x1, c) M24_MAD(x1, x1, x0, c)
\n#define M24_2_16                 M24_2_G M24_2_G M24_2_G M24_2_G M24_2_G M24_2_G M24_2_G M24_2_G
\n#define M24_2_RES                (x0 + x1)
\n

)
