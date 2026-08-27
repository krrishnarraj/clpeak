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
// MP* is the mixed-precision family's second shape, and it is deliberately a
// one-token edit of the squaring kernel rather than a rewrite:
//
//   squaring   a = wide(x) * wide(x) + a;  x = narrow(a);   sources {x, x, a}
//   MP*        a = wide(m) * wide(x) + a;  x = narrow(a);   sources {m, x, a}
//
// The fp32 accumulator stays loop-carried and the narrowing stays where it
// was, so the only difference is the duplicated multiplicand.  An earlier
// version carried the fp16 value through the loop instead and collapsed at
// vector widths -- 0.12x the squaring rate on Intel's CPU runtime from width
// 4 up.  The narrowing blocks any fold outright, affine form or not.
//
// runComputeTest also refuses an alt reading more than MAX_ALT_CHAIN_RATIO
// times the squaring one, as a backstop against a fold nobody predicted.
//
// Chain seeds are spaced CH_STRIDE apart, not 1 apart.  No two *scalar* chains
// may start on the same value: independent chains under the same recurrence
// stay bitwise identical forever, and a compiler that scalarises vectors then
// CSEs one of them away, inflating the reading by chains/(chains-1).  A vector
// seed already spans (A, A+1, ... A+W-1) across its own components, so at
// width 2 the old +1 spacing gave x0 = (A, A+1) and x1 = (A+1, A+2) -- one
// duplicated scalar chain out of four.  NVIDIA ran 3 of them and Vulkan's
// double2 read 423 GFLOPS on a 5060 whose FP64 units top out near 335.  Only
// AF* and MP* need this: RT*/M24_* seed the same way but rewrite x_k from the
// *other* accumulator, so equal starting values diverge on the first
// instruction rather than tracking each other.
//
// Live values per lane stay at ~5 in both shapes, which is what the MAD chain
// rules in include/common/common.h require.
//
// The chain is a plain contracted expression rather than fma()/mad() for the
// reason compute_sp_kernels.cl gives: some OpenCL frontends treat the builtin
// as a slow precise path.  Qualcomm's is 26x slower on it.

\n#undef CH_MAD
\n#undef CH_STRIDE
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
\n#define CH_STRIDE 4
\n
\n#define AF4_DECL(T, seed, inv) T a = (inv); T b = a + (T)(2); T x0 = (seed); T x1 = (seed) + (T)(CH_STRIDE); T x2 = (seed) + (T)(2*CH_STRIDE); T x3 = (seed) + (T)(3*CH_STRIDE);
\n#define AF4_G                  CH_MAD(x0, a, x0, b) CH_MAD(x1, a, x1, b) CH_MAD(x2, a, x2, b) CH_MAD(x3, a, x3, b)
\n#define AF4_16                 AF4_G AF4_G AF4_G AF4_G
\n#define AF4_RES                ((x0 + x1) + (x2 + x3))
\n
\n#define AF2_DECL(T, seed, inv) T a = (inv); T b = a + (T)(2); T x0 = (seed); T x1 = (seed) + (T)(CH_STRIDE);
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
\n#define MP_STEP(HT, FT, x, a, m)  a = convert_##FT(m) * convert_##FT(x) + a; x = convert_##HT(a);
\n
\n#define MP4_DECL(HT, FT, mseed, xseed, aseed) HT m = (mseed); HT x0 = (xseed); HT x1 = (xseed) + (HT)(CH_STRIDE); HT x2 = (xseed) + (HT)(2*CH_STRIDE); HT x3 = (xseed) + (HT)(3*CH_STRIDE); FT a0 = (aseed); FT a1 = (aseed) + (FT)(CH_STRIDE); FT a2 = (aseed) + (FT)(2*CH_STRIDE); FT a3 = (aseed) + (FT)(3*CH_STRIDE);
\n#define MP4_G(HT, FT)          MP_STEP(HT,FT,x0,a0,m) MP_STEP(HT,FT,x1,a1,m) MP_STEP(HT,FT,x2,a2,m) MP_STEP(HT,FT,x3,a3,m)
\n#define MP4_16(HT, FT)         MP4_G(HT,FT) MP4_G(HT,FT) MP4_G(HT,FT) MP4_G(HT,FT)
\n#define MP4_RES(FT)            (((a0 + a1) + (a2 + a3)) + convert_##FT(x0))
\n
\n#define MP2_DECL(HT, FT, mseed, xseed, aseed) HT m = (mseed); HT x0 = (xseed); HT x1 = (xseed) + (HT)(CH_STRIDE); FT a0 = (aseed); FT a1 = (aseed) + (FT)(CH_STRIDE);
\n#define MP2_G(HT, FT)          MP_STEP(HT,FT,x0,a0,m) MP_STEP(HT,FT,x1,a1,m)
\n#define MP2_16(HT, FT)         MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT) MP2_G(HT,FT)
\n#define MP2_RES(FT)            ((a0 + a1) + convert_##FT(x0))
\n
\n#define MP1_DECL(HT, FT, mseed, xseed, aseed) HT m = (mseed); HT x0 = (xseed); FT a0 = (aseed);
\n#define MP1_G(HT, FT)          MP_STEP(HT,FT,x0,a0,m)
\n#define MP1_16(HT, FT)         MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT) MP1_G(HT,FT)
\n#define MP1_RES(FT)            (a0 + convert_##FT(x0))
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
