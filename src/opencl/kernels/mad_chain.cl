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
// MP* is the mixed-precision family's second shape.  It is now just AF* over
// an fp32 accumulator with a narrowing round-trip every 128 chain
// instructions (MP*_128 + MP*_NARROW), which is the shape the CUDA, ROCm,
// Metal, Vulkan and oneAPI mp kernels all use.
//
// It used to narrow once per MAC, inside the chain:
//
//   old   a = wide(m) * wide(x) + a;  x = narrow(a);   every step
//
// which put an uncounted conversion in the dependent chain for every counted
// FMA -- two instructions issued per FMA, so at best half the fp32 rate.  An
// Arc A380 read 2.78 TFLOPS against 4.89 for fp32.  Note this is not a return
// to the version that carried the fp16 value through the loop and collapsed
// at vector widths (0.12x the squaring rate on Intel's CPU runtime from width
// 4 up): there are no half operands inside the loop at all now, only fp32
// values that are periodically rounded to fp16 and back.
//
// Because nothing in the loop is half any more, the old requirement that the
// invariant multiplier stay uniform across work-items is gone -- a is an fp32
// value and is seeded from the lane id, as the oneAPI mp alt kernel does.  The
// per-work-item variation the family needs (see compute_mp_kernels.cl) comes
// from it either way.  Folding is held off by the same thing that holds it off
// for AF*: the fold needs FP reassociation, which nothing in the toolchain set
// does by default, with MAX_ALT_CHAIN_RATIO as the backstop.
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
\n#undef MP_NARROW
\n#undef MP4_128
\n#undef MP4_NARROW
\n#undef MP2_128
\n#undef MP2_NARROW
\n#undef MP1_128
\n#undef MP1_NARROW
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
\n// Mixed-precision (mp) chains.  The accumulator is FT throughout and the
\n// narrow round-trip happens once per MP*_128 round, not once per MAC: the
\n// conversion is an uncounted instruction sitting in the dependent chain, so
\n// at one per MAC it issued two instructions per counted FMA and halved the
\n// reading.  An Arc A380 read mp at 2.78 TFLOPS against 4.89 for fp32 (57%)
\n// before this; the other five backends had always amortised it.  Matches the
\n// depth compute_bf16_v*.comp and the oneAPI bf16 kernel already used.
\n//
\n// These reuse the AF* float chains -- there are no half operands left inside
\n// the loop, so the "keep the multiplier uniform" rule the old MP_STEP form
\n// needed no longer applies; a is a float and may vary per work-item, exactly
\n// as the oneAPI mp alt kernel seeds it.
\n#define MP_NARROW(HT, FT, x)   x = convert_##FT(convert_##HT(x));
\n
\n#define MP4_128                AF4_16 AF4_16 AF4_16 AF4_16 AF4_16 AF4_16 AF4_16 AF4_16
\n#define MP4_NARROW(HT, FT)     MP_NARROW(HT,FT,x0) MP_NARROW(HT,FT,x1) MP_NARROW(HT,FT,x2) MP_NARROW(HT,FT,x3)
\n
\n#define MP2_128                AF2_16 AF2_16 AF2_16 AF2_16 AF2_16 AF2_16 AF2_16 AF2_16
\n#define MP2_NARROW(HT, FT)     MP_NARROW(HT,FT,x0) MP_NARROW(HT,FT,x1)
\n
\n#define MP1_128                AF1_16 AF1_16 AF1_16 AF1_16 AF1_16 AF1_16 AF1_16 AF1_16
\n#define MP1_NARROW(HT, FT)     MP_NARROW(HT,FT,x0)
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
