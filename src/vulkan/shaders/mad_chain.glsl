// mad_chain.glsl -- the two chain shapes every compute-peak shader races.
//
// Each .comp that includes this is compiled twice: once plain, giving the
// squaring recurrence
//
//     x = x*x + c                       (c a per-lane loop invariant)
//
// and once with -DMAD_CHAIN_AFFINE, giving the affine recurrence over
// MAD_CHAINS independent accumulators
//
//     x_k = a*x_k + b                   (a, b per-lane loop invariants)
//
// Both spell exactly 16 chain instructions per MAD_16, so the loop trip counts
// and the per-work-item op budget are identical between the two and the
// readings are directly comparable.  runComputeKernel times both and reports
// the faster one.
//
// Why two shapes.  No single shape reaches peak on every vendor, because the
// two dominant register files have opposite constraints:
//
//  - Intel Alchemist (Xe-HPG) halves a three-source mad unless all three
//    source operands are distinct registers.  x = x*x + c reads {x, x, c} and
//    lands at 0.496 instructions per lane per clock on an A380 -- at every
//    vector width, every chain count and every work-group size.  So does
//    y = a*y + a, {a, y, a}: it is the duplication, not the squaring.
//
//  - NVIDIA is the mirror image.  A single dependent chain reading three
//    distinct registers is halved (0.51 on a 5060 and a 4060), while the
//    two-register x = x*x + c runs at full rate; four independent chains
//    restore the affine form to ~1.0.
//
// Racing the pair lands within 3% of the best measured shape on every device
// tested (Arc A380, RTX 5060, RTX 4060, Arc UHD 630, Adreno X1-45, M1 Pro via
// MoltenVK) where the squaring form alone reports half rate on Alchemist.
//
// MAD_CHAINS is set by the includer: 4 at vector width 1, 2 at width 2, 1 at
// width 4, keeping live values per lane at ~5 whatever the width -- the
// register-pressure rule in include/common/common.h still applies.
//
// MAD_CHAIN_INTEGER selects a rotating second shape instead of the affine one;
// integer shaders must set it, because an integer affine recurrence folds
// legally and one compiler in the fleet does fold it.  See the block below.
//
// MAD_OP defaults to the fma() builtin.  Integer shaders redefine it before
// including this file, since fma() is float-only.  The builtin costs nothing
// against a contracted expression on any Vulkan driver measured, including
// Adreno's -- unlike Qualcomm's OpenCL compiler, where it is 26x slower.

#ifndef MAD_CHAIN_GLSL
#define MAD_CHAIN_GLSL

#ifndef MAD_CHAINS
#define MAD_CHAINS 4
#endif

#ifndef MAD_OP
#define MAD_OP(d, m1, m2, ad)  d = fma(m1, m2, ad);
#endif

// Spacing between the seeds of two chains.  No two *scalar* chains may start
// on the same value: independent chains under the same recurrence stay bitwise
// identical forever, and a compiler that scalarises vectors then CSEs one of
// them away, inflating the reading by chains/(chains-1).  A vector seed already
// spans (A, A+1, ... A+W-1) across its own components, so chain k has to start
// a whole vector width away, not 1 away.  4 is the widest vector any shader
// here uses.  This is not hypothetical: at +1 the width-2 shaders had
// x0 = (A, A+1) and x1 = (A+1, A+2), and NVIDIA ran 3 of the 4 fp64 chains --
// double2 read 423 GFLOPS on a 5060 whose FP64 units top out near 335.
#define MAD_CHAIN_STRIDE 4

#if defined(MAD_CHAIN_AFFINE) && defined(MAD_CHAIN_INTEGER)

  // Integer families get a rotating chain rather than the affine one.  An
  // integer affine recurrence is legally foldable -- integer multiply and add
  // really are associative and distributive -- and Apple's OpenCL compiler
  // does fold it, returning 15.5x inflated numbers.  Rotating the multiplier
  // through the other accumulators keeps three distinct source registers and
  // the instruction-level parallelism while staying quadratic, so there is no
  // closed form to fold to.  It needs two accumulators minimum, so it never
  // drops to one chain the way the affine shape does at width 4.
  #if MAD_CHAINS == 4
    #define CHAIN_DECL(T, seed, inv)                                          \
        T c = (inv);                                                          \
        T x0 = (seed);                         T x1 = (seed) + T(MAD_CHAIN_STRIDE); \
        T x2 = (seed) + T(2*MAD_CHAIN_STRIDE); T x3 = (seed) + T(3*MAD_CHAIN_STRIDE);
    #define MAD_GROUP    MAD_OP(x0, x0, x1, c) MAD_OP(x1, x1, x2, c) \
                         MAD_OP(x2, x2, x3, c) MAD_OP(x3, x3, x0, c)
    #define MAD_16       MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP
    #define CHAIN_MAP(f) x0 = f(x0); x1 = f(x1); x2 = f(x2); x3 = f(x3);
    #define CHAIN_RESULT ((x0 + x1) + (x2 + x3))
  #else
    #define CHAIN_DECL(T, seed, inv)                                          \
        T c = (inv);  T x0 = (seed);  T x1 = (seed) + T(MAD_CHAIN_STRIDE);
    #define MAD_GROUP    MAD_OP(x0, x0, x1, c) MAD_OP(x1, x1, x0, c)
    #define MAD_16       MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP \
                         MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP
    #define CHAIN_MAP(f) x0 = f(x0); x1 = f(x1);
    #define CHAIN_RESULT (x0 + x1)
  #endif

#elif defined(MAD_CHAIN_AFFINE)

  #if MAD_CHAINS == 4
    #define CHAIN_DECL(T, seed, inv)                                          \
        T a = (inv);  T b = a + T(2);                                         \
        T x0 = (seed);                         T x1 = (seed) + T(MAD_CHAIN_STRIDE); \
        T x2 = (seed) + T(2*MAD_CHAIN_STRIDE); T x3 = (seed) + T(3*MAD_CHAIN_STRIDE);
    #define MAD_GROUP    MAD_OP(x0, a, x0, b) MAD_OP(x1, a, x1, b) \
                         MAD_OP(x2, a, x2, b) MAD_OP(x3, a, x3, b)
    #define MAD_16       MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP
    #define CHAIN_MAP(f) x0 = f(x0); x1 = f(x1); x2 = f(x2); x3 = f(x3);
    #define CHAIN_RESULT ((x0 + x1) + (x2 + x3))
  #elif MAD_CHAINS == 2
    #define CHAIN_DECL(T, seed, inv)                                          \
        T a = (inv);  T b = a + T(2);                                         \
        T x0 = (seed); T x1 = (seed) + T(MAD_CHAIN_STRIDE);
    #define MAD_GROUP    MAD_OP(x0, a, x0, b) MAD_OP(x1, a, x1, b)
    #define MAD_16       MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP \
                         MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP
    #define CHAIN_MAP(f) x0 = f(x0); x1 = f(x1);
    #define CHAIN_RESULT (x0 + x1)
  #else
    #define CHAIN_DECL(T, seed, inv)                                          \
        T a = (inv);  T b = a + T(2);                                         \
        T x0 = (seed);
    #define MAD_GROUP    MAD_OP(x0, a, x0, b)
    #define MAD_16       MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP \
                         MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP \
                         MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP \
                         MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP
    #define CHAIN_MAP(f) x0 = f(x0);
    #define CHAIN_RESULT (x0)
  #endif

#else  // squaring form -- one chain at every width

  #define CHAIN_DECL(T, seed, inv)  T c = (inv);  T x0 = (seed);
  #define MAD_GROUP    MAD_OP(x0, x0, x0, c)
  #define MAD_16       MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP \
                       MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP \
                       MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP \
                       MAD_GROUP MAD_GROUP MAD_GROUP MAD_GROUP
  #define CHAIN_MAP(f) x0 = f(x0);
  #define CHAIN_RESULT (x0)

#endif

// Deep inner loop for the shaders whose accumulator has to round-trip through
// a narrow type once per outer iteration (mp, bf16): 128 chain instructions,
// so the conversion amortises instead of sitting in the critical path.
#define MAD_128  MAD_16 MAD_16 MAD_16 MAD_16 MAD_16 MAD_16 MAD_16 MAD_16

#endif // MAD_CHAIN_GLSL
