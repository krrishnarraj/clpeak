// dp4a_chain.glsl -- the dot-product chains every int8_dp shader runs.
//
// Each .comp that includes this is compiled twice, plain and with
// -DDP4A_CHAIN_ALT, and runComputeKernel races the two -- the alt build at
// two subgroup widths too -- and reports the fastest.  The includer calls
//
//   DP_DECL_X(A)          once, with the push constant
//   DP_CHAIN_DECL(k, s)   for each independent chain k, seeded at s; chain
//                         seeds sit DP_CHAIN_STRIDE apart
//   DP_CHAIN_16(k)        for sixteen dots of chain k
//   DP_CHAIN_SUM(k)       for the value chain k leaves behind
//
// Each dotPacked4x8AccSatEXT is 4 signed INT8 multiply-adds into a 32-bit
// accumulator = 8 INT8 ops, on the uint32-packed operands that are the
// hardware-native form on NVIDIA, AMD and Intel (dp4a / V_DOT4 / IDPAS); an
// earlier i8vec4 version ran ~6x slower on NVIDIA from the pack/unpack around
// it.  The int (not uint) operands are what make glslang pick the signed
// overload and emit OpSDotAccSat + PackedVectorFormat4x8Bit.
//
// The chain shape.  Three constraints have to hold at once:
//
//  - Both multiplicands may not be loop-invariant.  dp4a with fixed
//    operands is `a + n*dot(x, y)`, which a driver may and does fold: the
//    shape read 74939 GOPS on an RTX 5060 whose dp4a peak, measured
//    through CUDA's own __dp4a on the same card, is 33928.
//
//  - Nothing may run between the dots.  The obvious way to keep an operand
//    moving is to rewrite it from the accumulator (`y ^= a`), but that XOR
//    is a second dependent integer op per dot and the op budget does not
//    credit it; on an Arc A380 it cost more than half the rate (8832 GOPS
//    against 19497 for the same instruction with nothing beside it).
//
//  - All three source operands must be distinct registers.  Intel
//    Alchemist halves a three-source op that reads the same register twice
//    -- the rule the MAD chains in mad_chain.glsl are built around -- so
//    `a = dot(x, a, a)` is not the answer either.
//
// The plain build: two accumulators feeding each other.  Each dot reads {x,
// the other accumulator, its own}, three distinct registers, and writes its
// own.  Every dot depends on the one before it, so a pair is one dependent
// chain; and because the dot extracts bytes of a value that is itself a
// 32-bit accumulator, the recurrence is not affine and has no closed form to
// fold to.
//
// The alt build: four accumulators in a cycle, each dot multiplying the next
// two -- a = dot(b, c, a), b = dot(c, d, b), c = dot(d, a, c), d = dot(a, b,
// d).  Nothing loop-invariant is left in it, and it keeps all three rules.
// It is there for Alchemist's register banks.  Intel's compiler places a
// three-source op's operands greedily (setupBankConflictsforMad in IGC's
// register allocator): values local to the block first, src2 before src1
// before src0, the first two it places going to opposite banks.  In the pair
// that puts each dot's src0 and src2 -- the two accumulators -- in opposite
// banks before it ever reaches x, the src1 of every dot, so x shares a bank
// with the other multiplicand at every other dot.  In the cycle the first two
// it reaches are both multiplicands, and consecutive members alternate banks
// cleanly.  It also has two dots in flight per chain where the pair has one.
// An Arc A380 read the pair at 13.0 TOPS pinned to SIMD32 and about 16 at the
// SIMD16 its driver picks unpinned, against vkpeak's 19.5 for a fixed-operand
// chain -- the shape the first rule above forbids.

#ifndef DP4A_CHAIN_GLSL
#define DP4A_CHAIN_GLSL

#define DP(p, q, acc)  dotPacked4x8AccSatEXT(p, q, acc)

#ifdef DP4A_CHAIN_ALT

  #define DP_CHAIN_STRIDE 16
  #define DP_DECL_X(A)  int dpBias = (A);
  #define DP_CHAIN_DECL(k, s)                                                 \
      int a##k = (s) + dpBias;       int b##k = (s) + dpBias + 4;             \
      int c##k = (s) + dpBias + 8;   int d##k = (s) + dpBias + 12;
  #define DP_CHAIN_4(k)                                                       \
      a##k = DP(b##k, c##k, a##k);  b##k = DP(c##k, d##k, b##k);              \
      c##k = DP(d##k, a##k, c##k);  d##k = DP(a##k, b##k, d##k);
  #define DP_CHAIN_SUM(k)  ((a##k + b##k) + (c##k + d##k))

#else

  #define DP_CHAIN_STRIDE 8
  // x holds four int8 lanes packed LSB->MSB: (A, A+1, A+2, A+3).
  #define DP_DECL_X(A)                                                        \
      int x = ((A) & 0xff)                                                    \
            | ((((A) + 1) & 0xff) << 8)                                       \
            | ((((A) + 2) & 0xff) << 16)                                      \
            | ((((A) + 3) & 0xff) << 24);
  #define DP_CHAIN_DECL(k, s)  int a##k = (s);  int b##k = (s) + 4;
  #define DP_CHAIN_4(k)                                                       \
      a##k = DP(x, b##k, a##k);  b##k = DP(x, a##k, b##k);                    \
      a##k = DP(x, b##k, a##k);  b##k = DP(x, a##k, b##k);
  #define DP_CHAIN_SUM(k)  (a##k + b##k)

#endif

#define DP_CHAIN_16(k)  DP_CHAIN_4(k) DP_CHAIN_4(k) DP_CHAIN_4(k) DP_CHAIN_4(k)

#endif // DP4A_CHAIN_GLSL
