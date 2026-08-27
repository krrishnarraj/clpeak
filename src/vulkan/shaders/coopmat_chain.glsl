// coopmat_chain.glsl -- the MulAdd run every cooperative-matrix shader runs.
//
// Includers define, before including this file:
//
//   CM_AB       the A/B component type      (float16_t, int8_t, floate4m3_t...)
//   CM_ACC      the accumulator type        (float, int32_t)
//   CM_VAL(i)   the i-th fill value, cast to CM_AB, derived from the push
//               constant so the driver cannot constant-fold a product
//
// and must have M/N/K in scope (the specialization constants carrying the
// tile the driver advertised).  The header then supplies CM_DECLARE for the
// matrices and CM_MMA_TRIP for one trip of the inner loop.
//
// Shape, and why.  Three properties have to hold at once, and each of them
// has cost this benchmark a wrong number before:
//
//  - The MulAdds must run as a straight-line block, not one per loop trip.
//    Intel's Xe-HPG groups consecutive accumulator-dependent DPAS into one
//    macro and issues them back to back; a loop back-edge between every
//    accumulate breaks the group and halves the rate (an A380 reads 12.5
//    TFLOPS fp16 rolled against 25.9 unrolled).  CM_MMA_TRIP is 16 of them.
//
//  - No two MulAdds in that block may share both operands.  Sixteen copies of
//    `matC += A@B` are `matC + 16*(A@B)`, and a compiler that reassociates
//    them into one product plus matrix adds reports double the real rate --
//    which is what an RTX 5060 did to the K=32 rows (int8 167 -> 339 TOPS,
//    past that card's 2:4-sparse ceiling).  Four A tiles and four B tiles give
//    sixteen distinct products, so the block has nothing to combine.
//
//  - The trip count must not be a compile-time constant.  It arrives in the
//    push constant, so the driver can neither unroll the whole run (512
//    MulAdds of an emulated tile is what a shader compiler chokes on) nor
//    turn it into a closed form.
//
// One accumulator chain, deliberately.  A second chain fed loop-invariant
// operands differs from the first by a constant at every step, so it is
// derivable rather than independent -- no choice of seed fixes that.  The
// single chain already reaches the hardware peak where we can check it: an
// RTX 5060 reads 42.33 TFLOPS fp16 coopmat against 42.53 for the same tile
// through CUDA WMMA.

#ifndef COOPMAT_CHAIN_GLSL
#define COOPMAT_CHAIN_GLSL

// Must match COOPMAT_MMA_PER_TRIP in include/common/common.h, which is what
// the host divides the MulAdd budget by to get the trip count it pushes.
#define CM_MMA_PER_TRIP 16

#define CM_TA  coopmat<CM_AB,  gl_ScopeSubgroup, M, K, gl_MatrixUseA>
#define CM_TB  coopmat<CM_AB,  gl_ScopeSubgroup, K, N, gl_MatrixUseB>
#define CM_TC  coopmat<CM_ACC, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>

#define CM_DECLARE                                                            \
    CM_TA matA0 = CM_TA(CM_VAL(0));  CM_TA matA1 = CM_TA(CM_VAL(1));          \
    CM_TA matA2 = CM_TA(CM_VAL(2));  CM_TA matA3 = CM_TA(CM_VAL(3));          \
    CM_TB matB0 = CM_TB(CM_VAL(4));  CM_TB matB1 = CM_TB(CM_VAL(5));          \
    CM_TB matB2 = CM_TB(CM_VAL(6));  CM_TB matB3 = CM_TB(CM_VAL(7));          \
    CM_TC matC  = CM_TC(0);

#define CM_MMA(a, b)  matC = coopMatMulAdd(a, b, matC);
#define CM_MMA_ROW(a) CM_MMA(a, matB0) CM_MMA(a, matB1) \
                      CM_MMA(a, matB2) CM_MMA(a, matB3)
#define CM_MMA_TRIP   CM_MMA_ROW(matA0) CM_MMA_ROW(matA1) \
                      CM_MMA_ROW(matA2) CM_MMA_ROW(matA3)

#endif // COOPMAT_CHAIN_GLSL
