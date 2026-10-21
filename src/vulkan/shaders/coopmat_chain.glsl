// coopmat_chain.glsl -- the MulAdd run every cooperative-matrix shader runs.
//
// Includers define, before including this file:
//
//   CM_AB         the A/B component type    (float16_t, int8_t, floate4m3_t...)
//   CM_ACC        the accumulator type      (float, int32_t)
//   CM_VAL(i, n)  element n of the i-th tile, cast to CM_AB, derived from the
//                 push constant so the driver cannot constant-fold a
//                 product.  n = 0 is the value the whole tile is built from;
//                 every other n must give an expression of its own (the
//                 fourth property below)
//
// and must have M/N/K in scope (the specialization constants carrying the
// tile the driver advertised), a push-constant block `pc` with an int
// `spread` the host leaves 0, and an `outputBuf.data` array to store into.
// The header then supplies CM_DECLARE for the matrices, CM_MMA_TRIP for one
// trip of the inner loop and CM_STORE_ALL for the store at the end.  An
// includer whose accumulator is not what it stores defines CM_STORE(mat,
// slot) itself first.
//
// Shape, and why.  Four properties have to hold at once, and each of them
// has cost this benchmark a wrong number before:
//
//  - The MulAdds must run as a straight-line block, not one per loop trip.
//    A back-edge between every MulAdd halved an A380's rate (12.5 TFLOPS fp16
//    rolled against 25.9 unrolled).  CM_MMA_TRIP is 16 of them.
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
//  - No compiler may be able to see that a tile is one value.  A driver
//    without a matrix unit lowers the MulAdd to scalar arithmetic, a dot
//    product per output element, and in a tile built from one scalar every
//    one of those dot products is the same expression, which it computes
//    once.  Lavapipe does exactly that -- NIR scalarizes its lowered MulAdd
//    and CSEs it -- so on a Threadripper PRO 3955WX llvmpipe ran one
//    multiply, a seven-add reduction and eight vector adds per 8x8x8 MulAdd,
//    72 of the 1024 flops credited, and read 3.06 TFLOPS fp16 on a CPU whose
//    fp32 peak is 2.2.  With every element its own expression it runs all
//    1024 and reads 0.13.  Filling every element costs a GPU, though: in
//    work-groups of 8 trips it took 11% off an RTX 5060's int8 row and 2%
//    off fp8.  So the tiles are built from one scalar as before, and
//    refilled element by element only if the push constant's `spread` is
//    set, which the host never does.  No compiler can rule that out, so none
//    can treat a tile as one value -- and losing that, not the branch, is
//    what a GPU still pays, once per work-group before the loop: 1.6% of the
//    5060's int8 reading at 8 trips a work-group, nothing measurable at 32,
//    which is why work-groups run four times the budget (kCoopGroupScale in
//    coopmat.cpp).
//
// Two builds of every shader, raced by runComputeKernel: the same sixteen
// products, carried on one accumulator or on four.
//
// One accumulator is the plain build, and where we can check it, it is the
// peak: an RTX 5060 reads 42.33 TFLOPS fp16 coopmat against 42.53 for the
// same tile through CUDA WMMA.
//
// Four accumulators (CM_CHAIN_ALT) are for hardware that cannot overlap a
// MulAdd with the one it accumulates onto.  Intel's compiler forwards an
// accumulator between dependent DPAS only on Xe3P (IGC's
// hasDpasFwdAndDoubleSrcReadSupression), so on Alchemist every MulAdd of the
// single chain waits out the last one.  What it does group into one
// back-to-back macro is consecutive DPAS that share no register they write
// and read the same B tile (canInSameDPASMacro in IGC's local scheduler), and
// the alt build issues exactly that: each B tile meets all four accumulators
// in a row.  The single chain's cost shows in
// int8, whose 8x8x32 tile is twice fp16's 8x8x16 and so must run at twice
// its rate: an A380 read 22.0 TOPS int8 against 14.5 TFLOPS fp16 (1.52x),
// where oneAPI's joint_matrix read 31.0 against 15.5 and vkpeak 32 against
// 16.2 on the same driver.
//
// Four accumulators fed the same products would each differ from the first
// by a constant -- derivable, not independent, whatever their seeds.  These
// take four different products each, so their per-trip increments differ and
// none can be had without the MulAdds.  They are stored to four tiles, not
// summed: a sum would be the one coopmat arithmetic op in the module, and an
// accumulator nobody reads is one the compiler may drop.  The host races the
// alt only where four accumulators fit in a modest share of the registers,
// and sizes the buffer for four tiles when it does (coopmat.cpp).

#ifndef COOPMAT_CHAIN_GLSL
#define COOPMAT_CHAIN_GLSL

// Must match COOPMAT_MMA_PER_TRIP in include/common/common.h, which is what
// the host divides the MulAdd budget by to get the trip count it pushes.
#define CM_MMA_PER_TRIP 16

#define CM_TA  coopmat<CM_AB,  gl_ScopeSubgroup, M, K, gl_MatrixUseA>
#define CM_TB  coopmat<CM_AB,  gl_ScopeSubgroup, K, N, gl_MatrixUseB>
#define CM_TC  coopmat<CM_ACC, gl_ScopeSubgroup, M, N, gl_MatrixUseAccumulator>

// Tile `i` element by element: element e of this invocation's share is
// numbered e + length * invocation, so no two elements of the tile share a
// number, in one invocation or across the subgroup.  Written through a copy:
// the element writes index the matrix at run time, which keeps the variable
// they write in memory, and the MulAdds should read a value.
#define CM_SPREAD(type, mat, i)                                               \
    {                                                                         \
        type elems;                                                           \
        for (int e = 0; e < elems.length(); e++)                              \
            elems[e] = CM_VAL(i, uint(e) + uint(elems.length()) *             \
                                          gl_LocalInvocationID.x);            \
        mat = elems;                                                          \
    }

// The eight tiles, each built from one scalar -- or element by element, on a
// branch the host never takes and no compiler can rule out (the fourth
// property above).
#define CM_DECLARE_AB                                                         \
    CM_TA matA0 = CM_TA(CM_VAL(0, 0));  CM_TA matA1 = CM_TA(CM_VAL(1, 0));    \
    CM_TA matA2 = CM_TA(CM_VAL(2, 0));  CM_TA matA3 = CM_TA(CM_VAL(3, 0));    \
    CM_TB matB0 = CM_TB(CM_VAL(4, 0));  CM_TB matB1 = CM_TB(CM_VAL(5, 0));    \
    CM_TB matB2 = CM_TB(CM_VAL(6, 0));  CM_TB matB3 = CM_TB(CM_VAL(7, 0));    \
    if (pc.spread != 0)                                                       \
    {                                                                         \
        CM_SPREAD(CM_TA, matA0, 0)  CM_SPREAD(CM_TA, matA1, 1)                \
        CM_SPREAD(CM_TA, matA2, 2)  CM_SPREAD(CM_TA, matA3, 3)                \
        CM_SPREAD(CM_TB, matB0, 4)  CM_SPREAD(CM_TB, matB1, 5)                \
        CM_SPREAD(CM_TB, matB2, 6)  CM_SPREAD(CM_TB, matB3, 7)                \
    }

#ifdef CM_CHAIN_ALT

// Tile `slot` of this work-group's four.
#define CM_TILES 4u

#define CM_DECLARE  CM_DECLARE_AB                                             \
    CM_TC matC0 = CM_TC(0);  CM_TC matC1 = CM_TC(0);                          \
    CM_TC matC2 = CM_TC(0);  CM_TC matC3 = CM_TC(0);

// One B tile against all four accumulators, each with a different A: four
// MulAdds that touch none of each other's registers.
#define CM_MMA_COL(b, a0, a1, a2, a3)                                         \
    matC0 = coopMatMulAdd(a0, b, matC0);  matC1 = coopMatMulAdd(a1, b, matC1); \
    matC2 = coopMatMulAdd(a2, b, matC2);  matC3 = coopMatMulAdd(a3, b, matC3);
#define CM_MMA_TRIP CM_MMA_COL(matB0, matA0, matA1, matA2, matA3)             \
                    CM_MMA_COL(matB1, matA1, matA2, matA3, matA0)             \
                    CM_MMA_COL(matB2, matA2, matA3, matA0, matA1)             \
                    CM_MMA_COL(matB3, matA3, matA0, matA1, matA2)

#define CM_STORE_ALL CM_STORE(matC0, 0u) CM_STORE(matC1, 1u)                  \
                     CM_STORE(matC2, 2u) CM_STORE(matC3, 3u)

#else

#define CM_TILES 1u

#define CM_DECLARE  CM_DECLARE_AB  CM_TC matC = CM_TC(0);

#define CM_MMA(a, b)  matC = coopMatMulAdd(a, b, matC);
#define CM_MMA_ROW(a) CM_MMA(a, matB0) CM_MMA(a, matB1) \
                      CM_MMA(a, matB2) CM_MMA(a, matB3)
#define CM_MMA_TRIP   CM_MMA_ROW(matA0) CM_MMA_ROW(matA1) \
                      CM_MMA_ROW(matA2) CM_MMA_ROW(matA3)

#define CM_STORE_ALL CM_STORE(matC, 0u)

#endif

#ifndef CM_STORE
#define CM_STORE(mat, slot)                                                   \
    coopMatStore(mat, outputBuf.data,                                         \
                 (gl_WorkGroupID.x * CM_TILES + (slot)) * M * N, N,           \
                 gl_CooperativeMatrixLayoutRowMajor);
#endif

#endif // COOPMAT_CHAIN_GLSL
