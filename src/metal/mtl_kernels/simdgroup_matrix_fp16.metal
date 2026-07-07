// Apple simdgroup_matrix: fp16 inputs, fp32 accumulator, 8x8x8 tile per
// simdgroup (32 threads).  M1-family GPUs have no dedicated matrix hardware:
// these ops lower onto the fp32 FMA pipes, so the ceiling is the device's
// fp32 FMA peak (~5.3 TFLOPS on a 16-core M1 Pro), not a higher "tensor"
// number.
//
// Sixteen independent accumulator chains keep the mma pipeline full: a
// single c -> c chain is latency-bound (~3.3 TFLOPS on M1 Pro); sixteen
// live chains reach ~5.1, ~97% of the FMA peak.
//
// simdgroup_multiply_accumulate(d, a, b, c) computes d = a*b + c -- the
// first argument is a pure DESTINATION, not accumulated into.  Never fold
// chains by writing into an accumulator that is still being counted: the
// overwritten chain becomes dead code, the compiler deletes it, and the
// reported TFLOPS inflate by the chain count (this kernel used to report
// 4x its real throughput that way).  Chains are reduced by storing each
// tile and summing per lane instead.
//
// Per simdgroup ops = 256 outer * 16 chains * 8*8*8*2 = 4,194,304;
// per thread = 131,072 (= MTL_SIMDGROUP_WORK_PER_WI in mtl_internal.h).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_matrix_fp16(
    device float* out [[buffer(0)]],
    constant float& A [[buffer(1)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]])
{
    threadgroup half tg_a[64];
    threadgroup half tg_b[64];

    // Distinct per-element operand values: a splat matrix would let the
    // compiler collapse each dot product sum_k(A*A) into one multiply
    // under fast math, eliminating most of the counted flops.
    tg_a[lid * 2 + 0] = (half)(A + (float)lid * 0.010f);
    tg_a[lid * 2 + 1] = (half)(A - (float)lid * 0.007f);
    tg_b[lid * 2 + 0] = (half)(A + (float)lid * 0.003f);
    tg_b[lid * 2 + 1] = (half)(A - (float)lid * 0.005f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<half,  8, 8> a;
    simdgroup_matrix<half,  8, 8> b;
    simdgroup_load(a, tg_a, 8);
    simdgroup_load(b, tg_b, 8);

    simdgroup_matrix<float, 8, 8> c0 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c1 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c2 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c3 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c4 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c5 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c6 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c7 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c8 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c9 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c10 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c11 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c12 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c13 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c14 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c15 = simdgroup_matrix<float, 8, 8>(0.0f);

    for (int i = 0; i < 256; i++)
    {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
        simdgroup_multiply_accumulate(c4, a, b, c4);
        simdgroup_multiply_accumulate(c5, a, b, c5);
        simdgroup_multiply_accumulate(c6, a, b, c6);
        simdgroup_multiply_accumulate(c7, a, b, c7);
        simdgroup_multiply_accumulate(c8, a, b, c8);
        simdgroup_multiply_accumulate(c9, a, b, c9);
        simdgroup_multiply_accumulate(c10, a, b, c10);
        simdgroup_multiply_accumulate(c11, a, b, c11);
        simdgroup_multiply_accumulate(c12, a, b, c12);
        simdgroup_multiply_accumulate(c13, a, b, c13);
        simdgroup_multiply_accumulate(c14, a, b, c14);
        simdgroup_multiply_accumulate(c15, a, b, c15);
    }

    // Reduce with every chain live: store each accumulator tile and sum the
    // two elements this lane owns.  A handful of stores + adds, ~0.1% of
    // the mma work.
    threadgroup float tg_out[64];
    float acc0 = 0.0f;
    float acc1 = 0.0f;
#define CLPEAK_FOLD_CHAIN(c)                                \
    simdgroup_store(c, tg_out, 8);                          \
    threadgroup_barrier(mem_flags::mem_threadgroup);        \
    acc0 += tg_out[lid * 2 + 0];                            \
    acc1 += tg_out[lid * 2 + 1];                            \
    threadgroup_barrier(mem_flags::mem_threadgroup);
    CLPEAK_FOLD_CHAIN(c0)
    CLPEAK_FOLD_CHAIN(c1)
    CLPEAK_FOLD_CHAIN(c2)
    CLPEAK_FOLD_CHAIN(c3)
    CLPEAK_FOLD_CHAIN(c4)
    CLPEAK_FOLD_CHAIN(c5)
    CLPEAK_FOLD_CHAIN(c6)
    CLPEAK_FOLD_CHAIN(c7)
    CLPEAK_FOLD_CHAIN(c8)
    CLPEAK_FOLD_CHAIN(c9)
    CLPEAK_FOLD_CHAIN(c10)
    CLPEAK_FOLD_CHAIN(c11)
    CLPEAK_FOLD_CHAIN(c12)
    CLPEAK_FOLD_CHAIN(c13)
    CLPEAK_FOLD_CHAIN(c14)
    CLPEAK_FOLD_CHAIN(c15)
#undef CLPEAK_FOLD_CHAIN

    out[tg_id * 64 + lid * 2 + 0] = acc0;
    out[tg_id * 64 + lid * 2 + 1] = acc1;
}
