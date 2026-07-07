// Apple simdgroup_matrix bf16 -- M3+ only (Apple9).  bfloat type requires
// Metal 3.1 (macOS 14+).  Same 16-live-chain structure as the fp16 variant;
// see simdgroup_matrix_fp16.metal for why the chains must be reduced via
// store+sum (an mma "fold" dead-codes the overwritten chains and inflates
// the reported TFLOPS) and why the operands are non-uniform.
//
// Per simdgroup ops = 256 outer * 16 chains * 8*8*8*2 = 4,194,304;
// per thread = 131,072 (= MTL_SIMDGROUP_WORK_PER_WI in mtl_internal.h).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_matrix_bf16(
    device float* out [[buffer(0)]],
    constant float& A [[buffer(1)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]])
{
    threadgroup bfloat tg_a[64];
    threadgroup bfloat tg_b[64];

    tg_a[lid * 2 + 0] = (bfloat)(A + (float)lid * 0.010f);
    tg_a[lid * 2 + 1] = (bfloat)(A - (float)lid * 0.007f);
    tg_b[lid * 2 + 0] = (bfloat)(A + (float)lid * 0.003f);
    tg_b[lid * 2 + 1] = (bfloat)(A - (float)lid * 0.005f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<bfloat, 8, 8> a;
    simdgroup_matrix<bfloat, 8, 8> b;
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
