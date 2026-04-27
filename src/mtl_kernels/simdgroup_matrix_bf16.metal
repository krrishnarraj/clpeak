// Apple simdgroup_matrix bf16 -- M3+ only (Apple9).  bfloat type requires
// Metal 3.1 (macOS 14+).  Same 4-chain ILP structure as the fp16 variant.

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

    tg_a[lid * 2 + 0] = (bfloat)A;
    tg_a[lid * 2 + 1] = (bfloat)A;
    tg_b[lid * 2 + 0] = (bfloat)A;
    tg_b[lid * 2 + 1] = (bfloat)A;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<bfloat, 8, 8> a;
    simdgroup_matrix<bfloat, 8, 8> b;
    simdgroup_load(a, tg_a, 8);
    simdgroup_load(b, tg_b, 8);

    simdgroup_matrix<float, 8, 8> c0 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c1 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c2 = simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c3 = simdgroup_matrix<float, 8, 8>(0.0f);

    for (int i = 0; i < 1024; i++)
    {
        simdgroup_multiply_accumulate(c0, a, b, c0);
        simdgroup_multiply_accumulate(c1, a, b, c1);
        simdgroup_multiply_accumulate(c2, a, b, c2);
        simdgroup_multiply_accumulate(c3, a, b, c3);
    }

    simdgroup_multiply_accumulate(c0, a, b, c1);
    simdgroup_multiply_accumulate(c2, a, b, c3);
    simdgroup_multiply_accumulate(c0, a, b, c2);

    threadgroup float tg_out[64];
    simdgroup_store(c0, tg_out, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    out[tg_id * 64 + lid * 2 + 0] = tg_out[lid * 2 + 0];
    out[tg_id * 64 + lid * 2 + 1] = tg_out[lid * 2 + 1];
}
