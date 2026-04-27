// Apple simdgroup_matrix: fp16 inputs, fp32 accumulator, 8x8x8 tile per
// simdgroup (32 threads).  Apple silicon's tensor-core analog -- the only
// way to reach the device's headline fp16 TFLOPS number.
//
// Four independent accumulator chains (c0..c3) so the matrix engine's
// multi-cycle pipeline can issue back-to-back instead of stalling on the
// c -> c dependency every iter (single-chain version on M1 Pro topped out
// at ~3.9 TFLOPS == 38% of the ~10.4 TFLOPS spec; 4 chains is the
// canonical ILP fix used in CUDA WMMA / Vulkan coopmat tuning).
//
// Per simdgroup ops = 1024 outer * 4 chains * 8*8*8*2 = 4,194,304;
// per thread = 131,072 (= MTL_SIMDGROUP_WORK_PER_WI in mtl_peak.mm).

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

    tg_a[lid * 2 + 0] = (half)A;
    tg_a[lid * 2 + 1] = (half)A;
    tg_b[lid * 2 + 0] = (half)A;
    tg_b[lid * 2 + 1] = (half)A;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<half,  8, 8> a;
    simdgroup_matrix<half,  8, 8> b;
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

    // Reduce so the stored tile depends on every accumulator -- otherwise
    // the compiler is free to drop chains it sees as dead.  Each fold is
    // one extra mma (4096 -> 4099 mma per simdgroup, ~0.07% noise on the
    // measured TFLOPS).
    simdgroup_multiply_accumulate(c0, a, b, c1);
    simdgroup_multiply_accumulate(c2, a, b, c3);
    simdgroup_multiply_accumulate(c0, a, b, c2);

    threadgroup float tg_out[64];
    simdgroup_store(c0, tg_out, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    out[tg_id * 64 + lid * 2 + 0] = tg_out[lid * 2 + 0];
    out[tg_id * 64 + lid * 2 + 1] = tg_out[lid * 2 + 1];
}
