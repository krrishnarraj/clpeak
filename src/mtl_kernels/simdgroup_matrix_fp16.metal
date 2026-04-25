// Apple simdgroup_matrix: fp16 inputs, fp32 accumulator, 8x8x8 tile per
// simdgroup (32 threads).  Apple silicon's tensor-core analog -- the only
// way to reach the device's headline fp16 TFLOPS number.
//
// Per simdgroup, one matmul = 8*8*8*2 = 1024 ops.  1024 outer iters per
// kernel launch keeps each launch long enough to amortize dispatch
// overhead while keeping the math meaningful: 1024 * 1024 = 1,048,576 ops
// per simdgroup = 32,768 ops/thread per launch (= MTL_SIMDGROUP_WORK_PER_WI).

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void simdgroup_matrix_fp16(
    device float* out [[buffer(0)]],
    constant float& A [[buffer(1)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]])
{
    // Stage tile values through threadgroup memory and load via
    // simdgroup_load -- portable across all MSL versions that ship the
    // simdgroup_matrix type, without depending on the make_filled_*
    // helper that older toolchains lack.
    threadgroup half  tg_a[64];
    threadgroup half  tg_b[64];

    // 32 threads in one simdgroup; each writes 2 elements to fill the 64-elt tile.
    tg_a[lid * 2 + 0] = (half)A;
    tg_a[lid * 2 + 1] = (half)A;
    tg_b[lid * 2 + 0] = (half)A;
    tg_b[lid * 2 + 1] = (half)A;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_matrix<half,  8, 8> a;
    simdgroup_matrix<half,  8, 8> b;
    simdgroup_matrix<float, 8, 8> c = simdgroup_matrix<float, 8, 8>(0.0f);

    simdgroup_load(a, tg_a, 8);
    simdgroup_load(b, tg_b, 8);

    for (int i = 0; i < 1024; i++)
    {
        simdgroup_multiply_accumulate(c, a, b, c);
    }

    threadgroup float tg_out[64];
    simdgroup_store(c, tg_out, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    out[tg_id * 64 + lid * 2 + 0] = tg_out[lid * 2 + 0];
    out[tg_id * 64 + lid * 2 + 1] = tg_out[lid * 2 + 1];
}
