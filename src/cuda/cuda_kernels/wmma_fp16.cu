// WMMA fp16xfp16+fp32 m16n16k16 -- the canonical Volta+ tensor-core
// throughput test.  Mirrors src/shaders/coopmat_fp16.comp at the same
// tile shape and iteration count.
//
// Eight independent accumulator chains (c0..c7) so the tensor unit's
// multi-cycle pipeline can issue back-to-back MMAs instead of stalling on
// the c -> c dependency every iter.  Single-chain on RTX 5060 topped out at
// ~42 TFLOPS = 35% of spec; four live chains still only reached ~42 (the
// fragment K=16 path under-saturates Blackwell -- cuBLASLt fp16 hits ~78),
// so we raise to eight to hide more MMA latency.  All chains are reduced
// element-wise at the end (NOT folded via mma_sync, which overwrites its
// destination and would dead-code the extra chains) so the host op count
// stays honest; same lesson as the Metal simdgroup_matrix fix.
//
// Per warp ops = 256 outer * 8 chains * (16*16*16*2) = 16,777,216;
// per thread = 524,288 (= 8 * COOPMAT_WORK_PER_WI).

#include <mma.h>
#include <cuda_fp16.h>

extern "C" __global__ void wmma_fp16(float *out, float A)
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, __half, row_major> a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c0, c1, c2, c3, c4, c5, c6, c7;

    fill_fragment(a, __float2half(A));
    fill_fragment(b, __float2half(A));
    fill_fragment(c0, 0.0f);
    fill_fragment(c1, 0.0f);
    fill_fragment(c2, 0.0f);
    fill_fragment(c3, 0.0f);
    fill_fragment(c4, 0.0f);
    fill_fragment(c5, 0.0f);
    fill_fragment(c6, 0.0f);
    fill_fragment(c7, 0.0f);

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        mma_sync(c0, a, b, c0);
        mma_sync(c1, a, b, c1);
        mma_sync(c2, a, b, c2);
        mma_sync(c3, a, b, c3);
        mma_sync(c4, a, b, c4);
        mma_sync(c5, a, b, c5);
        mma_sync(c6, a, b, c6);
        mma_sync(c7, a, b, c7);
    }

    // Reduce all eight chains element-wise before the store.  mma_sync writes
    // its FIRST argument, so folding via mma_sync(c0, a, b, c1) overwrites c0
    // instead of accumulating into it -- that dead-codes the chains the host
    // still counts and inflates the reported TFLOPS.  Summing the accumulator
    // fragments keeps every chain live.
    #pragma unroll
    for (int t = 0; t < c0.num_elements; t++)
        c0.x[t] += c1.x[t] + c2.x[t] + c3.x[t] + c4.x[t] + c5.x[t] + c6.x[t] + c7.x[t];

    store_matrix_sync(out + blockIdx.x * 16 * 16, c0, 16, mem_row_major);
}
