// WMMA fp16xfp16+fp32 m16n16k16 -- the canonical Volta+ tensor-core
// throughput test via the portable nvcuda::wmma fragment API.  Mirrors
// src/shaders/coopmat_fp16.comp at the same tile shape and iteration count.
//
// Four independent accumulator chains (c0..c3) to hide the tensor MMA
// latency.  Measured on RTX 5060 (sm_120): this fp32-accumulate path tops out
// at ~42 TFLOPS and does NOT move with more chains or a native mma.sync tile
// (both 42) -- on GeForce fp16xfp16 with an fp32 accumulator runs at HALF
// rate.  The full-rate ~78 (cuBLASLt) number needs an fp16 accumulator; see
// wmma_fp16_f16.cu.  This kernel is the honest fp32-accumulate measurement.
//
// All four chains are reduced element-wise at the end (NOT folded via
// mma_sync, which overwrites its destination and would dead-code the extra
// chains -- that inflated this number 4x before the fix; same lesson as the
// Metal simdgroup_matrix fix).
//
// Per warp ops = 256 outer * 4 chains * (16*16*16*2) = 8,388,608;
// per thread = 262,144 (= 4 * COOPMAT_WORK_PER_WI).

#include <mma.h>
#include <cuda_fp16.h>

extern "C" __global__ void wmma_fp16(float *out, float A)
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, __half, row_major> a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c0, c1, c2, c3;

    fill_fragment(a, __float2half(A));
    fill_fragment(b, __float2half(A));
    fill_fragment(c0, 0.0f);
    fill_fragment(c1, 0.0f);
    fill_fragment(c2, 0.0f);
    fill_fragment(c3, 0.0f);

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        mma_sync(c0, a, b, c0);
        mma_sync(c1, a, b, c1);
        mma_sync(c2, a, b, c2);
        mma_sync(c3, a, b, c3);
    }

    // Reduce all four chains element-wise before the store.  mma_sync writes
    // its FIRST argument, so folding via mma_sync(c0, a, b, c1) overwrites c0
    // instead of accumulating into it -- that dead-codes the chains the host
    // still counts and inflates the reported TFLOPS.  Summing the accumulator
    // fragments keeps every chain live.
    #pragma unroll
    for (int t = 0; t < c0.num_elements; t++)
        c0.x[t] += c1.x[t] + c2.x[t] + c3.x[t];

    store_matrix_sync(out + blockIdx.x * 16 * 16, c0, 16, mem_row_major);
}
