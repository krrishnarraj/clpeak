// WMMA fp16xfp16+fp32 m16n16k16 -- the canonical Volta+ tensor-core
// throughput test.  Mirrors src/shaders/coopmat_fp16.comp at the same
// tile shape and iteration count.
//
// Four independent accumulator chains (c0..c3) so the tensor unit's
// multi-cycle pipeline can issue back-to-back MMAs instead of stalling
// on the c -> c dependency every iter.  Single-chain on RTX 5060 topped
// out at ~42 TFLOPS = 35% of the spec peak; same fix we used for Metal
// simdgroup_matrix and the Vulkan int8_dp4 variant.
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

    // Fold so every accumulator contributes to the stored tile -- prevents
    // the compiler from dropping any chain it sees as dead.  3 extra MMAs
    // is ~0.3% noise on the measured TFLOPS.
    mma_sync(c0, a, b, c1);
    mma_sync(c2, a, b, c3);
    mma_sync(c0, a, b, c2);

    store_matrix_sync(out + blockIdx.x * 16 * 16, c0, 16, mem_row_major);
}
