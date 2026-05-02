// WMMA tf32xtf32+fp32 m16n16k8 -- Ampere+ tensor-core TF32 throughput.
// Mirrors wmma_fp16.cu's 4-chain ILP structure; only the fragment dtype
// (precision::tf32) and K dimension (8 instead of 16) change.
//
// Per warp ops = 256 outer * 4 chains * (16*16*8*2) = 4,194,304;
// per thread = 131,072 (= 2 * COOPMAT_WORK_PER_WI).

#include <mma.h>

extern "C" __global__ void wmma_tf32(float *out, float A)
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a;
    fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b;
    fragment<accumulator, 16, 16, 8, float> c0, c1, c2, c3;

    fill_fragment(a, __float_to_tf32(A));
    fill_fragment(b, __float_to_tf32(A));
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

    mma_sync(c0, a, b, c1);
    mma_sync(c2, a, b, c3);
    mma_sync(c0, a, b, c2);

    store_matrix_sync(out + blockIdx.x * 16 * 16, c0, 16, mem_row_major);
}
