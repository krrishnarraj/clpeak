// WMMA bf16xbf16+fp32 m16n16k16 -- Ampere+ tensor-core throughput test.
// Same 4-chain ILP structure as wmma_fp16; only the input fragment type
// differs.

#include <mma.h>
#include <cuda_bf16.h>

extern "C" __global__ void wmma_bf16(float *out, float A)
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c0, c1, c2, c3;

    fill_fragment(a, __float2bfloat16(A));
    fill_fragment(b, __float2bfloat16(A));
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
