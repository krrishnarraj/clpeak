// WMMA bf16xbf16+fp32 m16n16k16 -- Ampere+ tensor-core throughput test.
// Identical shape to wmma_fp16; only the input fragment type differs.

#include <mma.h>
#include <cuda_bf16.h>

extern "C" __global__ void wmma_bf16(float *out, float A)
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c;

    fill_fragment(a, __float2bfloat16(A));
    fill_fragment(b, __float2bfloat16(A));
    fill_fragment(c, 0.0f);

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        mma_sync(c, a, b, c);
    }

    store_matrix_sync(out + blockIdx.x * 16 * 16, c, 16, mem_row_major);
}
