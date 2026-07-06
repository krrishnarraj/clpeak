// WMMA bf16xbf16+fp32 m16n16k16 -- Ampere+ tensor-core throughput via the
// nvcuda::wmma fragment API.  Same 4-chain structure as wmma_fp16; only the
// input fragment type differs.  Like wmma_fp16 this fragment K=16 path
// under-saturates consumer Blackwell (~42 TFLOPS, flat vs chain count); the
// native mma.sync path (wmma_bf16_mma, m16n8k16) chases peak.
//
// Per warp ops = 256 outer * 4 chains * (16*16*16*2) = 8,388,608;
// per thread = 262,144 (= 4 * COOPMAT_WORK_PER_WI).

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

    // Sum all four chains element-wise; see wmma_fp16.cu for why an mma_sync
    // fold would dead-code chains and inflate the number.
    #pragma unroll
    for (int t = 0; t < c0.num_elements; t++)
        c0.x[t] += c1.x[t] + c2.x[t] + c3.x[t];

    store_matrix_sync(out + blockIdx.x * 16 * 16, c0, 16, mem_row_major);
}
