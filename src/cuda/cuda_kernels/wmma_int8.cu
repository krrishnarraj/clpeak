// WMMA int8xint8+int32 m16n16k16 -- Turing+ tensor-core INT8 throughput.
// Same 8-chain ILP structure as wmma_fp16.
//
// Note: NVIDIA tensor cores do INT8 most efficiently at K=32 (the natural
// native tile width); m16n16k16 here halves the K dim, so this kernel
// captures the WMMA-fragment INT8 path.  The separate K=32 mma.sync path
// (wmma_int8_k32) reaches ~2x this on consumer Blackwell -- see that file.
//
// Per warp ops = 256 outer * 8 chains * (16*16*16*2) = 16,777,216;
// per thread = 524,288 (= 8 * COOPMAT_WORK_PER_WI).

#include <mma.h>

extern "C" __global__ void wmma_int8(int *out, int A)
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, signed char, row_major> a;
    fragment<matrix_b, 16, 16, 16, signed char, col_major> b;
    fragment<accumulator, 16, 16, 16, int> c0, c1, c2, c3, c4, c5, c6, c7;

    fill_fragment(a, (signed char)(A & 0x7f));
    fill_fragment(b, (signed char)(A & 0x7f));
    fill_fragment(c0, 0);
    fill_fragment(c1, 0);
    fill_fragment(c2, 0);
    fill_fragment(c3, 0);
    fill_fragment(c4, 0);
    fill_fragment(c5, 0);
    fill_fragment(c6, 0);
    fill_fragment(c7, 0);

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

    // Sum all eight chains element-wise; see wmma_fp16.cu for why an mma_sync
    // fold would dead-code chains and inflate the number.
    #pragma unroll
    for (int t = 0; t < c0.num_elements; t++)
        c0.x[t] += c1.x[t] + c2.x[t] + c3.x[t] + c4.x[t] + c5.x[t] + c6.x[t] + c7.x[t];

    store_matrix_sync(out + blockIdx.x * 16 * 16, c0, 16, mem_row_major);
}
