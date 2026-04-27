// WMMA int8xint8+int32 m16n16k16 -- Turing+ tensor-core INT8 throughput.
// Same 4-chain ILP structure as wmma_fp16.
//
// Note: NVIDIA tensor cores do INT8 most efficiently at K=32 (the natural
// native tile width); m16n16k16 here halves the K dim, so this kernel
// captures the WMMA-fragment INT8 path but a separate K=32 path via
// mma.sync inline PTX is the way to reach the device's true INT8 peak.

#include <mma.h>

extern "C" __global__ void wmma_int8(int *out, int A)
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, signed char, row_major> a;
    fragment<matrix_b, 16, 16, 16, signed char, col_major> b;
    fragment<accumulator, 16, 16, 16, int> c0, c1, c2, c3;

    fill_fragment(a, (signed char)(A & 0x7f));
    fill_fragment(b, (signed char)(A & 0x7f));
    fill_fragment(c0, 0);
    fill_fragment(c1, 0);
    fill_fragment(c2, 0);
    fill_fragment(c3, 0);

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
