// WMMA int8xint8+int32 m16n16k16 -- Turing+ tensor-core INT8 throughput.
// Output element type is int32; scaled to GIOPS at the host side.

#include <mma.h>

extern "C" __global__ void wmma_int8(int *out, int A)
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, signed char, row_major> a;
    fragment<matrix_b, 16, 16, 16, signed char, col_major> b;
    fragment<accumulator, 16, 16, 16, int> c;

    fill_fragment(a, (signed char)(A & 0x7f));
    fill_fragment(b, (signed char)(A & 0x7f));
    fill_fragment(c, 0);

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        mma_sync(c, a, b, c);
    }

    store_matrix_sync(out + blockIdx.x * 16 * 16, c, 16, mem_row_major);
}
