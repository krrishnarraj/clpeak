// WMMA fp16xfp16+fp32 m16n16k16 -- the canonical Volta+ tensor-core
// throughput test.  Mirrors src/shaders/coopmat_fp16.comp at the same
// tile shape and iteration count.
//
// One warp (32 threads) per block.  Each block holds one m16n16 output
// tile (256 fp32 elements).  256 outer mma_sync iters per warp:
//   16*16*16*2 ops/mma * 256 iters / 32 threads = 65536 ops/thread
// matching COOPMAT_WORK_PER_WI.

#include <mma.h>
#include <cuda_fp16.h>

extern "C" __global__ void wmma_fp16(float *out, float A)
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, 16, 16, 16, __half, row_major> a;
    fragment<matrix_b, 16, 16, 16, __half, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c;

    fill_fragment(a, __float2half(A));
    fill_fragment(b, __float2half(A));
    fill_fragment(c, 0.0f);

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        mma_sync(c, a, b, c);
    }

    store_matrix_sync(out + blockIdx.x * 16 * 16, c, 16, mem_row_major);
}
