// WMMA fp64xfp64+fp64 m8n8k4 -- Ampere+ DP tensor-core throughput.
// Native double tensor cores were added on sm_80; consumer Ada/Blackwell
// support the same fragment shape.  Same 4-chain ILP structure as the
// FP16/BF16 kernels.
//
// The m8n8k4 tile is 4x smaller than m16n16k16, so we run 4x more outer
// iterations (1024 vs 256) to keep the per-thread op budget aligned with
// the convention used by the rest of the WMMA suite.
//
// Per warp ops = 1024 outer * 4 chains * (8*8*4*2) = 2,097,152;
// per thread = 65,536 (= 1 * COOPMAT_WORK_PER_WI).

#include <mma.h>

extern "C" __global__ void wmma_fp64(double *out, double A)
{
    using namespace nvcuda::wmma;

    fragment<matrix_a, 8, 8, 4, double, row_major> a;
    fragment<matrix_b, 8, 8, 4, double, col_major> b;
    fragment<accumulator, 8, 8, 4, double> c0, c1, c2, c3;

    fill_fragment(a, A);
    fill_fragment(b, A);
    fill_fragment(c0, 0.0);
    fill_fragment(c1, 0.0);
    fill_fragment(c2, 0.0);
    fill_fragment(c3, 0.0);

    #pragma unroll 1
    for (int i = 0; i < 1024; i++)
    {
        mma_sync(c0, a, b, c0);
        mma_sync(c1, a, b, c1);
        mma_sync(c2, a, b, c2);
        mma_sync(c3, a, b, c3);
    }

    mma_sync(c0, a, b, c1);
    mma_sync(c2, a, b, c3);
    mma_sync(c0, a, b, c2);

    store_matrix_sync(out + blockIdx.x * 8 * 8, c0, 8, mem_row_major);
}
