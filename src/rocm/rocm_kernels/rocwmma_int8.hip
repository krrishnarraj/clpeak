#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdint>

#define ROCWMMA_ITERS 256

extern "C" __global__ void rocwmma_int8(int *out)
{
    constexpr int M = 16;
    constexpr int N = 16;
    // int8 needs BlockK >= the arch minimum: CDNA3 (gfx942/950) requires K=32,
    // while CDNA1/2 and RDNA3+ accept K>=16.  K=32 satisfies all of them, so the
    // one kernel builds across every rocWMMA-supported arch.
    constexpr int K = 32;

    using FragA = rocwmma::fragment<rocwmma::matrix_a, M, N, K,
                                    int8_t, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, M, N, K,
                                    int8_t, rocwmma::col_major>;
    using Accum = rocwmma::fragment<rocwmma::accumulator, M, N, K,
                                    int32_t>;

    FragA a;
    FragB b;
    Accum c;

    rocwmma::fill_fragment(a, int8_t(1));
    rocwmma::fill_fragment(b, int8_t(1));
    rocwmma::fill_fragment(c, int32_t(0));

    #pragma unroll 1
    for (int i = 0; i < ROCWMMA_ITERS; i++)
    {
        rocwmma::mma_sync(c, a, b, c);
    }

    rocwmma::store_matrix_sync(out + blockIdx.x * M * N, c, N,
                               rocwmma::layout_t::mem_row_major);
}
