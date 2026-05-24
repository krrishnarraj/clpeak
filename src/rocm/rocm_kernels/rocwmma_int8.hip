#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdint>

#define ROCWMMA_ITERS 256

extern "C" __global__ void rocwmma_int8(int *out)
{
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;

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
