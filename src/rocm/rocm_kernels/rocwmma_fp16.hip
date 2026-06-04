#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#define ROCWMMA_ITERS 256

extern "C" __global__ void rocwmma_fp16(float *out)
{
    constexpr int M = 16;
    constexpr int N = 16;
    constexpr int K = 16;

    using FragA = rocwmma::fragment<rocwmma::matrix_a, M, N, K,
                                    rocwmma::float16_t, rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b, M, N, K,
                                    rocwmma::float16_t, rocwmma::col_major>;
    using Accum = rocwmma::fragment<rocwmma::accumulator, M, N, K,
                                    rocwmma::float32_t>;

    FragA a;
    FragB b;
    Accum c;

    rocwmma::fill_fragment(a, rocwmma::float16_t(1.0f));
    rocwmma::fill_fragment(b, rocwmma::float16_t(1.0f));
    rocwmma::fill_fragment(c, 0.0f);

    #pragma unroll 1
    for (int i = 0; i < ROCWMMA_ITERS; i++)
    {
        rocwmma::mma_sync(c, a, b, c);
    }

    rocwmma::store_matrix_sync(out + blockIdx.x * M * N, c, N,
                               rocwmma::layout_t::mem_row_major);
}
