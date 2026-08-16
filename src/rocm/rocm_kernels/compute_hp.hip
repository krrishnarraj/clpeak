#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Two parallel chains (x0, x1) sharing one invariant c -- the scalar __hfma
// pipe is latency-bound at a single chain.
//
// Chain shape and why: see the MAD chain block in include/common/common.h.

#define MAD_4(x, c)  x = __hfma(x, x, c); x = __hfma(x, x, c); x = __hfma(x, x, c); x = __hfma(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

extern "C" __global__ void compute_hp(float *out, float A)
{
    __half x0 = __float2half(A);
    __half x1 = __float2half(A + 1.0f);
    __half c  = __float2half((float)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        MAD_16(x0, c)
        MAD_16(x1, c)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] =
        __half2float(__hadd(x0, x1));
}

#define MAD2_4(x, c)  x = __hfma2(x, x, c); x = __hfma2(x, x, c); x = __hfma2(x, x, c); x = __hfma2(x, x, c);
#define MAD2_16(x, c) MAD2_4(x, c) MAD2_4(x, c) MAD2_4(x, c) MAD2_4(x, c)

extern "C" __global__ void compute_hp2(float *out, float A)
{
    __half2 x = __float2half2_rn(A);
    __half2 c = __float2half2_rn((float)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        MAD2_16(x, c)
    }

    __half2 r = x;
    out[blockIdx.x * blockDim.x + threadIdx.x] =
        __half2float(__low2half(r)) + __half2float(__high2half(r));
}
