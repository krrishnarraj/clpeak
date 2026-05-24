#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define MAD_4(x, y)  x = __hfma(y, x, y); y = __hfma(x, y, x); x = __hfma(y, x, y); y = __hfma(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

extern "C" __global__ void compute_hp(float *out, float A)
{
    __half x0 = __float2half(A);
    __half y0 = __float2half((float)threadIdx.x);
    __half x1 = __float2half(A + 1.0f);
    __half y1 = __float2half((float)threadIdx.x + 7.0f);

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        MAD_16(x0, y0)
        MAD_16(x1, y1)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] =
        __half2float(__hadd(__hadd(x0, y0), __hadd(x1, y1)));
}

#define MAD2_4(x, y)  x = __hfma2(y, x, y); y = __hfma2(x, y, x); x = __hfma2(y, x, y); y = __hfma2(x, y, x);
#define MAD2_16(x, y) MAD2_4(x, y) MAD2_4(x, y) MAD2_4(x, y) MAD2_4(x, y)

extern "C" __global__ void compute_hp2(float *out, float A)
{
    __half2 x = __float2half2_rn(A);
    __half2 y = __float2half2_rn((float)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        MAD2_16(x, y)
    }

    __half2 r = __hadd2(x, y);
    out[blockIdx.x * blockDim.x + threadIdx.x] =
        __half2float(__low2half(r)) + __half2float(__high2half(r));
}
