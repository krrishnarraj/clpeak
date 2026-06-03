#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define MAD_4(x, y)  x = fmaf(y, x, y); y = fmaf(x, y, x); x = fmaf(y, x, y); y = fmaf(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

extern "C" __global__ void compute_mp(float *out, float A)
{
    float x0 = __half2float(__float2half(A));
    float y0 = __half2float(__float2half((float)threadIdx.x));
    float x1 = __half2float(__float2half(A + 1.0f));
    float y1 = __half2float(__float2half((float)threadIdx.x + 7.0f));

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        MAD_16(x0, y0)
        MAD_16(x1, y1)
        x0 = __half2float(__float2half(x0));
        y0 = __half2float(__float2half(y0));
        x1 = __half2float(__float2half(x1));
        y1 = __half2float(__float2half(y1));
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = (x0 + y0) + (x1 + y1);
}
