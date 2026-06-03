#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define MAD_4(x, y)  x = fmaf(y, x, y); y = fmaf(x, y, x); x = fmaf(y, x, y); y = fmaf(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

extern "C" __global__ void compute_mp(float *out, float A)
{
    float x = __half2float(__float2half(A));
    float y = __half2float(__float2half((float)threadIdx.x));

    #pragma unroll 1
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, y)
        x = __half2float(__float2half(x));
        y = __half2float(__float2half(y));
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}
