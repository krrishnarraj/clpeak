#include <hip/hip_runtime.h>
#if __has_include(<hip/hip_bf16.h>)
#include <hip/hip_bf16.h>
#else
#include <hip/hip_bfloat16.h>
#endif

#define MAD_4(x, y)   x = fmaf(y, x, y); y = fmaf(x, y, x); x = fmaf(y, x, y); y = fmaf(x, y, x);
#define MAD_16(x, y)  MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)
#define MAD_64(x, y)  MAD_16(x, y) MAD_16(x, y) MAD_16(x, y) MAD_16(x, y)
#define MAD_128(x, y) MAD_64(x, y) MAD_64(x, y)

extern "C" __global__ void compute_bf16(float *out, float A)
{
    float x0 = (float)hip_bfloat16(A);
    float y0 = (float)hip_bfloat16((float)threadIdx.x);
    float x1 = (float)hip_bfloat16(A + 1.0f);
    float y1 = (float)hip_bfloat16((float)threadIdx.x + 7.0f);

    #pragma unroll 1
    for (int i = 0; i < 8; i++)
    {
        MAD_128(x0, y0)
        MAD_128(x1, y1)
        x0 = (float)hip_bfloat16(x0);
        y0 = (float)hip_bfloat16(y0);
        x1 = (float)hip_bfloat16(x1);
        y1 = (float)hip_bfloat16(y1);
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = (x0 + y0) + (x1 + y1);
}
