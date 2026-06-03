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
    float x = (float)hip_bfloat16(A);
    float y = (float)hip_bfloat16((float)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 16; i++)
    {
        MAD_128(x, y)
        x = (float)hip_bfloat16(x);
        y = (float)hip_bfloat16(y);
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}
