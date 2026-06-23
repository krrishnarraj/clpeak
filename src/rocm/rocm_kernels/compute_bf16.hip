#include <hip/hip_runtime.h>
// ROCm 6.4 dropped the unprefixed `hip_bfloat16` alias in favor of
// `__hip_bfloat16` (and moved it from hip_bfloat16.h to hip_bf16.h).  Alias to
// whichever the installed headers expose so the kernel builds on old and new.
#if __has_include(<hip/hip_bf16.h>)
#include <hip/hip_bf16.h>
using clpeak_bf16 = __hip_bfloat16;
#else
#include <hip/hip_bfloat16.h>
using clpeak_bf16 = hip_bfloat16;
#endif

#define MAD_4(x, y)   x = fmaf(y, x, y); y = fmaf(x, y, x); x = fmaf(y, x, y); y = fmaf(x, y, x);
#define MAD_16(x, y)  MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)
#define MAD_64(x, y)  MAD_16(x, y) MAD_16(x, y) MAD_16(x, y) MAD_16(x, y)
#define MAD_128(x, y) MAD_64(x, y) MAD_64(x, y)

extern "C" __global__ void compute_bf16(float *out, float A)
{
    float x = (float)clpeak_bf16(A);
    float y = (float)clpeak_bf16((float)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 16; i++)
    {
        MAD_128(x, y)
        x = (float)clpeak_bf16(x);
        y = (float)clpeak_bf16(y);
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}
