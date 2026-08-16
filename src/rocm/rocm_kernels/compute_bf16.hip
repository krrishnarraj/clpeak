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

// Chain shape and why: see the MAD chain block in include/common/common.h.
// c is seeded through bf16 once and never updated, so it stays
// bf16-representable without a per-iteration round trip.
#define MAD_4(x, c)   x = fmaf(x, x, c); x = fmaf(x, x, c); x = fmaf(x, x, c); x = fmaf(x, x, c);
#define MAD_16(x, c)  MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)
#define MAD_64(x, c)  MAD_16(x, c) MAD_16(x, c) MAD_16(x, c) MAD_16(x, c)
#define MAD_128(x, c) MAD_64(x, c) MAD_64(x, c)

extern "C" __global__ void compute_bf16(float *out, float A)
{
    float x = (float)clpeak_bf16(A);
    float c = (float)clpeak_bf16((float)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 16; i++)
    {
        MAD_128(x, c)
        x = (float)clpeak_bf16(x);
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
