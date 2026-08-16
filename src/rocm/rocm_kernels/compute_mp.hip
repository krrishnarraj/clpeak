#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Chain shape and why: see the MAD chain block in include/common/common.h.
// c is seeded through __half once and never updated, so it stays
// fp16-representable without a per-iteration round trip.

#define MAD_4(x, c)  x = fmaf(x, x, c); x = fmaf(x, x, c); x = fmaf(x, x, c); x = fmaf(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

extern "C" __global__ void compute_mp(float *out, float A)
{
    float x = __half2float(__float2half(A));
    float c = __half2float(__float2half((float)threadIdx.x));

    #pragma unroll 1
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, c)
        x = __half2float(__float2half(x));
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
