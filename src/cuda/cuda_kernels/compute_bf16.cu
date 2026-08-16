// BF16 x BF16 + FP32 shader-core MAD path.  Same shape as compute_mp.cu but
// with __nv_bfloat16 cast instead of __half.  Mirrors compute_bf16_v1.comp:
// deeper inner FMA chain (MAD_128 instead of MAD_16) because bf16<->fp32
// casts are emulated on some drivers and we need to amortise their cost.
//
// 16 outer iters * 128 FMAs * 2 ops = 4096 ops per thread (= COMPUTE_FP_WORK_PER_WI).
//
// Chain shape and why: see the MAD chain block in include/common/common.h.
// c is seeded through bf16 once and never updated, so it stays
// bf16-representable without a per-iteration round trip.

#include <cuda_bf16.h>

#define MAD_4(x, c)   x = fmaf(x, x, c); x = fmaf(x, x, c); x = fmaf(x, x, c); x = fmaf(x, x, c);
#define MAD_16(x, c)  MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)
#define MAD_64(x, c)  MAD_16(x, c) MAD_16(x, c) MAD_16(x, c) MAD_16(x, c)
#define MAD_128(x, c) MAD_64(x, c) MAD_64(x, c)

extern "C" __global__ void compute_bf16(float *out, float A)
{
    float x = __bfloat162float(__float2bfloat16(A));
    float c = __bfloat162float(__float2bfloat16((float)threadIdx.x));

    #pragma unroll
    for (int i = 0; i < 16; i++)
    {
        MAD_128(x, c)
        x = __bfloat162float(__float2bfloat16(x));
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
