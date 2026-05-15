// BF16 x BF16 + FP32 shader-core MAD path.  Same shape as compute_mp.cu but
// with __nv_bfloat16 cast instead of __half.  Mirrors compute_bf16_v1.comp:
// deeper inner FMA chain (MAD_128 instead of MAD_16) because bf16<->fp32
// casts are emulated on some drivers and we need to amortise their cost.
//
// 16 outer iters * 128 FMAs * 2 ops = 4096 ops per thread (= COMPUTE_FP_WORK_PER_WI).

#include <cuda_bf16.h>

#define MAD_4(x, y)   x = fmaf(y, x, y); y = fmaf(x, y, x); x = fmaf(y, x, y); y = fmaf(x, y, x);
#define MAD_16(x, y)  MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)
#define MAD_64(x, y)  MAD_16(x, y) MAD_16(x, y) MAD_16(x, y) MAD_16(x, y)
#define MAD_128(x, y) MAD_64(x, y) MAD_64(x, y)

extern "C" __global__ void compute_bf16(float *out, float A)
{
    float x = __bfloat162float(__float2bfloat16(A));
    float y = __bfloat162float(__float2bfloat16((float)threadIdx.x));

    #pragma unroll 1
    for (int i = 0; i < 16; i++)
    {
        MAD_128(x, y)
        x = __bfloat162float(__float2bfloat16(x));
        y = __bfloat162float(__float2bfloat16(y));
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}
