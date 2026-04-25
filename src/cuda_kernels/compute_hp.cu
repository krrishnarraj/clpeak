// FP16 MAD-chain throughput.  Two variants:
//   compute_hp   -- scalar __half FMA (peaks at FP32 rate on shader cores
//                   that lack a separate scalar HFMA pipe).
//   compute_hp2  -- __half2 packed HFMA2 (NVIDIA's 2x FP32 fp16 peak; one
//                   instruction issues two fp16 FMAs per cycle from sm_53+).
//
// Op accounting -- both variants emit 4096 fp16 ops per thread to match
// COMPUTE_FP_WORK_PER_WI.  Scalar = 128 outer * 16 FMAs * 2 ops.  Packed =
// 64  outer * 16 HFMA2 * 4 ops (each HFMA2 == 2 fp16 FMAs).

#include <cuda_fp16.h>

#define MAD_4(x, y)  x = __hfma(y, x, y); y = __hfma(x, y, x); x = __hfma(y, x, y); y = __hfma(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

extern "C" __global__ void compute_hp(float *out, float A)
{
    __half x = __float2half(A);
    __half y = __float2half((float)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, y)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = __half2float(__hadd(x, y));
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
    out[blockIdx.x * blockDim.x + threadIdx.x] = __half2float(__low2half(r)) + __half2float(__high2half(r));
}
