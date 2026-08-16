// FP16 MAD-chain throughput.  Two variants:
//   compute_hp   -- scalar __half FMA, 2 parallel chains
//   compute_hp2  -- __half2 packed HFMA2 (NVIDIA's 2x FP32 fp16 peak from
//                   sm_53+; one instruction issues two fp16 FMAs/cycle).
//
// compute_hp uses 2 parallel chains (x0, x1 -- rather than the 1 chain in
// the SP/MP/BF16 kernels) because RTX 5060 single-chain scalar __hfma
// measured at 60% of compute_sp -- the scalar HFMA pipe is latency-bound
// at 1 chain.  Confirmed empirically: 1 chain hit 10.6 TFLOPS, 2 chains
// hit 21.1 TFLOPS, matching the half2 packed peak.  Both chains share one
// invariant c, so the ILP is unchanged at 3 live values instead of 4.
//
// Chain shape and why: see the MAD chain block in include/common/common.h.
//
// Op accounting matches compute_hp.cu prior version: 4096 fp16 ops/thread
// either way.  hp scalar = 64 outer * 16 FMAs * 2 chains * 2 ops; hp2 =
// 64 outer * 16 HFMA2 * 4 ops.

#include <cuda_fp16.h>

#define MAD_4(x, c)  x = __hfma(x, x, c); x = __hfma(x, x, c); x = __hfma(x, x, c); x = __hfma(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

extern "C" __global__ void compute_hp(float *out, float A)
{
    __half x0 = __float2half(A);
    __half x1 = __float2half(A + 1.0f);
    __half c  = __float2half((float)threadIdx.x);

    #pragma unroll
    for (int i = 0; i < 64; i++)
    {
        MAD_16(x0, c)
        MAD_16(x1, c)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] =
        __half2float(__hadd(x0, x1));
}

#define MAD2_4(x, c)  x = __hfma2(x, x, c); x = __hfma2(x, x, c); x = __hfma2(x, x, c); x = __hfma2(x, x, c);
#define MAD2_16(x, c) MAD2_4(x, c) MAD2_4(x, c) MAD2_4(x, c) MAD2_4(x, c)

extern "C" __global__ void compute_hp2(float *out, float A)
{
    __half2 x = __float2half2_rn(A);
    __half2 c = __float2half2_rn((float)threadIdx.x);

    #pragma unroll
    for (int i = 0; i < 64; i++)
    {
        MAD2_16(x, c)
    }

    __half2 r = x;
    out[blockIdx.x * blockDim.x + threadIdx.x] =
        __half2float(__low2half(r)) + __half2float(__high2half(r));
}
