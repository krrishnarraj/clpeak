// Single-precision MAD-chain throughput.  Structure mirrors
// src/shaders/compute_sp_v1.comp -- x = x*x + c, so each FMA depends on the
// previous and the compiler cannot hoist anything out of the loop.
//
// 128 outer iters * 16 FMAs * 2 ops = 4096 ops per thread (= COMPUTE_FP_WORK_PER_WI).
//
// Chain shape and why: see the MAD chain block in include/common/common.h.

#define MAD_4(x, c)  x = fmaf(x, x, c); x = fmaf(x, x, c); x = fmaf(x, x, c); x = fmaf(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

extern "C" __global__ void compute_sp(float *out, float A)
{
    float x = A;
    float c = (float)threadIdx.x;

    #pragma unroll
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, c)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
