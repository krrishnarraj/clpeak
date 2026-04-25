// Single-precision MAD-chain throughput.  Structure mirrors
// src/shaders/compute_sp_v1.comp -- ping-pong x/y feedback so the compiler
// can't hoist invariants and so each FMA depends on the previous, exposing
// ILP across the two parallel accumulator chains.
//
// 128 outer iters * 16 FMAs * 2 ops = 4096 ops per thread (= COMPUTE_FP_WORK_PER_WI).

#define MAD_4(x, y)  x = fmaf(y, x, y); y = fmaf(x, y, x); x = fmaf(y, x, y); y = fmaf(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

extern "C" __global__ void compute_sp(float *out, float A)
{
    float x = A;
    float y = (float)threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, y)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = y;
}
