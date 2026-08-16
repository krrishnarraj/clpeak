// 32-bit integer IMAD-chain throughput.  Mirrors compute_sp.cu's chain shape
// (x = x*x + c) so the compiler can't hoist invariants and each MAD depends
// on the previous.
//
// Chain shape and why: see the MAD chain block in include/common/common.h.
//
// Distinct hardware path from __dp4a: this is the shader-core IMAD pipe used by ordinary `int`
// arithmetic.  Reported in GOPS.
//
// 128 outer iters * 16 MADs * 2 ops = 4096 ops per thread (= COMPUTE_FP_WORK_PER_WI).

#define MAD_4(x, c)  x = x * x + c; x = x * x + c; x = x * x + c; x = x * x + c;
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

extern "C" __global__ void compute_int32(int *out, int A)
{
    int x = A;
    int c = (int)threadIdx.x;

    #pragma unroll
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, c)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
