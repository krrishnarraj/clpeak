// 32-bit integer IMAD-chain throughput.  Mirrors compute_sp.cu's two-chain
// ping-pong structure so the compiler can't hoist invariants and each MAD
// depends on the previous, exposing ILP across the (x, y) accumulators.
//
// Distinct hardware path from compute_int8_dp (__dp4a) and the int4-packed
// emulation: this is the shader-core IMAD pipe used by ordinary `int`
// arithmetic.  Reported in GOPS.
//
// 128 outer iters * 16 MADs * 2 ops = 4096 ops per thread (= COMPUTE_FP_WORK_PER_WI).

#define MAD_4(x, y)  x = y * x + y; y = x * y + x; x = y * x + y; y = x * y + x;
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

extern "C" __global__ void compute_int32(int *out, int A)
{
    int x = A;
    int y = (int)threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, y)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = y;
}
