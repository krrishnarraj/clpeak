// Double-precision MAD-chain throughput.  Identical shape to compute_sp.cu
// with double / fma.  RTX consumer cards have a 1:32 fp64 ratio; expect
// numbers ~3% of the SP peak.
//
// Chain shape and why: see the MAD chain block in include/common/common.h.

#define MAD_4(x, c)  x = fma(x, x, c); x = fma(x, x, c); x = fma(x, x, c); x = fma(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

extern "C" __global__ void compute_dp(double *out, double A)
{
    double x = A;
    double c = (double)threadIdx.x;

    #pragma unroll
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, c)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
