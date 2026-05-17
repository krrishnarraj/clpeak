// Double-precision MAD-chain throughput.  Identical shape to compute_sp.cu
// with double / fma.  RTX consumer cards have a 1:32 fp64 ratio; expect
// numbers ~3% of the SP peak.

#define MAD_4(x, y)  x = fma(y, x, y); y = fma(x, y, x); x = fma(y, x, y); y = fma(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

extern "C" __global__ void compute_dp(double *out, double A)
{
    double x = A;
    double y = (double)threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, y)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = y;
}
