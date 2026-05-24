#include <hip/hip_runtime.h>

#define MAD_4(x, y)  x = fma(y, x, y); y = fma(x, y, x); x = fma(y, x, y); y = fma(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

extern "C" __global__ void compute_dp(double *out, double A)
{
    double x = A;
    double y = (double)threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < 16; i++)
    {
        MAD_16(x, y)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = y;
}
