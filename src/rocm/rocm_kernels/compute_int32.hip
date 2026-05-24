#include <hip/hip_runtime.h>

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
