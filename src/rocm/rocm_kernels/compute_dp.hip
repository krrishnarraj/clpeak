#include <hip/hip_runtime.h>

// Double-precision MAD-chain throughput across the HIP SDK's native vector
// widths (double, double2, double4). Same constant-work-per-thread scheme as
// compute_sp.hip: the loop trip count is divided by the vector width so every
// variant does 512 flops/thread and the numbers are directly comparable. HIP
// vector types have no operators, so the chain is built component-wise via fma.

// ---- scalar: double ------------------------------------------------------
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

// ---- double2 -------------------------------------------------------------
#define DFMA2(a, b, c) make_double2(fma((a).x, (b).x, (c).x), fma((a).y, (b).y, (c).y))
#define DMAD2_4(x, y)  x = DFMA2(y, x, y); y = DFMA2(x, y, x); x = DFMA2(y, x, y); y = DFMA2(x, y, x);
#define DMAD2_16(x, y) DMAD2_4(x, y) DMAD2_4(x, y) DMAD2_4(x, y) DMAD2_4(x, y)

extern "C" __global__ void compute_dp_v2(double *out, double A)
{
    double2 x = make_double2(A, A + 1.0);
    double2 y = make_double2((double)threadIdx.x, (double)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 8; i++)
    {
        DMAD2_16(x, y)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = y.x + y.y;
}

// ---- double4 -------------------------------------------------------------
#define DFMA4(a, b, c) make_double4(fma((a).x, (b).x, (c).x), fma((a).y, (b).y, (c).y), \
                                    fma((a).z, (b).z, (c).z), fma((a).w, (b).w, (c).w))
#define DMAD4_4(x, y)  x = DFMA4(y, x, y); y = DFMA4(x, y, x); x = DFMA4(y, x, y); y = DFMA4(x, y, x);
#define DMAD4_16(x, y) DMAD4_4(x, y) DMAD4_4(x, y) DMAD4_4(x, y) DMAD4_4(x, y)

extern "C" __global__ void compute_dp_v4(double *out, double A)
{
    double4 x = make_double4(A, A + 1.0, A + 2.0, A + 3.0);
    double4 y = make_double4((double)threadIdx.x, (double)threadIdx.x,
                             (double)threadIdx.x, (double)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 4; i++)
    {
        DMAD4_16(x, y)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = y.x + y.y + y.z + y.w;
}
