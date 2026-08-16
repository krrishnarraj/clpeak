#include <hip/hip_runtime.h>

// Double-precision MAD-chain throughput across the HIP SDK's native vector
// widths (double, double2, double4). Same constant-work-per-thread scheme as
// compute_sp.hip: the loop trip count is divided by the vector width so every
// variant does 512 flops/thread and the numbers are directly comparable. HIP
// vector types have no operators, so the chain is built component-wise via fma.
//
// Chain shape and why: see the MAD chain block in include/common/common.h.

// ---- scalar: double ------------------------------------------------------
#define MAD_4(x, c)  x = fma(x, x, c); x = fma(x, x, c); x = fma(x, x, c); x = fma(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

extern "C" __global__ void compute_dp(double *out, double A)
{
    double x = A;
    double c = (double)threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < 16; i++)
    {
        MAD_16(x, c)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- double2 -------------------------------------------------------------
#define DFMA2(a, b, c) make_double2(fma((a).x, (b).x, (c).x), fma((a).y, (b).y, (c).y))
#define DMAD2_4(x, c)  x = DFMA2(x, x, c); x = DFMA2(x, x, c); x = DFMA2(x, x, c); x = DFMA2(x, x, c);
#define DMAD2_16(x, c) DMAD2_4(x, c) DMAD2_4(x, c) DMAD2_4(x, c) DMAD2_4(x, c)

extern "C" __global__ void compute_dp_v2(double *out, double A)
{
    double2 x = make_double2(A, A + 1.0);
    double2 c = make_double2((double)threadIdx.x, (double)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 8; i++)
    {
        DMAD2_16(x, c)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x.x + x.y;
}

// ---- double4 -------------------------------------------------------------
#define DFMA4(a, b, c) make_double4(fma((a).x, (b).x, (c).x), fma((a).y, (b).y, (c).y), \
                                    fma((a).z, (b).z, (c).z), fma((a).w, (b).w, (c).w))
#define DMAD4_4(x, c)  x = DFMA4(x, x, c); x = DFMA4(x, x, c); x = DFMA4(x, x, c); x = DFMA4(x, x, c);
#define DMAD4_16(x, c) DMAD4_4(x, c) DMAD4_4(x, c) DMAD4_4(x, c) DMAD4_4(x, c)

extern "C" __global__ void compute_dp_v4(double *out, double A)
{
    double4 x = make_double4(A, A + 1.0, A + 2.0, A + 3.0);
    double4 c = make_double4((double)threadIdx.x, (double)threadIdx.x,
                             (double)threadIdx.x, (double)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 4; i++)
    {
        DMAD4_16(x, c)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x.x + x.y + x.z + x.w;
}
