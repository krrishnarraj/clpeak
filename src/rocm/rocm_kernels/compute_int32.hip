#include <hip/hip_runtime.h>

// 32-bit integer IMAD-chain throughput across the HIP SDK's native vector
// widths (int, int2, int4). Same constant-work-per-thread scheme as
// compute_sp.hip: the loop trip count is divided by the vector width so every
// variant does 4096 ops/thread and the numbers are directly comparable. HIP
// vector types have no operators, so the chain is built component-wise.

// ---- scalar: int ---------------------------------------------------------
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

// ---- int2 ----------------------------------------------------------------
#define IMAD2(a, b, c) make_int2((a).x * (b).x + (c).x, (a).y * (b).y + (c).y)
#define IMAD2_4(x, y)  x = IMAD2(y, x, y); y = IMAD2(x, y, x); x = IMAD2(y, x, y); y = IMAD2(x, y, x);
#define IMAD2_16(x, y) IMAD2_4(x, y) IMAD2_4(x, y) IMAD2_4(x, y) IMAD2_4(x, y)

extern "C" __global__ void compute_int32_v2(int *out, int A)
{
    int2 x = make_int2(A, A + 1);
    int2 y = make_int2((int)threadIdx.x, (int)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        IMAD2_16(x, y)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = y.x + y.y;
}

// ---- int4 ----------------------------------------------------------------
#define IMAD4(a, b, c) make_int4((a).x * (b).x + (c).x, (a).y * (b).y + (c).y, \
                                 (a).z * (b).z + (c).z, (a).w * (b).w + (c).w)
#define IMAD4_4(x, y)  x = IMAD4(y, x, y); y = IMAD4(x, y, x); x = IMAD4(y, x, y); y = IMAD4(x, y, x);
#define IMAD4_16(x, y) IMAD4_4(x, y) IMAD4_4(x, y) IMAD4_4(x, y) IMAD4_4(x, y)

extern "C" __global__ void compute_int32_v4(int *out, int A)
{
    int4 x = make_int4(A, A + 1, A + 2, A + 3);
    int4 y = make_int4((int)threadIdx.x, (int)threadIdx.x,
                       (int)threadIdx.x, (int)threadIdx.x);

    #pragma unroll 1
    for (int i = 0; i < 32; i++)
    {
        IMAD4_16(x, y)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = y.x + y.y + y.z + y.w;
}
