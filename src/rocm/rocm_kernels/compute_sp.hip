#include <hip/hip_runtime.h>

// Single-precision MAD-chain throughput across the HIP SDK's native vector
// widths (float, float2, float4). hip_vector_types.h provides up to width 4;
// there is no float8/float16 HIP type as there is in OpenCL C. Each variant
// keeps the SAME work per thread (4096 flops): the loop trip count is divided
// by the vector width, so wider vectors expose more per-thread ILP at constant
// work and the throughput numbers are directly comparable. HIP vector types
// carry no arithmetic operators, so the MAD chain is built component-wise via
// fmaf.

// ---- scalar: float -------------------------------------------------------
#define MAD_4(x, y)  x = fmaf(y, x, y); y = fmaf(x, y, x); x = fmaf(y, x, y); y = fmaf(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

extern "C" __global__ void compute_sp(float *out, float A)
{
    float x0 = A;
    float y0 = (float)threadIdx.x;
    float x1 = A + 1.0f;
    float y1 = (float)threadIdx.x + 7.0f;

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        MAD_16(x0, y0)
        MAD_16(x1, y1)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = y0 + y1;
}

// ---- float2 --------------------------------------------------------------
#define FMA2(a, b, c) make_float2(fmaf((a).x, (b).x, (c).x), fmaf((a).y, (b).y, (c).y))
#define MAD2_4(x, y)  x = FMA2(y, x, y); y = FMA2(x, y, x); x = FMA2(y, x, y); y = FMA2(x, y, x);
#define MAD2_16(x, y) MAD2_4(x, y) MAD2_4(x, y) MAD2_4(x, y) MAD2_4(x, y)

extern "C" __global__ void compute_sp_v2(float *out, float A)
{
    float2 x0 = make_float2(A, A + 1.0f);
    float2 y0 = make_float2((float)threadIdx.x, (float)threadIdx.x);
    float2 x1 = make_float2(A + 2.0f, A + 3.0f);
    float2 y1 = make_float2((float)threadIdx.x + 7.0f, (float)threadIdx.x + 7.0f);

    #pragma unroll 1
    for (int i = 0; i < 32; i++)
    {
        MAD2_16(x0, y0)
        MAD2_16(x1, y1)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = (y0.x + y0.y) + (y1.x + y1.y);
}

// ---- float4 --------------------------------------------------------------
#define FMA4(a, b, c) make_float4(fmaf((a).x, (b).x, (c).x), fmaf((a).y, (b).y, (c).y), \
                                  fmaf((a).z, (b).z, (c).z), fmaf((a).w, (b).w, (c).w))
#define MAD4_4(x, y)  x = FMA4(y, x, y); y = FMA4(x, y, x); x = FMA4(y, x, y); y = FMA4(x, y, x);
#define MAD4_16(x, y) MAD4_4(x, y) MAD4_4(x, y) MAD4_4(x, y) MAD4_4(x, y)

extern "C" __global__ void compute_sp_v4(float *out, float A)
{
    float4 x0 = make_float4(A, A + 1.0f, A + 2.0f, A + 3.0f);
    float4 y0 = make_float4((float)threadIdx.x, (float)threadIdx.x,
                            (float)threadIdx.x, (float)threadIdx.x);
    float4 x1 = make_float4(A + 4.0f, A + 5.0f, A + 6.0f, A + 7.0f);
    float4 y1 = make_float4((float)threadIdx.x + 7.0f, (float)threadIdx.x + 7.0f,
                            (float)threadIdx.x + 7.0f, (float)threadIdx.x + 7.0f);

    #pragma unroll 1
    for (int i = 0; i < 16; i++)
    {
        MAD4_16(x0, y0)
        MAD4_16(x1, y1)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] =
        (y0.x + y0.y + y0.z + y0.w) + (y1.x + y1.y + y1.z + y1.w);
}
