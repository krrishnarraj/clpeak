// Single-precision MAD-chain throughput.  Mirrors compute_sp_v1.comp /
// compute_sp.cu: ping-pong x/y feedback, 128 outer iters * 16 FMAs * 2 ops
// = 4096 ops/thread = COMPUTE_FP_WORK_PER_WI.

#include <metal_stdlib>
using namespace metal;

#define MAD_4(x, y)  x = fma(y, x, y); y = fma(x, y, x); x = fma(y, x, y); y = fma(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

kernel void compute_sp(device float* out [[buffer(0)]],
                       constant float& A [[buffer(1)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    float x = A;
    float y = (float)lid;

    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, y)
    }

    out[tid] = y;
}

// 64 outer * 16 packed FMAs * 4 ops (2 lanes * 2 ops) = 4096 ops/thread.
kernel void compute_sp2(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    float2 x = float2(A, A + 1.0f);
    float2 y = float2((float)lid);

    for (int i = 0; i < 64; i++)
    {
        MAD_16(x, y)
    }

    float2 r = x + y;
    out[tid] = r.x + r.y;
}

// 32 outer * 16 packed FMAs * 8 ops = 4096 ops/thread.
kernel void compute_sp4(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    float4 x = float4(A, A + 1.0f, A + 2.0f, A + 3.0f);
    float4 y = float4((float)lid);

    for (int i = 0; i < 32; i++)
    {
        MAD_16(x, y)
    }

    float4 r = x + y;
    out[tid] = r.x + r.y + r.z + r.w;
}

// 16 outer * 16 packed FMAs * 16 ops = 4096 ops/thread.
kernel void compute_sp8(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    // MSL has no float8, but a pair of float4s gives the same packed-FMA shape
    // and keeps the dependency chain identical to compute_sp_v8 in OpenCL.
    float4 xa = float4(A,        A + 1.0f, A + 2.0f, A + 3.0f);
    float4 xb = float4(A + 4.0f, A + 5.0f, A + 6.0f, A + 7.0f);
    float4 ya = float4((float)lid);
    float4 yb = float4((float)lid);

    for (int i = 0; i < 16; i++)
    {
        MAD_16(xa, ya)
        MAD_16(xb, yb)
    }

    float4 r = xa + ya + xb + yb;
    out[tid] = r.x + r.y + r.z + r.w;
}
