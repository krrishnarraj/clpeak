// Single-precision MAD-chain throughput.  Mirrors compute_sp_v1.comp /
// compute_sp.cu: x = x*x + c with c a per-thread loop invariant, 128 outer
// iters * 16 FMAs * 2 ops = 4096 ops/thread = COMPUTE_FP_WORK_PER_WI.
//
// Chain shape and why: see the MAD chain block in include/common/common.h.
// This kernel is where that shape was measured -- the ping-pong form it
// replaced ran the 8-wide variant at half rate on Apple GPUs.

#include <metal_stdlib>
using namespace metal;

#define MAD_4(x, c)  x = fma(x, x, c); x = fma(x, x, c); x = fma(x, x, c); x = fma(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

kernel void compute_sp(device float* out [[buffer(0)]],
                       constant float& A [[buffer(1)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    float x = A;
    float c = (float)lid;

    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, c)
    }

    out[tid] = x;
}

// 64 outer * 16 packed FMAs * 4 ops (2 lanes * 2 ops) = 4096 ops/thread.
kernel void compute_sp2(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    float2 x = float2(A, A + 1.0f);
    float2 c = float2((float)lid);

    for (int i = 0; i < 64; i++)
    {
        MAD_16(x, c)
    }

    out[tid] = x.x + x.y;
}

// 32 outer * 16 packed FMAs * 8 ops = 4096 ops/thread.
kernel void compute_sp4(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    float4 x = float4(A, A + 1.0f, A + 2.0f, A + 3.0f);
    float4 c = float4((float)lid);

    for (int i = 0; i < 32; i++)
    {
        MAD_16(x, c)
    }

    out[tid] = x.x + x.y + x.z + x.w;
}

// 16 outer * 16 packed FMAs * 16 ops = 4096 ops/thread.
kernel void compute_sp8(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    // MSL has no float8, but a pair of float4s gives the same packed-FMA shape
    // and keeps the dependency chain identical to compute_sp_v8 in OpenCL.
    // Both chains share one invariant, so 8 values in flight cost 12 live
    // registers here rather than the 16 the ping-pong form needed.
    float4 xa = float4(A,        A + 1.0f, A + 2.0f, A + 3.0f);
    float4 xb = float4(A + 4.0f, A + 5.0f, A + 6.0f, A + 7.0f);
    float4 c  = float4((float)lid);

    for (int i = 0; i < 16; i++)
    {
        MAD_16(xa, c)
        MAD_16(xb, c)
    }

    float4 r = xa + xb;
    out[tid] = r.x + r.y + r.z + r.w;
}
