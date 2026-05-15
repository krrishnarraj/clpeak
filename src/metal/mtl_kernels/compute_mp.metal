// Mixed-precision MAC: half inputs, float accumulator.  Mirrors the
// Vulkan compute_mp_v1.comp and CUDA compute_mp.cu structure -- inner FMA
// chain stays in fp32 (the data path the shader-core actually uses for
// fp16xfp16+fp32) and the half cast happens once per outer iter to force
// the conversion units to participate.

#include <metal_stdlib>
using namespace metal;

#define MAD_4(x, y)  x = fma(y, x, y); y = fma(x, y, x); x = fma(y, x, y); y = fma(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

kernel void compute_mp(device float* out [[buffer(0)]],
                       constant float& A [[buffer(1)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    float x = (float)((half)A);
    float y = (float)((half)lid);

    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, y)
        x = (float)((half)x);
        y = (float)((half)y);
    }

    out[tid] = x + y;
}

// float2 accumulator, half2 cast per outer iter.
// 64 outer * 16 fmas * 4 ops = 4096 ops/thread.
kernel void compute_mp2(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    float2 x = float2((float)((half)A),         (float)((half)(A + 1.0f)));
    float2 y = float2((float)((half)lid),       (float)((half)(lid + 1u)));

    for (int i = 0; i < 64; i++)
    {
        MAD_16(x, y)
        x = float2((float)(half)x.x, (float)(half)x.y);
        y = float2((float)(half)y.x, (float)(half)y.y);
    }

    float2 r = x + y;
    out[tid] = r.x + r.y;
}

// float4 accumulator, half4 cast per outer iter.
// 32 outer * 16 fmas * 8 ops = 4096 ops/thread.
kernel void compute_mp4(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    float4 x = float4((float)((half)A),
                      (float)((half)(A + 1.0f)),
                      (float)((half)(A + 2.0f)),
                      (float)((half)(A + 3.0f)));
    float4 y = float4((float)((half)lid),
                      (float)((half)(lid + 1u)),
                      (float)((half)(lid + 2u)),
                      (float)((half)(lid + 3u)));

    for (int i = 0; i < 32; i++)
    {
        MAD_16(x, y)
        half4 hx = half4(x);
        half4 hy = half4(y);
        x = float4(hx);
        y = float4(hy);
    }

    float4 r = x + y;
    out[tid] = r.x + r.y + r.z + r.w;
}
