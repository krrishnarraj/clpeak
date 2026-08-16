// Mixed-precision MAC: half inputs, float accumulator.  Mirrors the
// Vulkan compute_mp_v1.comp and CUDA compute_mp.cu structure -- inner FMA
// chain stays in fp32 (the data path the shader-core actually uses for
// fp16xfp16+fp32) and the half cast happens once per outer iter to force
// the conversion units to participate.
//
// Chain shape and why: see the MAD chain block in include/common/common.h.
// c is seeded through half once and never updated, so it stays
// half-representable without a per-iteration round trip.

#include <metal_stdlib>
using namespace metal;

#define MAD_4(x, c)  x = fma(x, x, c); x = fma(x, x, c); x = fma(x, x, c); x = fma(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

kernel void compute_mp(device float* out [[buffer(0)]],
                       constant float& A [[buffer(1)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    float x = (float)((half)A);
    float c = (float)((half)lid);

    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, c)
        x = (float)((half)x);
    }

    out[tid] = x;
}

// float2 accumulator, half2 cast per outer iter.
// 64 outer * 16 fmas * 4 ops = 4096 ops/thread.
kernel void compute_mp2(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    float2 x = float2((float)((half)A),         (float)((half)(A + 1.0f)));
    float2 c = float2((float)((half)lid),       (float)((half)(lid + 1u)));

    for (int i = 0; i < 64; i++)
    {
        MAD_16(x, c)
        x = float2((float)(half)x.x, (float)(half)x.y);
    }

    out[tid] = x.x + x.y;
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
    float4 c = float4((float)((half)lid),
                      (float)((half)(lid + 1u)),
                      (float)((half)(lid + 2u)),
                      (float)((half)(lid + 3u)));

    for (int i = 0; i < 32; i++)
    {
        MAD_16(x, c)
        half4 hx = half4(x);
        x = float4(hx);
    }

    out[tid] = x.x + x.y + x.z + x.w;
}
