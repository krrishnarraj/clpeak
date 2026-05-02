// FP16 MAD-chain throughput.  Two variants:
//   compute_hp   -- scalar half FMA
//   compute_hp2  -- half2 packed FMA
//
// Apple silicon note: unlike NVIDIA HFMA2 (2x FP32 rate) or AMD's WMMA-
// adjacent fp16 path, the Apple silicon shader core does NOT have a fp16
// throughput advantage over fp32 -- both flavors lower to the same FMA
// pipe.  Both compute_hp variants therefore plateau near the FP32 peak
// from compute_sp; the only path to Apple's true fp16 throughput is
// simdgroup_matrix (the matrix engine), measured separately.  Reporting
// the shader-core hp number anyway is still useful for cross-backend
// comparison vs. NVIDIA / AMD where a delta does exist.
//
// Op accounting matches compute_hp.cu: 4096 fp16 ops/thread either way.

#include <metal_stdlib>
using namespace metal;

#define MAD_4(x, y)  x = fma(y, x, y); y = fma(x, y, x); x = fma(y, x, y); y = fma(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

kernel void compute_hp(device float* out [[buffer(0)]],
                       constant float& A [[buffer(1)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    half x = (half)A;
    half y = (half)lid;

    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, y)
    }

    out[tid] = (float)(x + y);
}

kernel void compute_hp2(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    half2 x = half2((half)A, (half)A);
    half2 y = half2((half)lid, (half)(lid + 1));

    // 64 outer * 16 packed FMAs * 4 ops (2 lanes * 2 ops) = 4096 ops/thread.
    for (int i = 0; i < 64; i++)
    {
        MAD_16(x, y)
    }

    half2 r = x + y;
    out[tid] = (float)(r.x + r.y);
}

// 32 outer * 16 packed FMAs * 8 ops (4 lanes * 2 ops) = 4096 ops/thread.
kernel void compute_hp4(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    half4 x = half4((half)A, (half)(A + 1.0f), (half)(A + 2.0f), (half)(A + 3.0f));
    half4 y = half4((half)lid);

    for (int i = 0; i < 32; i++)
    {
        MAD_16(x, y)
    }

    half4 r = x + y;
    out[tid] = (float)(r.x + r.y + r.z + r.w);
}

// MSL has no native half8.  Pair two half4 chains; 16 outer * 16 fmas * 8 ops
// * 2 chains = 4096 ops/thread.
kernel void compute_hp8(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    half4 xa = half4((half)A,        (half)(A + 1.0f), (half)(A + 2.0f), (half)(A + 3.0f));
    half4 xb = half4((half)(A + 4.0f), (half)(A + 5.0f), (half)(A + 6.0f), (half)(A + 7.0f));
    half4 ya = half4((half)lid);
    half4 yb = half4((half)lid);

    for (int i = 0; i < 16; i++)
    {
        MAD_16(xa, ya)
        MAD_16(xb, yb)
    }

    half4 r = xa + ya + xb + yb;
    out[tid] = (float)(r.x + r.y + r.z + r.w);
}
