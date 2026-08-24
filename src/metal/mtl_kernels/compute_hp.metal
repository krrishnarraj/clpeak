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
//
// Chain shape and why: see the MAD chain block in include/common/common.h.

#include <metal_stdlib>
using namespace metal;

#define MAD_4(x, c)  x = fma(x, x, c); x = fma(x, x, c); x = fma(x, x, c); x = fma(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

kernel void compute_hp(device float* out [[buffer(0)]],
                       constant float& A [[buffer(1)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    half x = (half)A;
    half c = (half)lid;

    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, c)
    }

    out[tid] = (float)x;
}

kernel void compute_hp2(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    half2 x = half2((half)A, (half)A);
    half2 c = half2((half)lid, (half)(lid + 1));

    // 64 outer * 16 packed FMAs * 4 ops (2 lanes * 2 ops) = 4096 ops/thread.
    for (int i = 0; i < 64; i++)
    {
        MAD_16(x, c)
    }

    out[tid] = (float)(x.x + x.y);
}

// 32 outer * 16 packed FMAs * 8 ops (4 lanes * 2 ops) = 4096 ops/thread.
kernel void compute_hp4(device float* out [[buffer(0)]],
                        constant float& A [[buffer(1)]],
                        uint tid [[thread_position_in_grid]],
                        uint lid [[thread_position_in_threadgroup]])
{
    half4 x = half4((half)A, (half)(A + 1.0f), (half)(A + 2.0f), (half)(A + 3.0f));
    half4 c = half4((half)lid);

    for (int i = 0; i < 32; i++)
    {
        MAD_16(x, c)
    }

    out[tid] = (float)(x.x + x.y + x.z + x.w);
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
    half4 c  = half4((half)lid);

    for (int i = 0; i < 16; i++)
    {
        MAD_16(xa, c)
        MAD_16(xb, c)
    }

    half4 r = xa + xb;
    out[tid] = (float)(r.x + r.y + r.z + r.w);
}

// ---- affine-chain variants (raced against the ones above; see mad_chain.metal) ----

kernel void compute_hp_alt(device float* out [[buffer(0)]],
                            constant float& A [[buffer(1)]],
                            uint tid [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]])
{
    AF4_DECL(half, (half)A, (half)lid)

    for (int i = 0; i < 128; i++)
    {
        AF4_16
    }

    half r = AF4_RES;
    out[tid] = (float)r;
}

kernel void compute_hp2_alt(device float* out [[buffer(0)]],
                             constant float& A [[buffer(1)]],
                             uint tid [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]])
{
    AF2_DECL(half2, half2((half)A, (half)(A + 1.0f)), half2((half)lid))

    for (int i = 0; i < 64; i++)
    {
        AF2_16
    }

    half2 r = AF2_RES;
    out[tid] = (float)(r.x + r.y);
}

kernel void compute_hp4_alt(device float* out [[buffer(0)]],
                             constant float& A [[buffer(1)]],
                             uint tid [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]])
{
    AF1_DECL(half4, half4((half)A, (half)(A + 1.0f), (half)(A + 2.0f), (half)(A + 3.0f)), half4((half)lid))

    for (int i = 0; i < 32; i++)
    {
        AF1_16
    }

    half4 r = AF1_RES;
    out[tid] = (float)(r.x + r.y + r.z + r.w);
}
