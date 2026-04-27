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
