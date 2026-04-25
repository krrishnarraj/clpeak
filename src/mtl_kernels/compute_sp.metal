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
