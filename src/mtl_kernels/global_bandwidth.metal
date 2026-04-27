// Global memory read bandwidth.  Per-threadgroup stride layout so adjacent
// threadgroups own disjoint regions of the input buffer (avoids
// inadvertently measuring L2/system-cache reuse).

#include <metal_stdlib>
using namespace metal;

#define FETCH_PER_WI 16

kernel void global_bandwidth(const device float* in [[buffer(0)]],
                             device float* out [[buffer(1)]],
                             uint tid [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint tg_id [[threadgroup_position_in_grid]],
                             uint tg_size [[threads_per_threadgroup]])
{
    uint offset = tg_id * tg_size * FETCH_PER_WI + lid;

    float sum = 0.0f;
    for (int i = 0; i < FETCH_PER_WI; i++)
    {
        sum += in[offset];
        offset += tg_size;
    }

    out[tid] = sum;
}
