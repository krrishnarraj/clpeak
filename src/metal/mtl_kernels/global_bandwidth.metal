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

kernel void global_bandwidth_v2(const device float2* in [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                uint tid [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint tg_id [[threadgroup_position_in_grid]],
                                uint tg_size [[threads_per_threadgroup]])
{
    uint offset = tg_id * tg_size * FETCH_PER_WI + lid;
    float2 sum = float2(0.0f);
    for (int i = 0; i < FETCH_PER_WI; i++)
    {
        sum += in[offset];
        offset += tg_size;
    }
    out[tid] = sum.x + sum.y;
}

kernel void global_bandwidth_v4(const device float4* in [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                uint tid [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint tg_id [[threadgroup_position_in_grid]],
                                uint tg_size [[threads_per_threadgroup]])
{
    uint offset = tg_id * tg_size * FETCH_PER_WI + lid;
    float4 sum = float4(0.0f);
    for (int i = 0; i < FETCH_PER_WI; i++)
    {
        sum += in[offset];
        offset += tg_size;
    }
    out[tid] = sum.x + sum.y + sum.z + sum.w;
}

// MSL has no float8 / float16 builtin.  Mirror the OpenCL v8/v16 shape with
// 2x / 4x float4 reads per iter so the per-WI byte budget matches.
kernel void global_bandwidth_v8(const device float4* in [[buffer(0)]],
                                device float* out [[buffer(1)]],
                                uint tid [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint tg_id [[threadgroup_position_in_grid]],
                                uint tg_size [[threads_per_threadgroup]])
{
    // Each thread reads FETCH_PER_WI logical "v8" elements = 2 * float4s
    // each.  Stride and base offset mimic the v1 layout but in v8 units.
    uint offset = tg_id * tg_size * FETCH_PER_WI * 2u + lid;
    float4 sa = float4(0.0f), sb = float4(0.0f);
    for (int i = 0; i < FETCH_PER_WI; i++)
    {
        sa += in[offset];
        sb += in[offset + tg_size];
        offset += tg_size * 2u;
    }
    float4 s = sa + sb;
    out[tid] = s.x + s.y + s.z + s.w;
}

kernel void global_bandwidth_v16(const device float4* in [[buffer(0)]],
                                 device float* out [[buffer(1)]],
                                 uint tid [[thread_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]],
                                 uint tg_id [[threadgroup_position_in_grid]],
                                 uint tg_size [[threads_per_threadgroup]])
{
    uint offset = tg_id * tg_size * FETCH_PER_WI * 4u + lid;
    float4 sa = float4(0.0f), sb = float4(0.0f), sc = float4(0.0f), sd = float4(0.0f);
    for (int i = 0; i < FETCH_PER_WI; i++)
    {
        sa += in[offset];
        sb += in[offset + tg_size];
        sc += in[offset + tg_size * 2u];
        sd += in[offset + tg_size * 3u];
        offset += tg_size * 4u;
    }
    float4 s = sa + sb + sc + sd;
    out[tid] = s.x + s.y + s.z + s.w;
}
