// Local memory (Metal: threadgroup memory) bandwidth.  Ping-pong
// write/read with two threadgroup_barriers per rep, mirrors
// local_bandwidth_kernels.cl::v1..v8.
//
// 64 reps * (1 write + 1 read) = 128 accesses per WI per kernel call.
// Bytes = LMEM_REPS * 2 * width * sizeof(float) * globalThreads.

#include <metal_stdlib>
using namespace metal;

kernel void local_bandwidth_v1(device float* out [[buffer(0)]],
                               uint tid [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float scratch[256];
    uint next = (lid + 1u) % tg_size;
    float sum = (float)tid;
    for (int i = 0; i < 64; i++)
    {
        scratch[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = scratch[next];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = sum;
}

kernel void local_bandwidth_v2(device float* out [[buffer(0)]],
                               uint tid [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float2 scratch[256];
    uint next = (lid + 1u) % tg_size;
    float2 sum = float2((float)tid, (float)(tid + 1u));
    for (int i = 0; i < 64; i++)
    {
        scratch[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = scratch[next];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = sum.x + sum.y;
}

kernel void local_bandwidth_v4(device float* out [[buffer(0)]],
                               uint tid [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float4 scratch[256];
    uint next = (lid + 1u) % tg_size;
    float4 sum = float4((float)tid,     (float)(tid + 1u),
                        (float)(tid + 2u), (float)(tid + 3u));
    for (int i = 0; i < 64; i++)
    {
        scratch[lid] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = scratch[next];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = sum.x + sum.y + sum.z + sum.w;
}

kernel void local_bandwidth_v8(device float* out [[buffer(0)]],
                               uint tid [[thread_position_in_grid]],
                               uint lid [[thread_position_in_threadgroup]],
                               uint tg_size [[threads_per_threadgroup]])
{
    // 256 threads * 8 floats * 4 bytes = 8 KB per threadgroup -- fits
    // in Apple silicon's per-threadgroup memory (32 KB on Apple7+).
    threadgroup float4 scratch_lo[256];
    threadgroup float4 scratch_hi[256];
    uint next = (lid + 1u) % tg_size;
    float4 lo = float4((float)tid,     (float)(tid + 1u),
                       (float)(tid + 2u), (float)(tid + 3u));
    float4 hi = float4((float)(tid + 4u), (float)(tid + 5u),
                       (float)(tid + 6u), (float)(tid + 7u));
    for (int i = 0; i < 64; i++)
    {
        scratch_lo[lid] = lo;
        scratch_hi[lid] = hi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        lo = scratch_lo[next];
        hi = scratch_hi[next];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = lo.x + lo.y + lo.z + lo.w + hi.x + hi.y + hi.z + hi.w;
}
