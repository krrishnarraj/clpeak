// 64-bit atomic add/sub/min/max requires a recent SDK and a GPU family that
// supports `atomic_ulong` fetch operations.  When unsupported, MSL rejects
// the fetch_add call with a `_valid_fetch_add_type` constraint failure.
// We isolate this kernel in its own .metal source so a compile failure here
// only affects the ulong variant; the int / uint / float variants in their
// own sources still work.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void atomic_throughput_global_ulong(device atomic_ulong* counter [[buffer(0)]],
                                           uint tid [[thread_position_in_grid]])
{
    device atomic_ulong* cnt = counter + tid;
    for (int i = 0; i < 512; i++)
    {
        atomic_fetch_add_explicit(cnt, 1ul, memory_order_relaxed);
    }
}

kernel void atomic_throughput_local_ulong(device ulong* out [[buffer(0)]],
                                          uint lid [[thread_position_in_threadgroup]],
                                          uint tg_id [[threadgroup_position_in_grid]])
{
    threadgroup atomic_ulong scratch;
    if (lid == 0)
        atomic_store_explicit(&scratch, 0ul, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < 512; i++)
    {
        atomic_fetch_add_explicit(&scratch, 1ul, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0)
        out[tg_id] = atomic_load_explicit(&scratch, memory_order_relaxed);
}
