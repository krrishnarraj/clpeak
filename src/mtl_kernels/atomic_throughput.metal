// Atomic throughput -- two patterns:
//   global: each thread hammers its own counter in device memory (no
//           cross-thread contention).
//   local:  every thread in the threadgroup contends on a single
//           threadgroup atomic_int.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void atomic_throughput_global(device atomic_int* counter [[buffer(0)]],
                                     uint tid [[thread_position_in_grid]])
{
    device atomic_int* cnt = counter + tid;
    for (int i = 0; i < 512; i++)
    {
        atomic_fetch_add_explicit(cnt, 1, memory_order_relaxed);
    }
}

kernel void atomic_throughput_local(device int* out [[buffer(0)]],
                                    uint lid [[thread_position_in_threadgroup]],
                                    uint tg_id [[threadgroup_position_in_grid]])
{
    threadgroup atomic_int scratch;
    if (lid == 0)
        atomic_store_explicit(&scratch, 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int i = 0; i < 512; i++)
    {
        atomic_fetch_add_explicit(&scratch, 1, memory_order_relaxed);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0)
        out[tg_id] = atomic_load_explicit(&scratch, memory_order_relaxed);
}
