// atomic_float: Metal exposes fetch_add for device address space only on this
// SDK family. threadgroup/local atomic_float is rejected by the compiler, so
// the float atomic throughput test has a global variant only.

#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void atomic_throughput_global_float(device atomic_float* counter [[buffer(0)]],
                                           uint tid [[thread_position_in_grid]])
{
    device atomic_float* cnt = counter + tid;
    for (int i = 0; i < 512; i++)
    {
        atomic_fetch_add_explicit(cnt, 1.0f, memory_order_relaxed);
    }
}
