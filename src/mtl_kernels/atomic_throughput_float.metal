// atomic_float: only the device-address-space form is valid in MSL.
// threadgroup atomic_float is rejected by the compiler, so we expose the
// global variant only.  Apple silicon Apple7+ on macOS 12+ has
// atomic<float> fetch_add.

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
