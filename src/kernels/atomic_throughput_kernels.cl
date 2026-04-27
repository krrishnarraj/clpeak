MSTRINGIFY(

// Global atomics: each work-item adds to its own counter (independent, no cross-WI
// contention). Measures peak global-memory atomic throughput. Each WI executes
// 512 serial atomic_add calls; the loop dependency prevents the compiler from
// hoisting or eliminating them.
__kernel void atomic_throughput_global(__global int* counter)
{
    __global int* cnt = counter + get_global_id(0);

    for (int i = 0; i < 512; i++) {
        atomic_add(cnt, 1);
    }
}


// Local atomics: all work-items within a workgroup contend on a single __local
// int, the most common pattern (histogram, reduction). WI 0 initialises the
// counter to 0 before the loop so results are deterministic across kernel
// re-invocations. The final value is written to global memory to prevent the
// entire computation from being eliminated by the compiler.
__kernel void atomic_throughput_local(__global int* output, __local int* scratch)
{
    if (get_local_id(0) == 0)
        *scratch = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < 512; i++) {
        atomic_add(scratch, 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) == 0)
        output[get_group_id(0)] = *scratch;
}

)
