MSTRINGIFY(

// Ping-pong kernel: each WI writes to lmem[lid] then reads from lmem[(lid+1)%lsize].
// Two barriers per rep ensure true read-after-write dependencies across work-items,
// preventing the compiler from eliminating the local memory traffic.
// Reps=64: 64 * (1 write + 1 read) = 128 accesses per WI per kernel call.

__kernel void local_bandwidth_v1(__global float* output, __local float* scratch)
{
    uint lid   = get_local_id(0);
    uint lsize = get_local_size(0);
    uint gid   = get_global_id(0);
    uint next  = (lid + 1) % lsize;

    float sum = (float)gid;

    for (int i = 0; i < 64; i++) {
        scratch[lid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        sum = scratch[next];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    output[gid] = sum;
}


__kernel void local_bandwidth_v2(__global float* output, __local float2* scratch)
{
    uint lid   = get_local_id(0);
    uint lsize = get_local_size(0);
    uint gid   = get_global_id(0);
    uint next  = (lid + 1) % lsize;

    float2 sum = (float2)((float)gid, (float)(gid + 1));

    for (int i = 0; i < 64; i++) {
        scratch[lid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        sum = scratch[next];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    output[gid] = sum.s0 + sum.s1;
}


__kernel void local_bandwidth_v4(__global float* output, __local float4* scratch)
{
    uint lid   = get_local_id(0);
    uint lsize = get_local_size(0);
    uint gid   = get_global_id(0);
    uint next  = (lid + 1) % lsize;

    float4 sum = (float4)((float)gid, (float)(gid + 1),
                          (float)(gid + 2), (float)(gid + 3));

    for (int i = 0; i < 64; i++) {
        scratch[lid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        sum = scratch[next];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    output[gid] = sum.s0 + sum.s1 + sum.s2 + sum.s3;
}


__kernel void local_bandwidth_v8(__global float* output, __local float8* scratch)
{
    uint lid   = get_local_id(0);
    uint lsize = get_local_size(0);
    uint gid   = get_global_id(0);
    uint next  = (lid + 1) % lsize;

    float8 sum = (float8)((float)gid,     (float)(gid + 1),
                          (float)(gid + 2), (float)(gid + 3),
                          (float)(gid + 4), (float)(gid + 5),
                          (float)(gid + 6), (float)(gid + 7));

    for (int i = 0; i < 64; i++) {
        scratch[lid] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        sum = scratch[next];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    output[gid] = sum.s0 + sum.s1 + sum.s2 + sum.s3 +
                  sum.s4 + sum.s5 + sum.s6 + sum.s7;
}

)
