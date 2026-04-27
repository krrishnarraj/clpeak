// Local memory (__shared__) bandwidth -- ping-pong write/read with two
// __syncthreads per rep.  Mirrors local_bandwidth_kernels.cl::v1..v8.
//
// 64 reps * (1 write + 1 read) = 128 accesses per WI per kernel call.
// Bytes = LMEM_REPS * 2 * width * sizeof(float) * globalThreads.

extern "C" __global__ void local_bandwidth_v1(float *out)
{
    __shared__ float scratch[256];
    unsigned int lid   = threadIdx.x;
    unsigned int lsize = blockDim.x;
    unsigned int gid   = blockIdx.x * lsize + lid;
    unsigned int next  = (lid + 1u) % lsize;

    float sum = (float)gid;
    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        scratch[lid] = sum;
        __syncthreads();
        sum = scratch[next];
        __syncthreads();
    }
    out[gid] = sum;
}

extern "C" __global__ void local_bandwidth_v2(float *out)
{
    __shared__ float2 scratch[256];
    unsigned int lid   = threadIdx.x;
    unsigned int lsize = blockDim.x;
    unsigned int gid   = blockIdx.x * lsize + lid;
    unsigned int next  = (lid + 1u) % lsize;

    float2 sum = make_float2((float)gid, (float)(gid + 1u));
    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        scratch[lid] = sum;
        __syncthreads();
        sum = scratch[next];
        __syncthreads();
    }
    out[gid] = sum.x + sum.y;
}

extern "C" __global__ void local_bandwidth_v4(float *out)
{
    __shared__ float4 scratch[256];
    unsigned int lid   = threadIdx.x;
    unsigned int lsize = blockDim.x;
    unsigned int gid   = blockIdx.x * lsize + lid;
    unsigned int next  = (lid + 1u) % lsize;

    float4 sum = make_float4((float)gid,     (float)(gid + 1u),
                             (float)(gid + 2u), (float)(gid + 3u));
    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        scratch[lid] = sum;
        __syncthreads();
        sum = scratch[next];
        __syncthreads();
    }
    out[gid] = sum.x + sum.y + sum.z + sum.w;
}

extern "C" __global__ void local_bandwidth_v8(float *out)
{
    // 256 threads * 8 floats * 4 bytes = 8 KB per block -- well within
    // the per-SM shared-memory budget on every CUDA arch sm_50+.
    __shared__ float4 scratch_lo[256];
    __shared__ float4 scratch_hi[256];
    unsigned int lid   = threadIdx.x;
    unsigned int lsize = blockDim.x;
    unsigned int gid   = blockIdx.x * lsize + lid;
    unsigned int next  = (lid + 1u) % lsize;

    float4 lo = make_float4((float)gid,     (float)(gid + 1u),
                            (float)(gid + 2u), (float)(gid + 3u));
    float4 hi = make_float4((float)(gid + 4u), (float)(gid + 5u),
                            (float)(gid + 6u), (float)(gid + 7u));
    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        scratch_lo[lid] = lo;
        scratch_hi[lid] = hi;
        __syncthreads();
        lo = scratch_lo[next];
        hi = scratch_hi[next];
        __syncthreads();
    }
    out[gid] = lo.x + lo.y + lo.z + lo.w + hi.x + hi.y + hi.z + hi.w;
}
