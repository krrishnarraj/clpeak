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

