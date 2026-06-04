#include <hip/hip_runtime.h>

#define FETCH_PER_WI 16

extern "C" __global__ void global_bandwidth_v1(const float *in, float *out)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lid = threadIdx.x;
    unsigned int wid = blockIdx.x;
    unsigned int lsz = blockDim.x;
    unsigned int offset = wid * lsz * FETCH_PER_WI + lid;

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < FETCH_PER_WI; i++)
    {
        sum += in[offset];
        offset += lsz;
    }
    out[gid] = sum;
}

extern "C" __global__ void global_bandwidth_v2(const float2 *in, float *out)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lid = threadIdx.x;
    unsigned int wid = blockIdx.x;
    unsigned int lsz = blockDim.x;
    unsigned int offset = wid * lsz * FETCH_PER_WI + lid;

    float2 sum = make_float2(0.0f, 0.0f);
    #pragma unroll
    for (int i = 0; i < FETCH_PER_WI; i++)
    {
        float2 v = in[offset];
        sum.x += v.x; sum.y += v.y;
        offset += lsz;
    }
    out[gid] = sum.x + sum.y;
}

extern "C" __global__ void global_bandwidth_v4(const float4 *in, float *out)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lid = threadIdx.x;
    unsigned int wid = blockIdx.x;
    unsigned int lsz = blockDim.x;
    unsigned int offset = wid * lsz * FETCH_PER_WI + lid;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int i = 0; i < FETCH_PER_WI; i++)
    {
        float4 v = in[offset];
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        offset += lsz;
    }
    out[gid] = sum.x + sum.y + sum.z + sum.w;
}
