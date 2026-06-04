#include <hip/hip_runtime.h>

extern "C" __global__ void atomic_throughput_global(int *counter)
{
    int *cnt = counter + blockIdx.x * blockDim.x + threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        atomicAdd(cnt, 1);
    }
}

extern "C" __global__ void atomic_throughput_local(int *out)
{
    __shared__ int scratch;
    if (threadIdx.x == 0) scratch = 0;
    __syncthreads();

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        atomicAdd(&scratch, 1);
    }

    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = scratch;
}
