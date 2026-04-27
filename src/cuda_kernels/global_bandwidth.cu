// Global memory read bandwidth.  One float per fetch, FETCH_PER_WI fetches
// per thread, with the per-block stride layout from global_bandwidth_v1.comp
// so adjacent blocks own disjoint memory regions and we measure VRAM rather
// than L2.

#define FETCH_PER_WI 16

extern "C" __global__ void global_bandwidth(const float *in, float *out)
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
