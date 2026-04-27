// Image (texture) bandwidth via cudaTextureObject_t / tex2D fetch.
// Each thread reads 16 RGBA float pixels with nearest-neighbour sampling.
//
// Bytes = IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads.

extern "C" __global__ void image_bandwidth(cudaTextureObject_t tex, float *out, int width, int height)
{
    int gid   = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int gsize = (int)(gridDim.x * blockDim.x);
    int total = width * height;

    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int i = 0; i < 16; i++)
    {
        int pixel = (gid + i * gsize) % total;
        int x = pixel % width;
        int y = pixel / width;
        float4 v = tex2D<float4>(tex, x, y);
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    out[gid] = sum.x + sum.y + sum.z + sum.w;
}
