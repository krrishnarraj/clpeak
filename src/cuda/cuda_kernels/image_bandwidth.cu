// Image (texture) bandwidth via cudaTextureObject_t / tex2D fetch.
// Each thread reads 16 RGBA float pixels with nearest-neighbour sampling.
//
// Bytes = IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads.

extern "C" __global__ void image_bandwidth(cudaTextureObject_t tex, float *out, int width, int height)
{
    int gid   = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int gsize = (int)(gridDim.x * blockDim.x);
    int total = width * height;

    float4 sum0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sum3 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll 4
    for (int i = 0; i < 16; i++)
    {
        int pixel = (gid + i * gsize) % total;
        int x = pixel % width;
        int y = pixel / width;
        float4 v = tex2D<float4>(tex, x, y);
        switch (i & 3) {
        case 0: sum0 += v; break;
        case 1: sum1 += v; break;
        case 2: sum2 += v; break;
        case 3: sum3 += v; break;
        }
    }
    float4 sum = make_float4(
        sum0.x + sum1.x + sum2.x + sum3.x,
        sum0.y + sum1.y + sum2.y + sum3.y,
        sum0.z + sum1.z + sum2.z + sum3.z,
        sum0.w + sum1.w + sum2.w + sum3.w);
    out[gid] = sum.x + sum.y + sum.z + sum.w;
}
