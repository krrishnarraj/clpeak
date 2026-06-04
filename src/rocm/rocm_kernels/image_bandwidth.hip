#include <hip/hip_runtime.h>

extern "C" __global__ void image_bandwidth(hipTextureObject_t tex, float *out, int width, int height)
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
        float4 v = tex2D<float4>(tex, (float)x, (float)y);
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    out[gid] = sum.x + sum.y + sum.z + sum.w;
}
