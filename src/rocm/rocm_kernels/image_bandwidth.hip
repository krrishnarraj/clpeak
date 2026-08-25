#include <hip/hip_runtime.h>

// walk == 0 gives each warp 32 texels consecutive along x; walk == 1 gives it
// 32 texels consecutive along y, one row pitch apart.  Both cover every pixel
// exactly once (the map is a transpose), so the byte count is the same and the
// two rates are directly comparable.  The host races them and reports the
// faster -- see image_bandwidth.cpp for why neither alone is enough.
extern "C" __global__ void image_bandwidth(hipTextureObject_t tex, float *out,
                                           int width, int height, int walk)
{
    int gid   = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int gsize = (int)(gridDim.x * blockDim.x);
    int total = width * height;

    // Separate loops rather than a select on the coordinate, so the row-major
    // path stays exactly the shape it had before the race existed.
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (walk == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            int pixel = (gid + i * gsize) % total;
            float4 v = tex2D<float4>(tex, (float)(pixel % width), (float)(pixel / width));
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            int pixel = (gid + i * gsize) % total;
            float4 v = tex2D<float4>(tex, (float)(pixel / height), (float)(pixel % height));
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
    }
    out[gid] = sum.x + sum.y + sum.z + sum.w;
}
