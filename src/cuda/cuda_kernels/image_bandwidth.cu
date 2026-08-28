// Image (texture) bandwidth via cudaTextureObject_t / tex2D fetch.
// Each thread reads 16 RGBA float pixels using nearest-neighbour
// sampling.  This measures the full CUDA texture-unit pipeline
// (coordinate → address → cache → data), which is architecturally
// distinct from raw buffer bandwidth.
//
// Bytes = IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads.

// walk == 0 gives each warp 32 texels consecutive along x; walk == 1 gives it
// 32 texels consecutive along y, one row pitch apart.  Both cover every pixel
// exactly once (the map is a transpose), so the byte count is the same and the
// two rates are directly comparable.  The pair is a --verbose-only probe of the
// image layout -- see image_bandwidth.cpp.
extern "C" __global__ void image_bandwidth(cudaTextureObject_t tex, float *out,
                                           int width, int height, int walk)
{
    int gid   = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int gsize = (int)(gridDim.x * blockDim.x);
    int total = width * height;

    // Separate loops rather than a select on the coordinate, so the row-major
    // path stays exactly the shape it had before the probe existed.
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (walk == 0)
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            int pixel = (gid + i * gsize) % total;
            float4 v = tex2D<float4>(tex, pixel % width, pixel / width);
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < 16; i++)
        {
            int pixel = (gid + i * gsize) % total;
            float4 v = tex2D<float4>(tex, pixel / height, pixel % height);
            sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
    }
    out[gid] = sum.x + sum.y + sum.z + sum.w;
}
