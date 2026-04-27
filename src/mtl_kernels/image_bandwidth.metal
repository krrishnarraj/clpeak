// Image (texture) bandwidth via Metal texture2d + sampler.  Each thread
// reads 16 RGBA float pixels with nearest-neighbour sampling.
//
// Bytes = IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads.

#include <metal_stdlib>
using namespace metal;

constexpr sampler nearest_clamp(coord::pixel,
                                address::clamp_to_edge,
                                filter::nearest);

kernel void image_bandwidth(texture2d<float, access::sample> img [[texture(0)]],
                            device float* out [[buffer(0)]],
                            uint tid [[thread_position_in_grid]],
                            uint gsize [[threads_per_grid]])
{
    int width  = (int)img.get_width();
    int height = (int)img.get_height();
    int total  = width * height;

    float4 sum = float4(0.0f);
    for (int i = 0; i < 16; i++)
    {
        int pixel = ((int)tid + i * (int)gsize) % total;
        float2 coord = float2((float)(pixel % width) + 0.5f,
                              (float)(pixel / width) + 0.5f);
        sum += img.sample(nearest_clamp, coord);
    }
    out[tid] = sum.x + sum.y + sum.z + sum.w;
}
