// Image (texture) bandwidth via Metal texture2d + sampler.  Each thread
// reads 16 RGBA float pixels with nearest-neighbour sampling.
//
// Bytes = IMAGE_FETCH_PER_WI * 4 * sizeof(float) * globalThreads.

#include <metal_stdlib>
using namespace metal;

constexpr sampler nearest_clamp(coord::pixel,
                                address::clamp_to_edge,
                                filter::nearest);

// Float-channel float4 sample (RGBA32Float, RGBA8Unorm).  texture2d<float>
// in MSL covers 32-bit-float and unorm/snorm formats with implicit
// conversion to float on read.
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

// half-channel sample (RGBA16Float).  Half accumulator keeps the load
// width matching the texture's native fp16 lane width.
kernel void image_bandwidth_half4(texture2d<half, access::sample> img [[texture(0)]],
                                  device float* out [[buffer(0)]],
                                  uint tid [[thread_position_in_grid]],
                                  uint gsize [[threads_per_grid]])
{
    int width  = (int)img.get_width();
    int height = (int)img.get_height();
    int total  = width * height;

    half4 sum = half4(0.0h);
    for (int i = 0; i < 16; i++)
    {
        int pixel = ((int)tid + i * (int)gsize) % total;
        float2 coord = float2((float)(pixel % width) + 0.5f,
                              (float)(pixel / width) + 0.5f);
        sum += img.sample(nearest_clamp, coord);
    }
    out[tid] = (float)(sum.x + sum.y + sum.z + sum.w);
}

// Single-channel float (R32Float).  The sample returns a float4 with .x
// populated, so we reduce only that lane.
kernel void image_bandwidth_r32f(texture2d<float, access::sample> img [[texture(0)]],
                                 device float* out [[buffer(0)]],
                                 uint tid [[thread_position_in_grid]],
                                 uint gsize [[threads_per_grid]])
{
    int width  = (int)img.get_width();
    int height = (int)img.get_height();
    int total  = width * height;

    float sum = 0.0f;
    for (int i = 0; i < 16; i++)
    {
        int pixel = ((int)tid + i * (int)gsize) % total;
        float2 coord = float2((float)(pixel % width) + 0.5f,
                              (float)(pixel / width) + 0.5f);
        sum += img.sample(nearest_clamp, coord).x;
    }
    out[tid] = sum;
}
