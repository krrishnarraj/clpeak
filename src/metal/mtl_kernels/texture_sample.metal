// Texture SAMPLE-RATE (bilinear filtered fetches/second), not bandwidth:
// the image_bandwidth kernels count bytes moved with nearest filtering; these
// count filtered texel fetches against the TMUs.  The texture is small enough
// to stay cache/SLC-resident, so the filter units -- not DRAM -- are the
// limiter.  Every coordinate lands mid-texel-quad (fractional .4/.6) so the
// hardware performs a real 4-texel weighted blend on every sample.
//
// Samples = TEX_SAMPLES_PER_WI * globalThreads per dispatch.

#include <metal_stdlib>
using namespace metal;

// coord::pixel samplers only allow clamp addressing; coords stay in-range
// (pixel % width) so the mode never actually triggers.
constexpr sampler bilinear_wrap(coord::pixel,
                                address::clamp_to_edge,
                                filter::linear);

// The texture is a power-of-two square (host side passes 1024x1024) so the
// per-sample address math is one mask + one shift -- an integer %/÷ here
// would throttle the loop below the TMU rate and measure the ALUs instead.

// float-channel (RGBA8Unorm reads as unorm->float).
kernel void texture_sample_rgba8(texture2d<float, access::sample> img [[texture(0)]],
                                 device float* out [[buffer(0)]],
                                 uint tid [[thread_position_in_grid]],
                                 uint gsize [[threads_per_grid]])
{
    uint width = img.get_width();               // power of two
    uint wmask = width - 1;
    uint shift = ctz(width);
    uint tmask = width * img.get_height() - 1;  // total is power of two too

    float4 sum = float4(0.0f);
    for (uint i = 0; i < 64; i++)
    {
        uint pixel = (tid + i * gsize) & tmask;
        float2 coord = float2((float)(pixel & wmask) + 0.4f,
                              (float)(pixel >> shift) + 0.6f);
        sum += img.sample(bilinear_wrap, coord);
    }
    out[tid] = sum.x + sum.y + sum.z + sum.w;
}

// half-channel (RGBA16Float).
kernel void texture_sample_rgba16f(texture2d<half, access::sample> img [[texture(0)]],
                                   device float* out [[buffer(0)]],
                                   uint tid [[thread_position_in_grid]],
                                   uint gsize [[threads_per_grid]])
{
    uint width = img.get_width();
    uint wmask = width - 1;
    uint shift = ctz(width);
    uint tmask = width * img.get_height() - 1;

    half4 sum = half4(0.0h);
    for (uint i = 0; i < 64; i++)
    {
        uint pixel = (tid + i * gsize) & tmask;
        float2 coord = float2((float)(pixel & wmask) + 0.4f,
                              (float)(pixel >> shift) + 0.6f);
        sum += img.sample(bilinear_wrap, coord);
    }
    out[tid] = (float)(sum.x + sum.y + sum.z + sum.w);
}
