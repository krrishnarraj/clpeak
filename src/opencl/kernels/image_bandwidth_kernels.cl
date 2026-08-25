MSTRINGIFY(

// Image (texture) bandwidth: each work-item reads 16 float4 pixels from a
// 2D RGBA-float image using integer coordinates and nearest-neighbour sampling.
// Reads are distributed across the image by striding with the global work size
// and wrapping within image bounds, so results are correct regardless of image
// size relative to the grid. The accumulated sum is written to global memory to
// prevent dead-code elimination by the compiler.
// walk == 0 gives each warp 32 texels consecutive along x; walk == 1 gives it
// 32 texels consecutive along y, one row pitch apart.  Both cover every pixel
// exactly once (the map is a transpose), so the byte count is the same and the
// two rates are directly comparable.  The host races them and reports the
// faster -- see image_bandwidth.cpp for why neither alone is enough.
__kernel void image_bandwidth_v1(__read_only image2d_t img, __global float* output, int walk)
{
    int gid   = (int)get_global_id(0);
    int gsize = (int)get_global_size(0);
    int width  = get_image_width(img);
    int height = get_image_height(img);
    int total  = width * height;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                        CLK_ADDRESS_CLAMP_TO_EDGE   |
                        CLK_FILTER_NEAREST;

    // Separate loops rather than a select on the coordinate, so the row-major
    // path stays exactly the shape it had before the race existed.
    float4 sum = (float4)(0.0f);
    if (walk == 0) {
        for (int i = 0; i < 16; i++) {
            int pixel  = (gid + i * gsize) % total;
            sum += read_imagef(img, sampler, (int2)(pixel % width, pixel / width));
        }
    } else {
        for (int i = 0; i < 16; i++) {
            int pixel  = (gid + i * gsize) % total;
            sum += read_imagef(img, sampler, (int2)(pixel / height, pixel % height));
        }
    }
    output[gid] = sum.x + sum.y + sum.z + sum.w;
}

)
