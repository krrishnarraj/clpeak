MSTRINGIFY(

// Image (texture) bandwidth: each work-item reads 16 float4 pixels from a
// 2D RGBA-float image using integer coordinates and nearest-neighbour sampling.
// Reads are distributed across the image by striding with the global work size
// and wrapping within image bounds, so results are correct regardless of image
// size relative to the grid. The accumulated sum is written to global memory to
// prevent dead-code elimination by the compiler.
__kernel void image_bandwidth_v1(__read_only image2d_t img, __global float* output)
{
    int gid   = (int)get_global_id(0);
    int gsize = (int)get_global_size(0);
    int width  = get_image_width(img);
    int height = get_image_height(img);
    int total  = width * height;

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                        CLK_ADDRESS_CLAMP_TO_EDGE   |
                        CLK_FILTER_NEAREST;

    float4 sum = (float4)(0.0f);
    for (int i = 0; i < 16; i++) {
        int pixel  = (gid + i * gsize) % total;
        int2 coord = (int2)(pixel % width, pixel / width);
        sum += read_imagef(img, sampler, coord);
    }
    output[gid] = sum.x + sum.y + sum.z + sum.w;
}

)
