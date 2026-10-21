MSTRINGIFY(

// Image (texture) bandwidth: each work-item reads 16 float4 pixels from a
// 2D RGBA-float image using integer coordinates and nearest-neighbour sampling.
// Reads stride by the global work size, so consecutive work-items touch
// consecutive pixels of the walk, and the accumulated sum is written to
// global memory to prevent dead-code elimination by the compiler.
//
// The walk shapes are the Vulkan backend's (shaders/image_bandwidth_v1.comp),
// raced by the host -- see the image-bandwidth block in
// include/common/common.h.  walk == 0 is the plain row-major sweep, giving
// consecutive work-items texels consecutive along x; walk == 1 transposes the
// whole image, giving them texels consecutive along y.  walk == 2 covers the
// image in TW x TH blocks, walked row-major and row-major within a block,
// handing each run of TW*TH consecutive work-items a 2D block, which is what
// a driver-swizzled image wants.  Every shape covers each pixel exactly once,
// so the byte count is identical and the rates are directly comparable.
//
// The extent and the block are powers of two, passed as log2, so every
// coordinate is a shift and a mask.  A divide or modulo by a run-time value is
// a long emulated sequence on GPUs without an integer divider -- Intel's among
// them -- and this kernel used to spend three on every fetch.  The row-major
// sweep is not spelled as a W x 1 block: with the shifts unknown until run
// time, the block arithmetic left over cost Intel's CPU runtime 3.5% there.
// No wrap on the pixel index either: the host sizes the dispatch so that
// global size * 16 <= W * H.
__kernel void image_bandwidth_v1(__read_only image2d_t img, __global float* output, int walk,
                                 int logW, int logH, int logTW, int logTH)
{
    int gid   = (int)get_global_id(0);
    int gsize = (int)get_global_size(0);

    sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                        CLK_ADDRESS_CLAMP_TO_EDGE   |
                        CLK_FILTER_NEAREST;

    // Separate loops rather than a select on the coordinate, so no walk
    // carries a branch another's shape would have to pay for.
    float4 sum = (float4)(0.0f);
    if (walk == 0) {
        for (int i = 0; i < 16; i++) {
            int pixel = gid + i * gsize;
            sum += read_imagef(img, sampler, (int2)(pixel & ((1 << logW) - 1), pixel >> logW));
        }
    } else if (walk == 2) {
        int logTile   = logTW + logTH;
        int logTilesX = logW - logTW;
        for (int i = 0; i < 16; i++) {
            int pixel = gid + i * gsize;
            int t  = pixel >> logTile;
            int it = pixel & ((1 << logTile) - 1);
            int x  = ((t & ((1 << logTilesX) - 1)) << logTW) + (it & ((1 << logTW) - 1));
            int y  = ((t >> logTilesX) << logTH) + (it >> logTW);
            sum += read_imagef(img, sampler, (int2)(x, y));
        }
    } else {
        for (int i = 0; i < 16; i++) {
            int pixel = gid + i * gsize;
            sum += read_imagef(img, sampler, (int2)(pixel >> logH, pixel & ((1 << logH) - 1)));
        }
    }
    output[gid] = sum.x + sum.y + sum.z + sum.w;
}

)
