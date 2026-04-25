// Mixed-precision MAC: half inputs, float accumulator.  Mirrors the
// Vulkan compute_mp_v1.comp and CUDA compute_mp.cu structure -- inner FMA
// chain stays in fp32 (the data path the shader-core actually uses for
// fp16xfp16+fp32) and the half cast happens once per outer iter to force
// the conversion units to participate.

#include <metal_stdlib>
using namespace metal;

#define MAD_4(x, y)  x = fma(y, x, y); y = fma(x, y, x); x = fma(y, x, y); y = fma(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

kernel void compute_mp(device float* out [[buffer(0)]],
                       constant float& A [[buffer(1)]],
                       uint tid [[thread_position_in_grid]],
                       uint lid [[thread_position_in_threadgroup]])
{
    float x = (float)((half)A);
    float y = (float)((half)lid);

    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, y)
        x = (float)((half)x);
        y = (float)((half)y);
    }

    out[tid] = x + y;
}
