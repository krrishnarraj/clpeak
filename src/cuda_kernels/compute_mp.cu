// Mixed-precision MAC: fp16 inputs, fp32 accumulator.  Mirrors
// src/shaders/compute_mp_v1.comp's structure -- the inner FMA chain is
// pure FP32 (because fp16xfp16+fp32 emits FFMA on NVIDIA shader cores at
// FP32 issue rate); the fp16 cast happens once per outer iter so the data
// path actually exercises the fp16 conversion units.
//
// Earlier sketch with __float2half / __half2float inside the FMA macro
// chained 2-3 conversions onto the critical path of every FMA -- that is
// exactly the lowering the Vulkan MP shader was rewritten to avoid (commit
// f6ea4c4); the same fix applies here.

#include <cuda_fp16.h>

#define MAD_4(x, y)  x = fmaf(y, x, y); y = fmaf(x, y, x); x = fmaf(y, x, y); y = fmaf(x, y, x);
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

extern "C" __global__ void compute_mp(float *out, float A)
{
    // Roundtrip through __half once at init + once per outer iter to force
    // the fp16 type into the data path while keeping the inner loop FFMA-only.
    float x = __half2float(__float2half(A));
    float y = __half2float(__float2half((float)threadIdx.x));

    #pragma unroll 1
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, y)
        x = __half2float(__float2half(x));
        y = __half2float(__float2half(y));
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}
