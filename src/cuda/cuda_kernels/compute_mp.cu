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
//
// Chain shape and why: see the MAD chain block in include/common/common.h.
// c is seeded through __half once and never updated, so it stays
// fp16-representable without a per-iteration round trip.

#include <cuda_fp16.h>

#define MAD_4(x, c)  x = fmaf(x, x, c); x = fmaf(x, x, c); x = fmaf(x, x, c); x = fmaf(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

extern "C" __global__ void compute_mp(float *out, float A)
{
    // Roundtrip through __half once at init + once per outer iter to force
    // the fp16 type into the data path while keeping the inner loop FFMA-only.
    float x = __half2float(__float2half(A));
    float c = __half2float(__float2half((float)threadIdx.x));

    #pragma unroll
    for (int i = 0; i < 128; i++)
    {
        MAD_16(x, c)
        x = __half2float(__float2half(x));
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
