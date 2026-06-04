#include <hip/hip_runtime.h>

// Raw MFMA (matrix-core) peak microbenchmark: fp8(e4m3) x fp8 + fp32, 16x16x32.
// CDNA3 fp8 MFMA (gfx94x / gfx950+); on a part that lacks it the HIPRTC compile
// fails and the host reports Unsupported (no arch list, see mfma_fp16.hip).
// Each operand packs 8 fp8 lanes into a 64-bit value.
// MFMA_ITERS * MFMA_ACC must match the host-side op count.
#define MFMA_ITERS 512
#define MFMA_ACC   8

typedef float float4v __attribute__((ext_vector_type(4)));

extern "C" __global__ void mfma_fp8(float *out)
{
    // 8 packed fp8 (e4m3 fnuz) values per operand; value is irrelevant for
    // throughput (0x38 ~ 1.0 in e4m3).
    const long a = 0x3838383838383838L;
    const long b = 0x3838383838383838L;

    float4v acc[MFMA_ACC];
    #pragma unroll
    for (int j = 0; j < MFMA_ACC; j++)
        acc[j] = float4v{0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < MFMA_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < MFMA_ACC; j++)
            acc[j] = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc[j], 0, 0, 0);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < MFMA_ACC; j++)
        s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
