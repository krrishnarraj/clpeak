#include <hip/hip_runtime.h>

// 2:4 structured-sparse MFMA peak microbenchmark: fp8(e4m3), 16x16x64 (SMFMAC).
// Sparse fp8 matrix-core: 2:4-compressed A (v2i32 = 8 packed fp8 nonzeros)
// times dense B (v4i32 = 16 packed fp8), fp32 accumulate. Does the work of a
// dense 16x16x64 in dense-16x16x32 time -> ~2x, targeting the datasheet's
// "FP8 with Structured Sparsity" 5.22 PFLOPS column. See smfmac_fp16.hip.
// SMFMAC_ITERS * SMFMAC_ACC must match the host-side op count.
#define SMFMAC_ITERS 512
#define SMFMAC_ACC   8

typedef int   i32x2   __attribute__((ext_vector_type(2)));
typedef int   i32x4   __attribute__((ext_vector_type(4)));
typedef float float4v __attribute__((ext_vector_type(4)));

extern "C" __global__ void smfmac_fp8(float *out)
{
    // packed fp8 (e4m3 fnuz) lanes (0x38 ~ 1.0); value irrelevant for throughput.
    i32x2 a = {0x38383838, 0x38383838};
    i32x4 b = {0x38383838, 0x38383838, 0x38383838, 0x38383838};
    const int idx = 0;

    float4v acc[SMFMAC_ACC];
    #pragma unroll
    for (int j = 0; j < SMFMAC_ACC; j++)
        acc[j] = float4v{0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < SMFMAC_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < SMFMAC_ACC; j++)
            acc[j] = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a, b, acc[j], idx, 0, 0);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < SMFMAC_ACC; j++)
        s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
