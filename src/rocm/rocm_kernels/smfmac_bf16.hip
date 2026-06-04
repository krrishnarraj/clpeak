#include <hip/hip_runtime.h>

// 2:4 structured-sparse MFMA peak microbenchmark: bf16, 16x16x32 (SMFMAC).
// See smfmac_fp16.hip for the sparse-instruction / op-accounting rationale.
// SMFMAC_ITERS * SMFMAC_ACC must match sparse_mfma.cpp.
#define SMFMAC_ITERS 512
#define SMFMAC_ACC   8

typedef __bf16 bf16x4  __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x8  __attribute__((ext_vector_type(8)));
typedef float  float4v __attribute__((ext_vector_type(4)));

extern "C" __global__ void smfmac_bf16(float *out)
{
    bf16x4 a = {(__bf16)1, (__bf16)1, (__bf16)1, (__bf16)1};
    bf16x8 b = {(__bf16)1, (__bf16)1, (__bf16)1, (__bf16)1,
                (__bf16)1, (__bf16)1, (__bf16)1, (__bf16)1};
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
            acc[j] = __builtin_amdgcn_smfmac_f32_16x16x32_bf16(a, b, acc[j], idx, 0, 0);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < SMFMAC_ACC; j++)
        s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
