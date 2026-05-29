#include <hip/hip_runtime.h>

// Raw MFMA (matrix-core) peak microbenchmark: bf16 x bf16 + fp32, 16x16x16.
// Uses the CDNA "_1k" bf16 MFMA (gfx90a and newer); on a CDNA part that lacks
// it the HIPRTC compile fails and the host reports Unsupported. See
// mfma_fp16.hip for the accumulator-chain and no-#ifdef rationale.
// MFMA_ITERS * MFMA_ACC must match mfma.cpp.
#define MFMA_ITERS 512
#define MFMA_ACC   8

typedef __bf16 bf16x4  __attribute__((ext_vector_type(4)));
typedef float  float4v __attribute__((ext_vector_type(4)));

extern "C" __global__ void mfma_bf16(float *out)
{
    bf16x4 a = {(__bf16)1, (__bf16)1, (__bf16)1, (__bf16)1};
    bf16x4 b = {(__bf16)1, (__bf16)1, (__bf16)1, (__bf16)1};

    float4v acc[MFMA_ACC];
    #pragma unroll
    for (int j = 0; j < MFMA_ACC; j++)
        acc[j] = float4v{0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < MFMA_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < MFMA_ACC; j++)
            acc[j] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc[j], 0, 0, 0);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < MFMA_ACC; j++)
        s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
