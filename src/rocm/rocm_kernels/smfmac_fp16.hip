#include <hip/hip_runtime.h>

// 2:4 structured-sparse MFMA peak microbenchmark: fp16, 16x16x32 (SMFMAC).
// The sparse matrix-core instruction (__builtin_amdgcn_smfmac_*) processes a
// 2:4-compressed A operand against a dense B, doing the work of a dense
// 16x16x32 in the time of a dense 16x16x16 -- i.e. ~2x throughput, which is
// what the datasheet's "with Structured Sparsity" column reflects.
//
// A is compressed (4 fp16 nonzeros), B is full (8 fp16); the extra operand is
// the sparsity index (which of the 4 columns are nonzero) -- value irrelevant
// for throughput. No arch #ifdef: SMFMAC is CDNA3, and HIPRTC compile failure
// on a part that lacks it is reported Unsupported (see sparse_mfma.cpp).
// SMFMAC_ITERS * SMFMAC_ACC must match the host-side op count.
#define SMFMAC_ITERS 512
#define SMFMAC_ACC   8

typedef __fp16 half4   __attribute__((ext_vector_type(4)));
typedef __fp16 half8   __attribute__((ext_vector_type(8)));
typedef float  float4v __attribute__((ext_vector_type(4)));

extern "C" __global__ void smfmac_fp16(float *out)
{
    half4 a = {(__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1};
    half8 b = {(__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1,
               (__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1};
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
            acc[j] = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a, b, acc[j], idx, 0, 0);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < SMFMAC_ACC; j++)
        s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
