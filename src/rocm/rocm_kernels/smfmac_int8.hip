#include <hip/hip_runtime.h>

// 2:4 structured-sparse MFMA peak microbenchmark: int8, 16x16x64 (SMFMAC).
// Sparse int8 matrix-core: 2:4-compressed A (v2i32 = 8 packed int8 nonzeros)
// times dense B (v4i32 = 16 packed int8), int32 accumulate. Does the work of a
// dense 16x16x64 in dense-16x16x32 time -> ~2x, targeting the datasheet's
// "INT8 with Structured Sparsity" 5.22 POPS column. See smfmac_fp16.hip.
// SMFMAC_ITERS * SMFMAC_ACC must match the host-side op count.
#define SMFMAC_ITERS 512
#define SMFMAC_ACC   8

typedef int i32x2  __attribute__((ext_vector_type(2)));
typedef int i32x4  __attribute__((ext_vector_type(4)));

extern "C" __global__ void smfmac_int8(int *out)
{
    // packed int8 lanes (each byte = 1); value irrelevant for throughput.
    i32x2 a = {0x01010101, 0x01010101};
    i32x4 b = {0x01010101, 0x01010101, 0x01010101, 0x01010101};
    const int idx = 0;

    i32x4 acc[SMFMAC_ACC];
    #pragma unroll
    for (int j = 0; j < SMFMAC_ACC; j++)
        acc[j] = i32x4{0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < SMFMAC_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < SMFMAC_ACC; j++)
            acc[j] = __builtin_amdgcn_smfmac_i32_16x16x64_i8(a, b, acc[j], idx, 0, 0);
    }

    int s = 0;
    #pragma unroll
    for (int j = 0; j < SMFMAC_ACC; j++)
        s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
