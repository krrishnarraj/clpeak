#include <hip/hip_runtime.h>

// Raw MFMA (matrix-core) peak microbenchmark: fp16 x fp16 + fp32, 16x16x16.
// Keeps MFMA_ACC independent accumulators in registers and issues a tight
// unrolled chain with no memory traffic, exposing peak matrix-core throughput.
// MFMA_ITERS * MFMA_ACC must match the host-side op count in mfma.cpp.
//
// No arch #ifdef guard on purpose: HIPRTC compiles this for the real device
// arch, and clang gates __builtin_amdgcn_mfma_* per sub-target. If the GPU
// lacks the instruction the compile fails and the host reports it Unsupported,
// so future CDNA parts are picked up automatically without editing any list.
#define MFMA_ITERS 512
#define MFMA_ACC   8

typedef __fp16 half4   __attribute__((ext_vector_type(4)));
typedef float  float4v __attribute__((ext_vector_type(4)));

extern "C" __global__ void mfma_fp16(float *out)
{
    half4 a = {(__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1};
    half4 b = {(__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1};

    float4v acc[MFMA_ACC];
    #pragma unroll
    for (int j = 0; j < MFMA_ACC; j++)
        acc[j] = float4v{0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < MFMA_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < MFMA_ACC; j++)
            acc[j] = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc[j], 0, 0, 0);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < MFMA_ACC; j++)
        s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
