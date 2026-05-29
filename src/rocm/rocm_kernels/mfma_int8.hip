#include <hip/hip_runtime.h>

// Raw MFMA (matrix-core) peak microbenchmark: int8 x int8 + int32, 16x16x32.
// CDNA3 wide-K int8 MFMA (gfx94x / gfx950+); on a part that lacks it the HIPRTC
// compile fails and the host reports Unsupported (no arch list, see
// mfma_fp16.hip). Each operand packs 8 int8 lanes into a 64-bit value.
// MFMA_ITERS * MFMA_ACC must match the host op count.
#define MFMA_ITERS 512
#define MFMA_ACC   8

typedef int int4v __attribute__((ext_vector_type(4)));

extern "C" __global__ void mfma_int8(int *out)
{
    // 8 packed int8 values per operand (each byte = 1); value is irrelevant
    // for throughput.
    const long a = 0x0101010101010101L;
    const long b = 0x0101010101010101L;

    int4v acc[MFMA_ACC];
    #pragma unroll
    for (int j = 0; j < MFMA_ACC; j++)
        acc[j] = int4v{0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < MFMA_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < MFMA_ACC; j++)
            acc[j] = __builtin_amdgcn_mfma_i32_16x16x32_i8(a, b, acc[j], 0, 0, 0);
    }

    int s = 0;
    #pragma unroll
    for (int j = 0; j < MFMA_ACC; j++)
        s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
