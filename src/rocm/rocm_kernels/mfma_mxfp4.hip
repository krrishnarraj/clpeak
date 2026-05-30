#include <hip/hip_runtime.h>

// Raw scaled-MFMA (matrix-core) peak microbenchmark: mxfp4(e2m1) x mxfp4 + fp32,
// 16x16x128 with a block scale.  This is the CDNA4 (gfx950 / MI350) FP4 path; on
// a part that lacks the scaled-MFMA instruction the HIPRTC compile fails and the
// host reports Unsupported (no arch list, see mfma_fp8.hip / mfma.cpp).
//
// The f8f6f4 family selects the operand datatype via the cbsz/blgp format ids:
//   0=e4m3 1=e5m2 2=e3m2 3=e2m3 4=e2m1(fp4).  We use 4/4 for E2M1 x E2M1.
// Operands are <8 x i32> (256-bit) packed FP4; the block-scale operands are
// arbitrary for a throughput probe because all inputs are synthetic constants.
// MFMA_ITERS * MFMA_ACC must match the host-side op count.
#define MFMA_ITERS 512
#define MFMA_ACC   8

typedef float float4v __attribute__((ext_vector_type(4)));
typedef int   int8v   __attribute__((ext_vector_type(8)));

extern "C" __global__ void mfma_mxfp4(float *out)
{
    // Packed E2M1 (fp4) operands; value is irrelevant for throughput
    // (0x4 ~ 1.0 in e2m1, replicated across the 256-bit operand).
    const int8v a = {0x44444444, 0x44444444, 0x44444444, 0x44444444,
                     0x44444444, 0x44444444, 0x44444444, 0x44444444};
    const int8v b = a;

    // Block-scale data (UE8M0); 0x3f ~ 1.0. Selectors are 0 (first block).
    const int scaleA = 0x3f3f3f3f;
    const int scaleB = 0x3f3f3f3f;
    const int cbsz = 4; // matrix A format = e2m1 (fp4)
    const int blgp = 4; // matrix B format = e2m1 (fp4)

    float4v acc[MFMA_ACC];
    #pragma unroll
    for (int j = 0; j < MFMA_ACC; j++)
        acc[j] = float4v{0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < MFMA_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < MFMA_ACC; j++)
            acc[j] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a, b, acc[j], cbsz, blgp, 0, scaleA, 0, scaleB);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < MFMA_ACC; j++)
        s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
