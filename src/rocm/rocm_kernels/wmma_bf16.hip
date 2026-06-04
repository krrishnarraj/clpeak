#include <hip/hip_runtime.h>

// Raw WMMA (RDNA matrix-core) peak microbenchmark: bf16 x bf16 + fp32,
// 16x16x16, wave32. RDNA analogue of the Vulkan cooperative-matrix bf16 path.
// See wmma_fp16.hip for the gfx11/gfx12 fragment-width rationale and the
// degrade-to-Unsupported guarantee. The bf16 WMMA builtin takes the operands as
// i16 (short) vectors holding the bf16 bit pattern (0x3F80 == 1.0f); the value
// is irrelevant for throughput.
//
// WMMA_ITERS * WMMA_ACC must match the host-side op count in wmma.cpp.
#define WMMA_ITERS 512
#define WMMA_ACC   8

typedef float float8v __attribute__((ext_vector_type(8)));

#if defined(__gfx1200__) || defined(__gfx1201__)
typedef short shortN __attribute__((ext_vector_type(8)));
#define WMMA_AB_ONES \
  shortN { (short)0x3F80, (short)0x3F80, (short)0x3F80, (short)0x3F80, \
           (short)0x3F80, (short)0x3F80, (short)0x3F80, (short)0x3F80 }
#define WMMA_BF16(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c)
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
      defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
      defined(__gfx1152__)
typedef short shortN __attribute__((ext_vector_type(16)));
#define WMMA_AB_ONES \
  shortN { (short)0x3F80, (short)0x3F80, (short)0x3F80, (short)0x3F80, \
           (short)0x3F80, (short)0x3F80, (short)0x3F80, (short)0x3F80, \
           (short)0x3F80, (short)0x3F80, (short)0x3F80, (short)0x3F80, \
           (short)0x3F80, (short)0x3F80, (short)0x3F80, (short)0x3F80 }
#define WMMA_BF16(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c)
#else
#error "wmma_bf16: no WMMA builtin for this architecture"
#endif

extern "C" __global__ void wmma_bf16(float *out)
{
    shortN a = WMMA_AB_ONES;
    shortN b = WMMA_AB_ONES;

    float8v acc[WMMA_ACC];
    #pragma unroll
    for (int j = 0; j < WMMA_ACC; j++)
        acc[j] = float8v{0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < WMMA_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < WMMA_ACC; j++)
            acc[j] = WMMA_BF16(a, b, acc[j]);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < WMMA_ACC; j++)
        #pragma unroll
        for (int k = 0; k < 8; k++)
            s += acc[j][k];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
