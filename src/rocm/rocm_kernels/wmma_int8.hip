#include <hip/hip_runtime.h>

// Raw WMMA (RDNA matrix-core / "AI accelerator") peak microbenchmark:
// int8 x int8 + int32, 16x16x16, wave32. RDNA analogue of mfma_int8 and of the
// Vulkan cooperative-matrix int8 path; targets the datasheet INT8-matrix peak
// (e.g. 389 TOPS on RX 9070 XT / gfx1201). See wmma_fp16.hip for the gfx11 vs
// gfx12 fragment-width / builtin-suffix rationale and the degrade-to-Unsupported
// guarantee. iu8 is the mixed-sign builtin; both sign flags set => signed*signed.
//
// WMMA_ITERS * WMMA_ACC must match the host-side op count in wmma.cpp.
#define WMMA_ITERS 512
#define WMMA_ACC   8

typedef int int8v __attribute__((ext_vector_type(8)));

#if defined(__gfx1200__) || defined(__gfx1201__)
typedef int intN __attribute__((ext_vector_type(2)));
#define WMMA_AB_ONES intN { 0x01010101, 0x01010101 }
#define WMMA_I8(a, b, c) \
  __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false)
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
      defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
      defined(__gfx1152__)
typedef int intN __attribute__((ext_vector_type(4)));
#define WMMA_AB_ONES intN { 0x01010101, 0x01010101, 0x01010101, 0x01010101 }
#define WMMA_I8(a, b, c) \
  __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false)
#else
#error "wmma_int8: no WMMA builtin for this architecture"
#endif

extern "C" __global__ void wmma_int8(int *out)
{
    intN a = WMMA_AB_ONES;
    intN b = WMMA_AB_ONES;

    int8v acc[WMMA_ACC];
    #pragma unroll
    for (int j = 0; j < WMMA_ACC; j++)
        acc[j] = int8v{0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < WMMA_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < WMMA_ACC; j++)
            acc[j] = WMMA_I8(a, b, acc[j]);
    }

    int s = 0;
    #pragma unroll
    for (int j = 0; j < WMMA_ACC; j++)
        #pragma unroll
        for (int k = 0; k < 8; k++)
            s += acc[j][k];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
