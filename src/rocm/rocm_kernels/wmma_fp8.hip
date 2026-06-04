#include <hip/hip_runtime.h>

// Raw WMMA (RDNA matrix-core) peak microbenchmarks: fp8 x fp8 + fp32,
// 16x16x16, wave32. RDNA analogue of the Vulkan cooperative-matrix fp8 paths.
// AMD naming: fp8 == E4M3, bf8 == E5M2 -- two kernels, one per format. fp8 WMMA
// is RDNA4 (gfx12) only; RDNA3 (gfx11) has no fp8 matrix path, so the #error
// makes the HIPRTC compile fail there and the host reports Unsupported.
//
// Operands are packed as 8 fp8 lanes per i32 (2 x i32 = 16, matching K=16); the
// byte value is irrelevant for throughput (0x38 ~ 1.0 in E4M3, 0x3C ~ 1.0 in
// E5M2). WMMA_ITERS * WMMA_ACC must match the host-side op count in wmma.cpp.
#define WMMA_ITERS 512
#define WMMA_ACC   8

typedef float float8v __attribute__((ext_vector_type(8)));
typedef int   int2v   __attribute__((ext_vector_type(2)));

#if defined(__gfx1200__) || defined(__gfx1201__)
#define WMMA_FP8_E4M3(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c)
#define WMMA_FP8_E5M2(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_bf8_bf8_w32_gfx12(a, b, c)
#else
#error "wmma_fp8: fp8 WMMA requires RDNA4 (gfx12)"
#endif

extern "C" __global__ void wmma_fp8_e4m3(float *out)
{
    int2v a = int2v{0x38383838, 0x38383838};
    int2v b = int2v{0x38383838, 0x38383838};

    float8v acc[WMMA_ACC];
    #pragma unroll
    for (int j = 0; j < WMMA_ACC; j++)
        acc[j] = float8v{0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < WMMA_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < WMMA_ACC; j++)
            acc[j] = WMMA_FP8_E4M3(a, b, acc[j]);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < WMMA_ACC; j++)
        #pragma unroll
        for (int k = 0; k < 8; k++)
            s += acc[j][k];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

extern "C" __global__ void wmma_fp8_e5m2(float *out)
{
    int2v a = int2v{0x3C3C3C3C, 0x3C3C3C3C};
    int2v b = int2v{0x3C3C3C3C, 0x3C3C3C3C};

    float8v acc[WMMA_ACC];
    #pragma unroll
    for (int j = 0; j < WMMA_ACC; j++)
        acc[j] = float8v{0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < WMMA_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < WMMA_ACC; j++)
            acc[j] = WMMA_FP8_E5M2(a, b, acc[j]);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < WMMA_ACC; j++)
        #pragma unroll
        for (int k = 0; k < 8; k++)
            s += acc[j][k];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
