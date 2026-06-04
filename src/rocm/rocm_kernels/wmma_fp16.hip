#include <hip/hip_runtime.h>

// Raw WMMA (RDNA matrix-core / "AI accelerator") peak microbenchmark:
// fp16 x fp16 + fp32, 16x16x16, wave32. This is the RDNA analogue of the CDNA
// MFMA test (mfma_fp16) and of the Vulkan cooperative-matrix fp16 path -- it
// drives the WMMA units directly via __builtin_amdgcn_wmma_* and targets the
// datasheet matrix-FP16 peak (e.g. 195 TFLOPS on RX 9070 XT / gfx1201).
//
// RDNA3 (gfx11) and RDNA4 (gfx12) both expose WMMA, but the per-lane A/B
// fragment width differs (gfx11 replicates the K=16 column as 16 halves/lane;
// gfx12 packs 8) and the gfx12 builtin carries a _gfx12 suffix. Select by
// clang's predefined __gfxNNNN__ processor macro. A part without WMMA never
// reaches here (the host gates on the RDNA family) but if it did, the #error /
// missing builtin makes HIPRTC fail and the host reports Unsupported -- never a
// fabricated number. The op count below (2*M*N*K per call) is independent of
// the fragment layout, so a successful run is physically meaningful.
//
// WMMA_ITERS * WMMA_ACC must match the host-side op count in wmma.cpp.
#define WMMA_ITERS 512
#define WMMA_ACC   8

typedef float  float8v __attribute__((ext_vector_type(8)));

#if defined(__gfx1200__) || defined(__gfx1201__)
typedef __fp16 halfN __attribute__((ext_vector_type(8)));
#define WMMA_AB_ONES \
  halfN { (__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1, \
          (__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1 }
#define WMMA_F16(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c)
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || \
      defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || \
      defined(__gfx1152__)
typedef __fp16 halfN __attribute__((ext_vector_type(16)));
#define WMMA_AB_ONES \
  halfN { (__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1, \
          (__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1, \
          (__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1, \
          (__fp16)1, (__fp16)1, (__fp16)1, (__fp16)1 }
#define WMMA_F16(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)
#else
#error "wmma_fp16: no WMMA builtin for this architecture"
#endif

extern "C" __global__ void wmma_fp16(float *out)
{
    halfN a = WMMA_AB_ONES;
    halfN b = WMMA_AB_ONES;

    float8v acc[WMMA_ACC];
    #pragma unroll
    for (int j = 0; j < WMMA_ACC; j++)
        acc[j] = float8v{0, 0, 0, 0, 0, 0, 0, 0};

    #pragma unroll 1
    for (int i = 0; i < WMMA_ITERS; i++)
    {
        #pragma unroll
        for (int j = 0; j < WMMA_ACC; j++)
            acc[j] = WMMA_F16(a, b, acc[j]);
    }

    float s = 0;
    #pragma unroll
    for (int j = 0; j < WMMA_ACC; j++)
        #pragma unroll
        for (int k = 0; k < 8; k++)
            s += acc[j][k];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
