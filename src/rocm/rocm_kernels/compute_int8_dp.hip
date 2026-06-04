#include <hip/hip_runtime.h>

// INT8 dot-product throughput via the AMD DP4a instruction. Each call computes
// dot(char4, char4) into an int32 accumulator: 4 INT8 multiply-adds = 8 INT8
// ops. This is the vector shader-core INT8 path, distinct from the matrix-core
// MFMA int8 (mfma_int8).
//
// The builtin differs by arch: older GCN/CDNA1-2 and RDNA1-2 expose the
// signed v_dot4_i32_i8 (__builtin_amdgcn_sdot4, gated by the dot1-insts
// feature); CDNA3 (gfx94x/gfx950) and RDNA3/4 (gfx11xx/gfx12xx) replaced it
// with the mixed-sign v_dot4_i32_iu8 (__builtin_amdgcn_sudot4) and no longer
// carry dot1-insts -- so sdot4 fails to compile there (the gfx12 "needs target
// feature dot1-insts" error). Select per-arch via clang's predefined
// __gfxNNNN__ processor macros; signed*signed maps to sudot4 with both sign
// flags set. Arches not listed fall back to sdot4 (unchanged behavior); if a
// part has neither, the compile fails and the test reports compile/load failed.
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) ||       \
    defined(__gfx950__) ||                                                     \
    defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) ||    \
    defined(__gfx1103__) ||                                                    \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) ||    \
    defined(__gfx1200__) || defined(__gfx1201__)
#define DOT4_I8(x, y, a) __builtin_amdgcn_sudot4(true, (x), true, (y), (a), false)
#else
#define DOT4_I8(x, y, a) __builtin_amdgcn_sdot4((x), (y), (a), false)
#endif
//
// Four variants scale instruction-level parallelism (independent accumulator
// chains). Op accounting is 8192 ops/thread for every variant so the numbers
// are directly comparable (matches COMPUTE_INT8_DP_WORK_PER_WI):
//   dp  = 64 * 16 dots * 8 ops, dp2 = 32 * (2*16) * 8, dp4 = 16 * (4*16) * 8,
//   dp8 = 8 * (8*16) * 8 -- all 8192.

#define STEP(x, y, a)    a = DOT4_I8(x, y, a); y ^= a;
#define STEP_4(x, y, a)  STEP(x, y, a) STEP(x, y, a) STEP(x, y, a) STEP(x, y, a)
#define STEP_16(x, y, a) STEP_4(x, y, a) STEP_4(x, y, a) STEP_4(x, y, a) STEP_4(x, y, a)

extern "C" __global__ void compute_int8_dp(int *out, int A)
{
    int x = (A & 0xff)
          | (((A + 1) & 0xff) << 8)
          | (((A + 2) & 0xff) << 16)
          | (((A + 3) & 0xff) << 24);
    int y = (int)threadIdx.x;
    int a = (int)threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        STEP_16(x, y, a)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = a;
}

extern "C" __global__ void compute_int8_dp2(int *out, int A)
{
    int x  = (A & 0xff) | (((A+1)&0xff)<<8) | (((A+2)&0xff)<<16) | (((A+3)&0xff)<<24);
    int y0 = (int)threadIdx.x;
    int y1 = (int)threadIdx.x + 1;
    int a0 = (int)threadIdx.x;
    int a1 = (int)threadIdx.x + 7;

    #pragma unroll 1
    for (int i = 0; i < 32; i++)
    {
        STEP_16(x, y0, a0)
        STEP_16(x, y1, a1)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1;
}

extern "C" __global__ void compute_int8_dp4(int *out, int A)
{
    int x  = (A & 0xff) | (((A+1)&0xff)<<8) | (((A+2)&0xff)<<16) | (((A+3)&0xff)<<24);
    int y0 = (int)threadIdx.x;
    int y1 = (int)threadIdx.x + 1;
    int y2 = (int)threadIdx.x + 2;
    int y3 = (int)threadIdx.x + 3;
    int a0 = (int)threadIdx.x;
    int a1 = (int)threadIdx.x + 7;
    int a2 = (int)threadIdx.x + 13;
    int a3 = (int)threadIdx.x + 19;

    #pragma unroll 1
    for (int i = 0; i < 16; i++)
    {
        STEP_16(x, y0, a0)
        STEP_16(x, y1, a1)
        STEP_16(x, y2, a2)
        STEP_16(x, y3, a3)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3;
}

extern "C" __global__ void compute_int8_dp8(int *out, int A)
{
    int x  = (A & 0xff) | (((A+1)&0xff)<<8) | (((A+2)&0xff)<<16) | (((A+3)&0xff)<<24);
    int y0 = (int)threadIdx.x;
    int y1 = (int)threadIdx.x + 1;
    int y2 = (int)threadIdx.x + 2;
    int y3 = (int)threadIdx.x + 3;
    int y4 = (int)threadIdx.x + 4;
    int y5 = (int)threadIdx.x + 5;
    int y6 = (int)threadIdx.x + 6;
    int y7 = (int)threadIdx.x + 7;
    int a0 = (int)threadIdx.x;
    int a1 = (int)threadIdx.x + 11;
    int a2 = (int)threadIdx.x + 17;
    int a3 = (int)threadIdx.x + 23;
    int a4 = (int)threadIdx.x + 29;
    int a5 = (int)threadIdx.x + 31;
    int a6 = (int)threadIdx.x + 37;
    int a7 = (int)threadIdx.x + 41;

    #pragma unroll 1
    for (int i = 0; i < 8; i++)
    {
        STEP_16(x, y0, a0)
        STEP_16(x, y1, a1)
        STEP_16(x, y2, a2)
        STEP_16(x, y3, a3)
        STEP_16(x, y4, a4)
        STEP_16(x, y5, a5)
        STEP_16(x, y6, a6)
        STEP_16(x, y7, a7)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] =
        a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}
