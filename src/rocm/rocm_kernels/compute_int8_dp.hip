#include <hip/hip_runtime.h>

// INT8 dot-product throughput via the AMD DP4a instruction. Each call computes
// dot(char4, char4) into an int32 accumulator: 4 INT8 multiply-adds = 8 INT8
// ops. This is the vector shader-core INT8 path, distinct from the matrix-core
// MFMA int8 (mfma_int8).
//
// The builtin differs by arch: GCN/CDNA (including gfx94x/gfx950) and RDNA1-2
// expose the signed v_dot4_i32_i8 (__builtin_amdgcn_sdot4, gated by the
// dot1-insts feature). RDNA3/4 (gfx11xx/gfx12xx) dropped dot1-insts and
// instead carry the mixed-sign v_dot4_i32_iu8 (__builtin_amdgcn_sudot4, gated
// by dot8-insts -- a feature the CDNA line never had, so sudot4 fails to
// compile on gfx94x and sdot4 fails on gfx12 with the "needs target feature
// dot1-insts" error). Select per-arch via clang's predefined __gfxNNNN__
// processor macros; signed*signed maps to sudot4 with both sign flags set.
// Arches not listed fall back to sdot4 (unchanged behavior); if a part has
// neither, the compile fails and the test reports compile/load failed.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) ||    \
    defined(__gfx1103__) ||                                                    \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) ||    \
    defined(__gfx1200__) || defined(__gfx1201__)
#define DOT4_I8(x, y, a) __builtin_amdgcn_sudot4(true, (x), true, (y), (a), false)
#else
#define DOT4_I8(x, y, a) __builtin_amdgcn_sdot4((x), (y), (a), false)
#endif
//
// Four variants scale instruction-level parallelism (independent accumulator
// chains).
//
// The chain shape. Three constraints have to hold at once, and each one has
// already produced a wrong reading in some backend:
//
//  - Both multiplicands may not be loop-invariant. a = DOT4_I8(x, y, a) with x
//    and y both fixed is a + n*dot(x, y), which a compiler may and does
//    strength-reduce; the OpenCL backend shipped that shape, and in Vulkan it
//    read 74939 GOPS on an RTX 5060 whose dp4a peak, measured by CUDA's own
//    __dp4a on the same card, is 33928.
//
//  - Nothing may run between the dots. This kernel used to keep an operand
//    moving by rewriting it from the accumulator (y ^= a), but that XOR is a
//    second dependent integer op per dot and the op budget credits none of it,
//    so the reading came out as the v_dot4 rate divided by however much issue
//    the extra op took. On an Arc A380 the same mistake cost more than half
//    the rate (8832 GOPS against 19497), which makes every number this kernel
//    has reported a floor rather than the part's real v_dot4 ceiling.
//
//  - All three source operands must be distinct registers. Intel Alchemist
//    halves a three-source op that reads the same register twice, so
//    a = DOT4_I8(x, a, a) is not the answer either. AMD does not have that
//    restriction, but the kernels are kept the same shape across backends so
//    the rows stay comparable.
//
// What satisfies all three: two accumulators feeding each other. Each dot
// reads {x, the other accumulator, its own}, three distinct registers, and
// writes its own. Every dot depends on the one before it, so a pair is one
// dependent chain, not two; and because the dot extracts the bytes of a value
// that is itself a 32-bit accumulator, the recurrence is not affine and has no
// closed form to fold to. dp2/dp4/dp8 run 2/4/8 independent copies of the pair.
//
// Op accounting is 8192 ops/thread for every variant so the numbers are
// directly comparable (matches COMPUTE_INT8_DP_WORK_PER_WI): 1024 v_dot4 calls
// per thread, each 4 INT8 multiply-adds = 8 ops. One STEP2 is two dots, so each
// variant issues 512 of them: dp = 64 iters * 8, dp2 = 64 * 4 per chain * 2
// chains, dp4 = 64 * 2 * 4, dp8 = 64 * 1 * 8. Chain k holds accumulators a<k>
// and b<k>, seeded 4 apart from every other accumulator in the thread: chains
// that start on the same value stay bitwise equal forever and a compiler is
// free to keep only one of them.

// Two dots, one dependent chain, no third instruction.
#define STEP2(x, p, q)    p = DOT4_I8(x, q, p); q = DOT4_I8(x, p, q);
#define STEP2_2(x, p, q)  STEP2(x, p, q) STEP2(x, p, q)
#define STEP2_4(x, p, q)  STEP2_2(x, p, q) STEP2_2(x, p, q)
#define STEP2_8(x, p, q)  STEP2_4(x, p, q) STEP2_4(x, p, q)

extern "C" __global__ void compute_int8_dp(int *out, int A)
{
    int x = (A & 0xff)
          | (((A + 1) & 0xff) << 8)
          | (((A + 2) & 0xff) << 16)
          | (((A + 3) & 0xff) << 24);
    int tid = (int)threadIdx.x;
    int a0 = tid, b0 = tid + 4;

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        STEP2_8(x, a0, b0)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + b0;
}

extern "C" __global__ void compute_int8_dp2(int *out, int A)
{
    int x  = (A & 0xff) | (((A+1)&0xff)<<8) | (((A+2)&0xff)<<16) | (((A+3)&0xff)<<24);
    int tid = (int)threadIdx.x;
    int a0 = tid,     b0 = tid + 4;
    int a1 = tid + 8, b1 = tid + 12;

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        STEP2(x, a0, b0) STEP2(x, a1, b1)
        STEP2(x, a0, b0) STEP2(x, a1, b1)
        STEP2(x, a0, b0) STEP2(x, a1, b1)
        STEP2(x, a0, b0) STEP2(x, a1, b1)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = (a0 + b0) + (a1 + b1);
}

extern "C" __global__ void compute_int8_dp4(int *out, int A)
{
    int x  = (A & 0xff) | (((A+1)&0xff)<<8) | (((A+2)&0xff)<<16) | (((A+3)&0xff)<<24);
    int tid = (int)threadIdx.x;
    int a0 = tid + 0,  b0 = tid + 4;
    int a1 = tid + 8,  b1 = tid + 12;
    int a2 = tid + 16, b2 = tid + 20;
    int a3 = tid + 24, b3 = tid + 28;

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        STEP2(x, a0, b0) STEP2(x, a1, b1) STEP2(x, a2, b2) STEP2(x, a3, b3)
        STEP2(x, a0, b0) STEP2(x, a1, b1) STEP2(x, a2, b2) STEP2(x, a3, b3)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] =
        ((a0 + b0) + (a1 + b1)) + ((a2 + b2) + (a3 + b3));
}

extern "C" __global__ void compute_int8_dp8(int *out, int A)
{
    int x  = (A & 0xff) | (((A+1)&0xff)<<8) | (((A+2)&0xff)<<16) | (((A+3)&0xff)<<24);
    int tid = (int)threadIdx.x;
    int a0 = tid + 0,  b0 = tid + 4;
    int a1 = tid + 8,  b1 = tid + 12;
    int a2 = tid + 16, b2 = tid + 20;
    int a3 = tid + 24, b3 = tid + 28;
    int a4 = tid + 32, b4 = tid + 36;
    int a5 = tid + 40, b5 = tid + 44;
    int a6 = tid + 48, b6 = tid + 52;
    int a7 = tid + 56, b7 = tid + 60;

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        STEP2(x, a0, b0) STEP2(x, a1, b1) STEP2(x, a2, b2) STEP2(x, a3, b3)
        STEP2(x, a4, b4) STEP2(x, a5, b5) STEP2(x, a6, b6) STEP2(x, a7, b7)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] =
        (((a0 + b0) + (a1 + b1)) + ((a2 + b2) + (a3 + b3)))
      + (((a4 + b4) + (a5 + b5)) + ((a6 + b6) + (a7 + b7)));
}
