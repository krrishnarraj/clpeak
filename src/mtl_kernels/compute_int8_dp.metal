// INT8 dot-product throughput -- EMULATED.  MSL has no native dp4a-style
// intrinsic in 2026, so we hand-roll: unpack four signed int8 lanes from
// a packed int32, multiply-accumulate against another packed int32 into a
// scalar int accumulator.  Reported as int8_dp for cross-backend comparison
// but tagged emulated="true" in XML so users don't mistake it for hardware
// peak.
//
// Each emulated dot4: 4 int8 multiplies + 3 adds + xor feedback ~ 8 ops.
// 64 outer iters * 16 dots * 8 ops = 8192 ops/thread (= COMPUTE_INT8_DP_WORK_PER_WI).

#include <metal_stdlib>
using namespace metal;

inline int dp4_emul(int packed_a, int packed_b, int acc)
{
    int a0 = (packed_a << 24) >> 24;
    int a1 = (packed_a << 16) >> 24;
    int a2 = (packed_a <<  8) >> 24;
    int a3 = (packed_a)       >> 24;
    int b0 = (packed_b << 24) >> 24;
    int b1 = (packed_b << 16) >> 24;
    int b2 = (packed_b <<  8) >> 24;
    int b3 = (packed_b)       >> 24;
    return acc + a0*b0 + a1*b1 + a2*b2 + a3*b3;
}

#define STEP(x, y, a)    a = dp4_emul(x, y, a); y ^= a;
#define STEP_4(x, y, a)  STEP(x, y, a) STEP(x, y, a) STEP(x, y, a) STEP(x, y, a)
#define STEP_16(x, y, a) STEP_4(x, y, a) STEP_4(x, y, a) STEP_4(x, y, a) STEP_4(x, y, a)

kernel void compute_int8_dp(device int* out [[buffer(0)]],
                            constant int& A [[buffer(1)]],
                            uint tid [[thread_position_in_grid]],
                            uint lid [[thread_position_in_threadgroup]])
{
    int x = (A & 0xff)
          | (((A + 1) & 0xff) << 8)
          | (((A + 2) & 0xff) << 16)
          | (((A + 3) & 0xff) << 24);
    int y = (int)lid;
    int a = (int)lid;

    for (int i = 0; i < 64; i++)
    {
        STEP_16(x, y, a)
    }

    out[tid] = a;
}
