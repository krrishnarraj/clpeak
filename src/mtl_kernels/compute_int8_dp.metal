// INT8 dot-product throughput -- EMULATED.  MSL has no native dp4a-style
// intrinsic in 2026, so we hand-roll: unpack four signed int8 lanes from
// a packed int32, multiply-accumulate against another packed int32 into a
// scalar int accumulator.  Reported as int8_dp for cross-backend comparison
// but marked emulated="true" in result output so users don't mistake it for hardware
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

// int2-vectorised emulated dp4a.  Per-lane accounting matches the scalar
// kernel: 32 outer * 16 dots * 8 ops * 2 lanes = 8192 ops/thread.
inline int2 dp4_emul_v2(int2 a, int2 b, int2 acc)
{
    return int2(dp4_emul(a.x, b.x, acc.x),
                dp4_emul(a.y, b.y, acc.y));
}

#define STEP_V2(x, y, a)    a = dp4_emul_v2(x, y, a); y ^= a;
#define STEP_V2_4(x, y, a)  STEP_V2(x, y, a) STEP_V2(x, y, a) STEP_V2(x, y, a) STEP_V2(x, y, a)
#define STEP_V2_16(x, y, a) STEP_V2_4(x, y, a) STEP_V2_4(x, y, a) STEP_V2_4(x, y, a) STEP_V2_4(x, y, a)

kernel void compute_int8_dp2(device int* out [[buffer(0)]],
                             constant int& A [[buffer(1)]],
                             uint tid [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]])
{
    int packed = (A & 0xff)
               | (((A + 1) & 0xff) << 8)
               | (((A + 2) & 0xff) << 16)
               | (((A + 3) & 0xff) << 24);
    int2 x = int2(packed, packed);
    int2 y = int2((int)lid, (int)lid);
    int2 a = int2((int)lid, (int)lid);

    for (int i = 0; i < 32; i++)
    {
        STEP_V2_16(x, y, a)
    }

    out[tid] = a.x + a.y;
}

inline int4 dp4_emul_v4(int4 a, int4 b, int4 acc)
{
    return int4(dp4_emul(a.x, b.x, acc.x),
                dp4_emul(a.y, b.y, acc.y),
                dp4_emul(a.z, b.z, acc.z),
                dp4_emul(a.w, b.w, acc.w));
}

#define STEP_V4(x, y, a)    a = dp4_emul_v4(x, y, a); y ^= a;
#define STEP_V4_4(x, y, a)  STEP_V4(x, y, a) STEP_V4(x, y, a) STEP_V4(x, y, a) STEP_V4(x, y, a)
#define STEP_V4_16(x, y, a) STEP_V4_4(x, y, a) STEP_V4_4(x, y, a) STEP_V4_4(x, y, a) STEP_V4_4(x, y, a)

// 16 outer * 16 dots * 8 ops * 4 lanes = 8192 ops/thread.
kernel void compute_int8_dp4(device int* out [[buffer(0)]],
                             constant int& A [[buffer(1)]],
                             uint tid [[thread_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]])
{
    int packed = (A & 0xff)
               | (((A + 1) & 0xff) << 8)
               | (((A + 2) & 0xff) << 16)
               | (((A + 3) & 0xff) << 24);
    int4 x = int4(packed);
    int4 y = int4((int)lid);
    int4 a = int4((int)lid);

    for (int i = 0; i < 16; i++)
    {
        STEP_V4_16(x, y, a)
    }

    out[tid] = a.x + a.y + a.z + a.w;
}
