// FP8 (E4M3) tensor-core throughput via inline mma.sync PTX.
// Tile: m16n8k32 with fp32 accumulator.  Available on sm_89+ (Ada) and
// sm_90+ (Hopper); built into PTX ISA 8.0 (CUDA 12+).
//
// Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major):
//   A: m16 x k32 = 512 bytes / 32 threads = 16 bytes/thread = 4 x .b32
//   B: k32 x  n8 = 256 bytes / 32 threads =  8 bytes/thread = 2 x .b32
//   C/D: m16 x n8 = 128 fp32 / 32 threads = 4 fp32/thread per accumulator
//
// Four independent accumulator chains for ILP -- single-chain on RTX 5060
// hit ~83 TFLOPS, well below the consumer-Blackwell FP8 ceiling (which on
// this part appears tied to INT8 at ~165 TOPS).
//
// Per warp ops = 256 outer * 4 chains * (16*8*32*2) = 8,388,608;
// per thread = 262,144 (= 4 * COOPMAT_WORK_PER_WI).

extern "C" __global__ void wmma_fp8_e4m3(float *out, float A)
{
    unsigned int packed = 0x3c3c3c3cu ^ (__float_as_uint(A) & 0x0f0f0f0fu);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed;

    float c0_0 = 0.f, c0_1 = 0.f, c0_2 = 0.f, c0_3 = 0.f;
    float c1_0 = 0.f, c1_1 = 0.f, c1_2 = 0.f, c1_3 = 0.f;
    float c2_0 = 0.f, c2_1 = 0.f, c2_2 = 0.f, c2_3 = 0.f;
    float c3_0 = 0.f, c3_1 = 0.f, c3_2 = 0.f, c3_3 = 0.f;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c0_0), "+f"(c0_1), "+f"(c0_2), "+f"(c0_3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c1_0), "+f"(c1_1), "+f"(c1_2), "+f"(c1_3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c2_0), "+f"(c2_1), "+f"(c2_2), "+f"(c2_3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c3_0), "+f"(c3_1), "+f"(c3_2), "+f"(c3_3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    // Fold all 4 accumulators into c0 so the store reflects every chain.
    c0_0 += c1_0 + c2_0 + c3_0;
    c0_1 += c1_1 + c2_1 + c3_1;
    c0_2 += c1_2 + c2_2 + c3_2;
    c0_3 += c1_3 + c2_3 + c3_3;

    unsigned int base = blockIdx.x * 16u * 8u + threadIdx.x * 4u;
    out[base + 0] = c0_0;
    out[base + 1] = c0_1;
    out[base + 2] = c0_2;
    out[base + 3] = c0_3;
}
