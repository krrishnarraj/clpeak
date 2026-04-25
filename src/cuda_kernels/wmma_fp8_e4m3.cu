// FP8 (E4M3) tensor-core throughput via inline mma.sync PTX.
// Tile: m16n8k32 with fp32 accumulator.
//
// Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major):
//   A: m16 x k32 = 512 bytes / 32 threads = 16 bytes/thread = 4 x .b32
//   B: k32 x  n8 = 256 bytes / 32 threads =  8 bytes/thread = 2 x .b32
//   C/D: m16 x n8 = 128 fp32 / 32 threads = 4 fp32/thread
//
// Per-mma op count: 16 * 8 * 32 * 2 = 8192 ops/warp = 256 ops/thread.
// 256 iters → 65536 ops/thread (= COOPMAT_WORK_PER_WI).
//
// Available on sm_89+ (Ada) and sm_90+ (Hopper).  Built into PTX ISA 8.0
// (CUDA 12+); no driver header dependency beyond the inline asm.

extern "C" __global__ void wmma_fp8_e4m3(float *out, float A)
{
    // Pack four fp8 E4M3 values into each .b32 register.  Bit patterns just
    // need to round-trip through the FMA chain without producing NaN/Inf;
    // a value derived from A keeps the buffer write meaningful.
    unsigned int packed = 0x3c3c3c3cu ^ (__float_as_uint(A) & 0x0f0f0f0fu);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
            "r"(b0), "r"(b1));
    }

    // m16n8 output tile = 128 floats per warp = 4 floats per thread.
    unsigned int base = blockIdx.x * 16u * 8u + threadIdx.x * 4u;
    out[base + 0] = c0;
    out[base + 1] = c1;
    out[base + 2] = c2;
    out[base + 3] = c3;
}
