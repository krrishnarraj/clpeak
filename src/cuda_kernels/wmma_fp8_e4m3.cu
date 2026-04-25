// FP8 (E4M3) tensor-core throughput via inline mma.sync PTX.
//
// Tile: m16n8k64 with fp32 accumulator -- the Blackwell-native FP8 shape.
// Earlier m16n8k32 form (Ada/Hopper-era) plateaued at ~85 TFLOPS on RTX
// 5060 regardless of chain count or asm-block packing, suggesting it
// lowers to a half-rate SASS path on sm_100+.  The m16n8k64 instruction
// (PTX ISA 8.4+, introduced for sm_100 / Hopper) doubles the K dim, so
// each issue does 2x the work.  Requires sm_100+ to assemble; on older
// arches (Ada / Hopper-G first revisions) the runtime should fall back
// to the legacy m16n8k32 form -- TODO when we have such a device.
//
// Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major):
//   A: m16 x k64 = 1024 bytes / 32 threads = 32 bytes/thread = 8 x .b32
//   B: k64 x  n8 =  512 bytes / 32 threads = 16 bytes/thread = 4 x .b32
//   C/D: m16 x n8 = 128 fp32 / 32 threads = 4 fp32/thread per accumulator
//
// 4 independent accumulator chains, packed into one non-volatile asm block
// so ptxas can interleave the per-chain mma.sync issues.
//
// Per warp ops = 256 outer * 4 chains * (16*8*64*2) = 16,777,216;
// per thread = 524,288 (= 8 * COOPMAT_WORK_PER_WI).

extern "C" __global__ void wmma_fp8_e4m3(float *out, float A)
{
    unsigned int packed = 0x3c3c3c3cu ^ (__float_as_uint(A) & 0x0f0f0f0fu);
    // 8 A regs / 4 B regs per thread for m16n8k64.
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int a4 = packed, a5 = packed, a6 = packed, a7 = packed;
    unsigned int b0 = packed, b1 = packed, b2 = packed, b3 = packed;

    float c00=0,c01=0,c02=0,c03=0;
    float c10=0,c11=0,c12=0,c13=0;
    float c20=0,c21=0,c22=0,c23=0;
    float c30=0,c31=0,c32=0,c33=0;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm(
          "mma.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%0,%1,%2,%3}, "
              "{%16,%17,%18,%19,%20,%21,%22,%23}, "
              "{%24,%25,%26,%27}, "
              "{%0,%1,%2,%3};\n"
          "mma.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%4,%5,%6,%7}, "
              "{%16,%17,%18,%19,%20,%21,%22,%23}, "
              "{%24,%25,%26,%27}, "
              "{%4,%5,%6,%7};\n"
          "mma.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%8,%9,%10,%11}, "
              "{%16,%17,%18,%19,%20,%21,%22,%23}, "
              "{%24,%25,%26,%27}, "
              "{%8,%9,%10,%11};\n"
          "mma.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%12,%13,%14,%15}, "
              "{%16,%17,%18,%19,%20,%21,%22,%23}, "
              "{%24,%25,%26,%27}, "
              "{%12,%13,%14,%15};\n"
          : "+f"(c00),"+f"(c01),"+f"(c02),"+f"(c03),
            "+f"(c10),"+f"(c11),"+f"(c12),"+f"(c13),
            "+f"(c20),"+f"(c21),"+f"(c22),"+f"(c23),
            "+f"(c30),"+f"(c31),"+f"(c32),"+f"(c33)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
            "r"(a4),"r"(a5),"r"(a6),"r"(a7),
            "r"(b0),"r"(b1),"r"(b2),"r"(b3));
    }

    c00 += c10+c20+c30;
    c01 += c11+c21+c31;
    c02 += c12+c22+c32;
    c03 += c13+c23+c33;

    unsigned int base = blockIdx.x * 16u * 8u + threadIdx.x * 4u;
    out[base + 0] = c00;
    out[base + 1] = c01;
    out[base + 2] = c02;
    out[base + 3] = c03;
}
