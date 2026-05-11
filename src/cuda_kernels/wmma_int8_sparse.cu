// INT8 tensor-core with 2:4 structured sparsity via inline mma.sp PTX.
// Tile m16n8k32, s8 x s8 + s32.  Requires sm_80+ (Ampere/Ada/Hopper/Blackwell).
//
// Per-thread fragment layout (32 threads/warp, A=row-major sparse,
// B=col-major dense):
//   A: m16 x k32 with 2:4 (half non-zero) = 16*16 bytes / 32 threads
//      = 8 bytes/thread = 2 x .b32
//   B: k32 x  n8 = 256 bytes / 32 threads = 8 bytes/thread = 2 x .b32
//   C/D: m16 x  n8 = 128 int32 / 32 threads = 4 int32/thread per accumulator
//   metadata: 1 x .b32 per thread; sparsity_selector immediate picks the
//   contributing lane-group.  We use psel=0x0.
//
// Metadata pattern: 0xeeeeeeee = 0b11_10 repeated, i.e. each pair of 2-bit
// fields selects element indices 2 and 3 within each 4-element K group.
// Any valid pattern works for a throughput probe; the math result is
// arbitrary because the inputs are constants.
//
// 4 independent accumulator chains packed into one non-volatile asm block
// (same pattern as wmma_int8_k32.cu / wmma_fp8_e4m3.cu).  If issue-rate
// limited, bump to 8 chains as in wmma_fp8_e4m3.cu.
//
// Ops accounting (matches the rest of the wmma tests): count nominal
// M*N*K*2 per mma instruction = 16*8*32*2 = 8192.  No sparsity multiplier
// on top -- mma.sp runs at ~2x the dense throughput at the same shape, so
// the reported TOPS naturally lands near the vendor "with sparsity" peak
// while the metric remains "instructions x nominal mnk*2 ops / time".

extern "C" __global__ void wmma_int8_sparse(int *out, int A)
{
    unsigned int packed = (A & 0xff)
                        | (((A + 1) & 0xff) << 8)
                        | (((A + 2) & 0xff) << 16)
                        | (((A + 3) & 0xff) << 24);
    unsigned int a0 = packed, a1 = packed;
    unsigned int b0 = packed, b1 = packed;
    unsigned int meta = 0xeeeeeeeeu;

    int c00=0,c01=0,c02=0,c03=0;
    int c10=0,c11=0,c12=0,c13=0;
    int c20=0,c21=0,c22=0,c23=0;
    int c30=0,c31=0,c32=0,c33=0;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm(
          "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%0,%1,%2,%3}, {%16,%17}, {%18,%19}, {%0,%1,%2,%3}, %20, 0x0;\n"
          "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%4,%5,%6,%7}, {%16,%17}, {%18,%19}, {%4,%5,%6,%7}, %20, 0x0;\n"
          "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%8,%9,%10,%11}, {%16,%17}, {%18,%19}, {%8,%9,%10,%11}, %20, 0x0;\n"
          "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%12,%13,%14,%15}, {%16,%17}, {%18,%19}, {%12,%13,%14,%15}, %20, 0x0;\n"
          : "+r"(c00),"+r"(c01),"+r"(c02),"+r"(c03),
            "+r"(c10),"+r"(c11),"+r"(c12),"+r"(c13),
            "+r"(c20),"+r"(c21),"+r"(c22),"+r"(c23),
            "+r"(c30),"+r"(c31),"+r"(c32),"+r"(c33)
          : "r"(a0),"r"(a1), "r"(b0),"r"(b1), "r"(meta));
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
