// INT8 tensor-core with 2:4 structured sparsity via inline mma.sp PTX.
// Tile m16n8k32, s8 x s8 + s32.  Requires sm_80+ (Ampere/Ada/Hopper/Blackwell).
//
// Uses the `mma.sp::ordered_metadata` qualifier (PTX ISA 8.5+).  Plain
// `mma.sp` still assembles on sm_90+ but maps to a much slower path on
// Hopper/Blackwell -- measured 35 TOPS on RTX 5060 (sm_120) with plain
// mma.sp vs ~165 TOPS with ordered_metadata.  On sm_80..sm_89 either
// qualifier gives the same throughput; pinning to ordered_metadata keeps
// one kernel for the whole sm_80+ range.
//
// EIGHT independent accumulator chains (vs 4 for wmma_int8_k32): the
// dense K=32 kernel saturates at 4 chains at ~165 TOPS on sm_120, but
// that's a per-shape issue-rate ceiling for m16n8k32 in the dense
// pipeline.  The sparse pipeline may have higher headroom -- bumping
// to 8 chains is the same trick that lifted wmma_fp8_e4m3 past its
// 4-chain ceiling.  All 8 mma.sp.ordered_metadata instructions live in
// a single non-volatile asm block so ptxas is free to interleave them.
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
// Ops accounting (matches the rest of the wmma tests): count nominal
// M*N*K*2 per mma instruction = 16*8*32*2 = 8192.  No sparsity multiplier
// on top -- mma.sp runs at ~2x the dense throughput at the same shape on
// hardware that actually accelerates sparsity, so the reported TOPS
// naturally lands near the vendor "with sparsity" peak while the metric
// remains "instructions x nominal mnk*2 ops / time".

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
    int c40=0,c41=0,c42=0,c43=0;
    int c50=0,c51=0,c52=0,c53=0;
    int c60=0,c61=0,c62=0,c63=0;
    int c70=0,c71=0,c72=0,c73=0;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        // 8 mma.sp in one non-volatile asm block -- ptxas may interleave
        // the per-chain ops to saturate the sparse issue rate.
        asm(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%0,%1,%2,%3}, {%32,%33}, {%34,%35}, {%0,%1,%2,%3}, %36, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%4,%5,%6,%7}, {%32,%33}, {%34,%35}, {%4,%5,%6,%7}, %36, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%8,%9,%10,%11}, {%32,%33}, {%34,%35}, {%8,%9,%10,%11}, %36, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%12,%13,%14,%15}, {%32,%33}, {%34,%35}, {%12,%13,%14,%15}, %36, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%16,%17,%18,%19}, {%32,%33}, {%34,%35}, {%16,%17,%18,%19}, %36, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%20,%21,%22,%23}, {%32,%33}, {%34,%35}, {%20,%21,%22,%23}, %36, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%24,%25,%26,%27}, {%32,%33}, {%34,%35}, {%24,%25,%26,%27}, %36, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%28,%29,%30,%31}, {%32,%33}, {%34,%35}, {%28,%29,%30,%31}, %36, 0x0;\n"
          : "+r"(c00),"+r"(c01),"+r"(c02),"+r"(c03),
            "+r"(c10),"+r"(c11),"+r"(c12),"+r"(c13),
            "+r"(c20),"+r"(c21),"+r"(c22),"+r"(c23),
            "+r"(c30),"+r"(c31),"+r"(c32),"+r"(c33),
            "+r"(c40),"+r"(c41),"+r"(c42),"+r"(c43),
            "+r"(c50),"+r"(c51),"+r"(c52),"+r"(c53),
            "+r"(c60),"+r"(c61),"+r"(c62),"+r"(c63),
            "+r"(c70),"+r"(c71),"+r"(c72),"+r"(c73)
          : "r"(a0),"r"(a1), "r"(b0),"r"(b1), "r"(meta));
    }

    // Fold all 8 accumulators into c0 so every chain is live at the store.
    c00 += c10+c20+c30+c40+c50+c60+c70;
    c01 += c11+c21+c31+c41+c51+c61+c71;
    c02 += c12+c22+c32+c42+c52+c62+c72;
    c03 += c13+c23+c33+c43+c53+c63+c73;

    unsigned int base = blockIdx.x * 16u * 8u + threadIdx.x * 4u;
    out[base + 0] = c00;
    out[base + 1] = c01;
    out[base + 2] = c02;
    out[base + 3] = c03;
}
