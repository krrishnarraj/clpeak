// INT8 tensor-core with 2:4 structured sparsity via inline mma.sp PTX.
// Tile m16n8k64, s8 x s8 + s32.  Requires sm_80+ (Ampere/Ada/Hopper/Blackwell).
//
// Shape: K is DOUBLED relative to the dense m16n8k32 INT8 instruction
// (wmma_int8_k32.cu).  Matrix A is stored 2:4-compressed -- so a k64 sparse A
// carries the same bytes as a k32 dense A -- plus a metadata operand selecting
// which 2 of every 4 K-elements are non-zero; B is dense over the full k64.
// This is the shape that converts 2:4 into throughput, and it mirrors what the
// FP8 (m16n8k64 vs dense k32) and FP4 (m16n8k128 vs dense k64) sparse kernels
// do.
//
// Why NOT the m16n8k32 sparse shape, which PTX also offers: its dense
// counterpart is m16n8k16, not the m16n8k32 row we report next to it.  Pinned
// at k32 this kernel measured 164.85 TOPS on RTX 5060 against 164.84 for dense
// wmma_int8_k32 -- a flat "no sparsity gain" that was an artefact of comparing
// a sparse k32 against a dense k32 rather than any hardware gating.  (Against
// its true counterpart, the k16-shaped wmma_int8 at 84.39, that same 164.85
// was already a ~1.95x.)
//
// Uses the `mma.sp::ordered_metadata` qualifier (PTX ISA 8.5+).  Plain
// `mma.sp` still assembles on sm_90+ but maps to a much slower path on
// Hopper/Blackwell -- measured 35 TOPS on RTX 5060 (sm_120) with plain
// mma.sp vs ~165 TOPS with ordered_metadata at the old k32 shape.  On
// sm_80..sm_89 either qualifier gives the same throughput; pinning to
// ordered_metadata keeps one kernel for the whole sm_80+ range.
//
// EIGHT independent accumulator chains in a single non-volatile asm block,
// matching wmma_fp8_sparse.cu (identical operand shape).  A volatile block
// would force in-order emission and defeat the ILP.
//
// Per-thread fragment layout (32 threads/warp, A=row-major sparse,
// B=col-major dense):
//   A: m16 x k64 @ 2:4 (half non-zero) = 1024 bytes/2 / 32 = 16 B/thread = 4 x .b32
//   B: k64 x  n8                       =  512 bytes    / 32 = 16 B/thread = 4 x .b32
//   metadata: m16 x k64 2:4 selectors  =  128 bytes    / 32 =  4 B/thread = 1 x .b32
//             -- the whole .b32 is consumed at this shape, so the
//             sparsity_selector immediate must be 0x0.
//   C/D: m16 x n8 = 128 int32 / 32 threads = 4 int32/thread per accumulator
//
// Metadata pattern: 0xeeeeeeee = 0b11_10 repeated, i.e. each pair of 2-bit
// fields selects element indices 2 and 3 within each 4-element K group.
// Any valid pattern works for a throughput probe; the math result is
// arbitrary because the inputs are constants.
//
// === Ops accounting ===
// Count the nominal instruction shape M*N*K*2 = 16*8*64*2 = 16384 ops per
// mma.sp.  No sparsity multiplier on top -- the doubled K already encodes the
// 2x.  One m16n8k64 sparse mma issues at the same rate as a dense m16n8k32 mma
// on hardware that accelerates 2:4, so this reports ~2x the dense INT8 TOPS,
// landing near the vendor "with sparsity" peak.
//
// Per warp ops = 256 outer * 8 chains * (16*8*64*2) = 33,554,432;
// per thread = 1,048,576 (= 16 * COOPMAT_WORK_PER_WI).

extern "C" __global__ void wmma_int8_sparse(int *out, int A)
{
    unsigned int packed = (A & 0xff)
                        | (((A + 1) & 0xff) << 8)
                        | (((A + 2) & 0xff) << 16)
                        | (((A + 3) & 0xff) << 24);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed, b2 = packed, b3 = packed;
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
        asm(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
              "{%0,%1,%2,%3}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%0,%1,%2,%3}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
              "{%4,%5,%6,%7}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%4,%5,%6,%7}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
              "{%8,%9,%10,%11}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%8,%9,%10,%11}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
              "{%12,%13,%14,%15}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%12,%13,%14,%15}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
              "{%16,%17,%18,%19}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%16,%17,%18,%19}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
              "{%20,%21,%22,%23}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%20,%21,%22,%23}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
              "{%24,%25,%26,%27}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%24,%25,%26,%27}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
              "{%28,%29,%30,%31}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%28,%29,%30,%31}, %40, 0x0;\n"
          : "+r"(c00),"+r"(c01),"+r"(c02),"+r"(c03),
            "+r"(c10),"+r"(c11),"+r"(c12),"+r"(c13),
            "+r"(c20),"+r"(c21),"+r"(c22),"+r"(c23),
            "+r"(c30),"+r"(c31),"+r"(c32),"+r"(c33),
            "+r"(c40),"+r"(c41),"+r"(c42),"+r"(c43),
            "+r"(c50),"+r"(c51),"+r"(c52),"+r"(c53),
            "+r"(c60),"+r"(c61),"+r"(c62),"+r"(c63),
            "+r"(c70),"+r"(c71),"+r"(c72),"+r"(c73)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),"r"(b2),"r"(b3),
            "r"(meta));
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
