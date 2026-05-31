// NVFP4 (E2M1 + UE4M3 scale) tensor-core throughput WITH 2:4 structured
// sparsity, via inline `mma.sp::ordered_metadata` PTX.  Sparse counterpart of
// `wmma_nvf4_e2m1.cu`.  Requires Blackwell sm_120a+.
//
// Shape: the sparse FP4 path is m16n8k128 -- K is DOUBLED relative to the dense
// m16n8k64 NVFP4 instruction (PTX ISA: "sparse mma.m16n8k128 with .e2m1 type").
// Matrix A is stored 2:4-compressed (half the elements) plus a metadata operand
// that selects which 2 of every 4 K-elements are non-zero; B is dense over the
// full k128.  `.kind::mxf4nvf4` keeps the NVFP4 block-scale encoding (UE4M3
// scale, 16-element blocks).
//
// === Ops accounting ===
// Count the nominal instruction shape M*N*K*2 = 16*8*128*2 = 32768 ops per
// mma.sp.  No sparsity multiplier on top.  On hardware that actually accelerates
// 2:4 (datacenter Blackwell), one m16n8k128 sparse mma issues at the same rate
// as a dense m16n8k64 mma, so this reports ~2x the dense NVFP4 TOPS and lands
// near the vendor "with sparsity" AI-TOPS peak.  On hardware that does NOT
// accelerate sparsity (GeForce sm_120 / RTX 50-series), the sparse instruction
// processes the full k128 at half the issue rate, so it reports the SAME ~327
// TOPS as dense -- see `wmma_int8_sparse.cu` for the same null result measured
// on RTX 5060 (int8_sparse == int8_k32, 0% gain).  That GeForce gating is the
// documented reason the advertised 615 "AI TOPS" is unreachable on consumer
// parts; this kernel makes the gating directly measurable.
//
// === Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major) ===
//   A: m16 x k128 @ 2:4 (half non-zero) = 1024 bytes/2 / 32 = 16 B/thread = 4 x .b32
//   B: k128 x  n8 packed FP4            = 1024 bytes   / 32 = 16 B/thread = 4 x .b32
//   metadata: m16 x k128 2:4 selectors  =  128 bytes   / 32 =  4 B/thread = 1 x .b32
//   C/D: m16 x n8 = 128 fp32 / 32 = 4 fp32/thread per accumulator
//
// 8 independent accumulator chains in one asm block (same ILP pattern as the
// dense NVFP4 kernel and wmma_int8_sparse.cu).
//
// NOTE (to confirm on first NVRTC build): the exact operand order for
// sparse + block_scale combined, the scale_vec size for the doubled K, and the
// metadata/sparsity-selector placement are best-effort here.  ptxas on the
// sm_120a box is the source of truth -- adjust the asm template against its
// diagnostics if it rejects this form.

extern "C" __global__ void wmma_nvf4_sparse(float *out, float A)
{
    unsigned int packed = 0x44444444u ^ (__float_as_uint(A) & 0x11111111u);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed, b2 = packed, b3 = packed;
    // 2:4 metadata: 0xeeeeeeee = 0b11_10 repeated -> selects elements 2,3 of
    // each 4-element K group.  Any valid pattern works for a throughput probe.
    unsigned int meta = 0xeeeeeeeeu;
    // UE4M3 block scale ~ 1.0 (0x3c packed four-per-word); value is irrelevant
    // for a throughput probe.
    unsigned int scaleA = 0x3c3c3c3cu;
    unsigned int scaleB = 0x3c3c3c3cu;

    float c00=0,c01=0,c02=0,c03=0;
    float c10=0,c11=0,c12=0,c13=0;
    float c20=0,c21=0,c22=0,c23=0;
    float c30=0,c31=0,c32=0,c33=0;
    float c40=0,c41=0,c42=0,c43=0;
    float c50=0,c51=0,c52=0,c53=0;
    float c60=0,c61=0,c62=0,c63=0;
    float c70=0,c71=0,c72=0,c73=0;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm(
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
              "{%0,%1,%2,%3}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%0,%1,%2,%3}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
              "{%4,%5,%6,%7}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%4,%5,%6,%7}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
              "{%8,%9,%10,%11}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%8,%9,%10,%11}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
              "{%12,%13,%14,%15}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%12,%13,%14,%15}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
              "{%16,%17,%18,%19}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%16,%17,%18,%19}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
              "{%20,%21,%22,%23}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%20,%21,%22,%23}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
              "{%24,%25,%26,%27}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%24,%25,%26,%27}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
              "{%28,%29,%30,%31}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%28,%29,%30,%31}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          : "+f"(c00),"+f"(c01),"+f"(c02),"+f"(c03),
            "+f"(c10),"+f"(c11),"+f"(c12),"+f"(c13),
            "+f"(c20),"+f"(c21),"+f"(c22),"+f"(c23),
            "+f"(c30),"+f"(c31),"+f"(c32),"+f"(c33),
            "+f"(c40),"+f"(c41),"+f"(c42),"+f"(c43),
            "+f"(c50),"+f"(c51),"+f"(c52),"+f"(c53),
            "+f"(c60),"+f"(c61),"+f"(c62),"+f"(c63),
            "+f"(c70),"+f"(c71),"+f"(c72),"+f"(c73)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),"r"(b2),"r"(b3),
            "r"(meta), "r"(scaleA), "r"(scaleB));
    }

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
