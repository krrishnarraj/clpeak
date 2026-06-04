// MXFP4 (E2M1 + UE8M0 scale) tensor-core throughput WITH 2:4 structured
// sparsity, via inline `mma.sp::ordered_metadata` PTX.  Sparse counterpart of
// `wmma_mxf4_e2m1.cu`.  Requires Blackwell sm_120a+.
//
// Shape: the sparse FP4 path is m16n8k128 -- K is DOUBLED relative to the dense
// m16n8k64 MXFP4 instruction (PTX ISA: "sparse mma.m16n8k128 with .e2m1 type").
// Matrix A is stored 2:4-compressed (half the elements) plus a metadata operand
// selecting which 2 of every 4 K-elements are non-zero; B is dense over the full
// k128.  `.kind::mxf4` keeps the MXFP4 block-scale encoding (UE8M0 scale,
// 32-element blocks).
//
// === Ops accounting ===
// Count the nominal instruction shape M*N*K*2 = 16*8*128*2 = 32768 ops per
// mma.sp.  The doubled K (k128 vs the dense k64) already encodes the 2x; one
// m16n8k128 sparse mma issues at the dense m16n8k64 rate on hardware that
// accelerates 2:4 -> reports ~2x the dense MXFP4 TOPS, matching NVIDIA's "with
// sparsity" AI-TOPS definition.
//
// Consumer Blackwell (RTX 5060, sm_120) DOES accelerate FP4 2:4 sparsity -- the
// sibling NVFP4 sparse kernel measures ~632 TFLOPS (~1.93x dense, above the
// advertised 615).  This is unlike INT8, where `wmma_int8_sparse.cu` measures
// int8_sparse == int8_k32 (0% gain) on the same GPU: NVIDIA gates the INT sparse
// path on GeForce but not the FP4 one.
//
// === Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major) ===
//   A: m16 x k128 @ 2:4 (half non-zero) = 1024 bytes/2 / 32 = 16 B/thread = 4 x .b32
//   B: k128 x  n8 packed FP4            = 1024 bytes   / 32 = 16 B/thread = 4 x .b32
//   metadata: m16 x k128 2:4 selectors  =  128 bytes   / 32 =  4 B/thread = 1 x .b32
//   C/D: m16 x n8 = 128 fp32 / 32 = 4 fp32/thread per accumulator
//
// 8 independent accumulator chains in one asm block (same ILP pattern as the
// dense MXFP4 kernel and wmma_int8_sparse.cu).
//
// scale_vec::2X tracks the COMPRESSED K=64 (mxf4 block32 -> 64/32 = 2), not the
// logical k128.  ptxas rejects `.scale_vec::4X` with `.kind::mxf4` (4X is the
// NVFP4/block16 size); 2X is the mxf4/block32 size.  Operand order matches the
// NVFP4 sparse kernel: D, A{4}, B{4}, C, meta, 0x0, scaleA, {0,0}, scaleB, {0,0}.

extern "C" __global__ void wmma_mxf4_sparse(float *out, float A)
{
    unsigned int packed = 0x44444444u ^ (__float_as_uint(A) & 0x11111111u);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed, b2 = packed, b3 = packed;
    // 2:4 metadata: 0xeeeeeeee selects elements 2,3 of each 4-element K group.
    unsigned int meta = 0xeeeeeeeeu;
    // UE8M0 block scale; value is irrelevant for a throughput probe.
    unsigned int scaleA = 0x3f3f3f3fu;
    unsigned int scaleB = 0x3f3f3f3fu;

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
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
              "{%0,%1,%2,%3}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%0,%1,%2,%3}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
              "{%4,%5,%6,%7}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%4,%5,%6,%7}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
              "{%8,%9,%10,%11}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%8,%9,%10,%11}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
              "{%12,%13,%14,%15}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%12,%13,%14,%15}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
              "{%16,%17,%18,%19}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%16,%17,%18,%19}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
              "{%20,%21,%22,%23}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%20,%21,%22,%23}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
              "{%24,%25,%26,%27}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%24,%25,%26,%27}, %40, 0x0, %41, {0, 0}, %42, {0, 0};\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k128.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
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
