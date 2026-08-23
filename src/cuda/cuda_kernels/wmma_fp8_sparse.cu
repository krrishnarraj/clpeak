// FP8 (E4M3) tensor-core throughput WITH 2:4 structured sparsity, via inline
// `mma.sp::ordered_metadata` PTX.  Sparse counterpart of wmma_fp8_e4m3.cu.
// Requires sm_89+ (Ada/Hopper/Blackwell) and CUDA 12.5+ to assemble.
//
// Shape: the sparse FP8 path is m16n8k64 -- K is DOUBLED relative to the dense
// m16n8k32 FP8 instruction (PTX ISA: "sparse mma.m16n8k64 with .u8/.s8/.e4m3/
// .e5m2 type").  Matrix A is stored 2:4-compressed (half the elements) plus a
// metadata operand that selects which 2 of every 4 K-elements are non-zero; B
// is dense over the full k64.
//
// Uses the `mma.sp::ordered_metadata` qualifier (PTX ISA 8.5+) rather than
// plain `mma.sp`, for the same reason wmma_int8_sparse.cu does: plain mma.sp
// still assembles on sm_90+ but maps to a much slower path there, while on
// sm_89 the two qualifiers are equivalent.  One kernel covers sm_89+.
//
// === Ops accounting ===
// Count the nominal instruction shape M*N*K*2 = 16*8*64*2 = 16384 ops per
// mma.sp.  No extra sparsity multiplier on top -- the doubled K (k64 vs the
// dense k32) already encodes the 2x.  One m16n8k64 sparse mma issues at the
// same rate as a dense m16n8k32 mma on hardware that accelerates 2:4, so this
// reports ~2x the dense FP8 TFLOPS, matching NVIDIA's "with sparsity" figures.
//
// MEASURED, RTX 5060 (sm_120, GeForce/consumer Blackwell): 169.74 TFLOPS,
// i.e. 1.99x the 85.13 of the dense fp32-accumulate FP8 row.  Consumer
// Blackwell accelerates FP8 2:4 at full rate, as it does INT8
// (wmma_int8_sparse.cu) and FP4 (wmma_nvf4_sparse.cu).
//
// This row is still accumulator-capped, though.  On this part the uncapped
// 8-bit tensor rate is ~165-170 (int8+int32 dense k32 = 164.81;
// fp8+fp16 dense k32 = 166.77, wmma_fp8_f16.cu), while fp8+fp32 dense is
// halved to 85.11.  Sparsity doubles whatever rate the accumulator allows:
// int8+int32 sparse k64 reaches 327.30, i.e. 2x the uncapped rate, whereas
// this kernel's 169.74 is 2x the CAPPED rate -- the 2x is spent climbing back
// to the uncapped dense rate rather than doubling past it.  Lifting the
// accumulator on top of the sparsity does compose: the same shape with an fp16
// accumulator measures 326.15 (wmma_fp8_sparse_f16.cu).  That form needs a
// newer target than this one -- ptxas refuses it for sm_89 -- which is why the
// two are separate kernels in separate arch groups.
//
// === Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major) ===
//   A: m16 x k64 @ 2:4 (half non-zero) = 1024 bytes/2 / 32 = 16 B/thread = 4 x .b32
//   B: k64 x  n8                       =  512 bytes    / 32 = 16 B/thread = 4 x .b32
//   metadata: m16 x k64 2:4 selectors  =  128 bytes    / 32 =  4 B/thread = 1 x .b32
//             -- the whole .b32 is consumed at this shape, so the
//             sparsity_selector immediate must be 0x0.
//   C/D: m16 x n8 = 128 fp32 / 32 = 4 fp32/thread per accumulator
//
// Metadata pattern: 0xeeeeeeee = 0b11_10 repeated, i.e. each pair of 2-bit
// fields selects element indices 2 and 3 within each 4-element K group.  Any
// valid pattern works for a throughput probe; the math result is arbitrary
// because the inputs are constants.
//
// EIGHT independent accumulator chains in one non-volatile asm block, matching
// the dense FP8 kernel -- 4 chains left that one latency-bound.
//
// Per warp ops = 256 outer * 8 chains * (16*8*64*2) = 33,554,432;
// per thread = 1,048,576 (= 16 * COOPMAT_WORK_PER_WI).
//
// Accumulates in fp32, like every other sparse row here; the fp16-accumulate
// question is asked separately by wmma_fp8_f16.cu on the dense shape.
//
// Datacenter parts (Ampere/Hopper/B200) are expected to show the full 2x too.

extern "C" __global__ void wmma_fp8_sparse(float *out, float A)
{
    unsigned int packed = 0x3c3c3c3cu ^ (__float_as_uint(A) & 0x0f0f0f0fu);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed, b2 = packed, b3 = packed;
    unsigned int meta = 0xeeeeeeeeu;

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
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%0,%1,%2,%3}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%0,%1,%2,%3}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%4,%5,%6,%7}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%4,%5,%6,%7}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%8,%9,%10,%11}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%8,%9,%10,%11}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%12,%13,%14,%15}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%12,%13,%14,%15}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%16,%17,%18,%19}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%16,%17,%18,%19}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%20,%21,%22,%23}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%20,%21,%22,%23}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%24,%25,%26,%27}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%24,%25,%26,%27}, %40, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
              "{%28,%29,%30,%31}, {%32,%33,%34,%35}, {%36,%37,%38,%39}, {%28,%29,%30,%31}, %40, 0x0;\n"
          : "+f"(c00),"+f"(c01),"+f"(c02),"+f"(c03),
            "+f"(c10),"+f"(c11),"+f"(c12),"+f"(c13),
            "+f"(c20),"+f"(c21),"+f"(c22),"+f"(c23),
            "+f"(c30),"+f"(c31),"+f"(c32),"+f"(c33),
            "+f"(c40),"+f"(c41),"+f"(c42),"+f"(c43),
            "+f"(c50),"+f"(c51),"+f"(c52),"+f"(c53),
            "+f"(c60),"+f"(c61),"+f"(c62),"+f"(c63),
            "+f"(c70),"+f"(c71),"+f"(c72),"+f"(c73)
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
