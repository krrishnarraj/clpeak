// MXFP4 (E2M1 + UE8M0 scale) tensor-core throughput via inline mma.sync PTX.
// Tile: m16n8k64 with fp32 accumulator.  This is the Blackwell FP4 path
// that best corresponds to NVIDIA's 5th-generation Tensor Core AI TOPS.
//
// Unlike `.kind::f8f6f4`, `.kind::mxf4` stores E2M1 values packed as true
// 4-bit elements, with separate UE8M0 block-scale metadata.  The scale
// selectors are constant for a throughput probe; the arithmetic value is
// arbitrary because all inputs are synthetic constants.
//
// Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major):
//   A: m16 x k64 packed FP4 = 512 bytes / 32 threads = 16 bytes/thread = 4 x .b32
//   B: k64 x  n8 packed FP4 = 256 bytes / 32 threads =  8 bytes/thread = 2 x .b32
//   C/D: m16 x n8 = 128 fp32 / 32 threads = 4 fp32/thread per accumulator
//
// Per warp ops = 256 outer * 8 chains * (16*8*64*2) = 33,554,432;
// per thread = 1,048,576 (= 16 * COOPMAT_WORK_PER_WI).

extern "C" __global__ void wmma_mxf4_e2m1(float *out, float A)
{
    unsigned int packed = 0x44444444u ^ (__float_as_uint(A) & 0x11111111u);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed;
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
          "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
              "{%0,%1,%2,%3}, {%32,%33,%34,%35}, {%36,%37}, {%0,%1,%2,%3}, %38, {2, 1}, %39, {2, 3};\n"
          "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
              "{%4,%5,%6,%7}, {%32,%33,%34,%35}, {%36,%37}, {%4,%5,%6,%7}, %38, {2, 1}, %39, {2, 3};\n"
          "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
              "{%8,%9,%10,%11}, {%32,%33,%34,%35}, {%36,%37}, {%8,%9,%10,%11}, %38, {2, 1}, %39, {2, 3};\n"
          "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
              "{%12,%13,%14,%15}, {%32,%33,%34,%35}, {%36,%37}, {%12,%13,%14,%15}, %38, {2, 1}, %39, {2, 3};\n"
          "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
              "{%16,%17,%18,%19}, {%32,%33,%34,%35}, {%36,%37}, {%16,%17,%18,%19}, %38, {2, 1}, %39, {2, 3};\n"
          "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
              "{%20,%21,%22,%23}, {%32,%33,%34,%35}, {%36,%37}, {%20,%21,%22,%23}, %38, {2, 1}, %39, {2, 3};\n"
          "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
              "{%24,%25,%26,%27}, {%32,%33,%34,%35}, {%36,%37}, {%24,%25,%26,%27}, %38, {2, 1}, %39, {2, 3};\n"
          "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.f32.e2m1.e2m1.f32.ue8m0 "
              "{%28,%29,%30,%31}, {%32,%33,%34,%35}, {%36,%37}, {%28,%29,%30,%31}, %38, {2, 1}, %39, {2, 3};\n"
          : "+f"(c00),"+f"(c01),"+f"(c02),"+f"(c03),
            "+f"(c10),"+f"(c11),"+f"(c12),"+f"(c13),
            "+f"(c20),"+f"(c21),"+f"(c22),"+f"(c23),
            "+f"(c30),"+f"(c31),"+f"(c32),"+f"(c33),
            "+f"(c40),"+f"(c41),"+f"(c42),"+f"(c43),
            "+f"(c50),"+f"(c51),"+f"(c52),"+f"(c53),
            "+f"(c60),"+f"(c61),"+f"(c62),"+f"(c63),
            "+f"(c70),"+f"(c71),"+f"(c72),"+f"(c73)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),
            "r"(scaleA), "r"(scaleB));
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
