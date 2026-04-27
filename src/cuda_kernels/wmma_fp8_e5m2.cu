// FP8 (E5M2) tensor-core throughput via inline mma.sync PTX.  Identical
// shape to wmma_fp8_e4m3.cu; only the dtype mnemonic on the mma.sync
// instruction changes.  Same 8-chain / single-asm-block structure for
// the same reason (consumer-Blackwell FP8 issue pipeline is deeper than
// INT8's; 4 chains in volatile blocks plateaued at ~84 TFLOPS).  See
// wmma_fp8_e4m3.cu for the consumer-Blackwell FP8 ceiling note.

extern "C" __global__ void wmma_fp8_e5m2(float *out, float A)
{
    unsigned int packed = 0x3c3c3c3cu ^ (__float_as_uint(A) & 0x0f0f0f0fu);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed;

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
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
              "{%0,%1,%2,%3}, {%32,%33,%34,%35}, {%36,%37}, {%0,%1,%2,%3};\n"
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
              "{%4,%5,%6,%7}, {%32,%33,%34,%35}, {%36,%37}, {%4,%5,%6,%7};\n"
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
              "{%8,%9,%10,%11}, {%32,%33,%34,%35}, {%36,%37}, {%8,%9,%10,%11};\n"
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
              "{%12,%13,%14,%15}, {%32,%33,%34,%35}, {%36,%37}, {%12,%13,%14,%15};\n"
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
              "{%16,%17,%18,%19}, {%32,%33,%34,%35}, {%36,%37}, {%16,%17,%18,%19};\n"
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
              "{%20,%21,%22,%23}, {%32,%33,%34,%35}, {%36,%37}, {%20,%21,%22,%23};\n"
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
              "{%24,%25,%26,%27}, {%32,%33,%34,%35}, {%36,%37}, {%24,%25,%26,%27};\n"
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
              "{%28,%29,%30,%31}, {%32,%33,%34,%35}, {%36,%37}, {%28,%29,%30,%31};\n"
          : "+f"(c00),"+f"(c01),"+f"(c02),"+f"(c03),
            "+f"(c10),"+f"(c11),"+f"(c12),"+f"(c13),
            "+f"(c20),"+f"(c21),"+f"(c22),"+f"(c23),
            "+f"(c30),"+f"(c31),"+f"(c32),"+f"(c33),
            "+f"(c40),"+f"(c41),"+f"(c42),"+f"(c43),
            "+f"(c50),"+f"(c51),"+f"(c52),"+f"(c53),
            "+f"(c60),"+f"(c61),"+f"(c62),"+f"(c63),
            "+f"(c70),"+f"(c71),"+f"(c72),"+f"(c73)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
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
