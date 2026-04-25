// FP8 (E5M2) tensor-core throughput via inline mma.sync PTX.  Identical
// shape to wmma_fp8_e4m3.cu; only the dtype mnemonic on the mma.sync
// instruction changes.  Same 4-chain ILP structure for the same reason.

extern "C" __global__ void wmma_fp8_e5m2(float *out, float A)
{
    unsigned int packed = 0x3c3c3c3cu ^ (__float_as_uint(A) & 0x0f0f0f0fu);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed;

    float c0_0 = 0.f, c0_1 = 0.f, c0_2 = 0.f, c0_3 = 0.f;
    float c1_0 = 0.f, c1_1 = 0.f, c1_2 = 0.f, c1_3 = 0.f;
    float c2_0 = 0.f, c2_1 = 0.f, c2_2 = 0.f, c2_3 = 0.f;
    float c3_0 = 0.f, c3_1 = 0.f, c3_2 = 0.f, c3_3 = 0.f;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c0_0), "+f"(c0_1), "+f"(c0_2), "+f"(c0_3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c1_0), "+f"(c1_1), "+f"(c1_2), "+f"(c1_3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c2_0), "+f"(c2_1), "+f"(c2_2), "+f"(c2_3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(c3_0), "+f"(c3_1), "+f"(c3_2), "+f"(c3_3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }

    c0_0 += c1_0 + c2_0 + c3_0;
    c0_1 += c1_1 + c2_1 + c3_1;
    c0_2 += c1_2 + c2_2 + c3_2;
    c0_3 += c1_3 + c2_3 + c3_3;

    unsigned int base = blockIdx.x * 16u * 8u + threadIdx.x * 4u;
    out[base + 0] = c0_0;
    out[base + 1] = c0_1;
    out[base + 2] = c0_2;
    out[base + 3] = c0_3;
}
