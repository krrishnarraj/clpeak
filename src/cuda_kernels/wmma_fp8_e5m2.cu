// FP8 (E5M2) tensor-core throughput via inline mma.sync PTX.  Identical
// shape to wmma_fp8_e4m3.cu; only the dtype mnemonic on the mma.sync
// instruction changes.  Same gating (sm_89+).

extern "C" __global__ void wmma_fp8_e5m2(float *out, float A)
{
    unsigned int packed = 0x3c3c3c3cu ^ (__float_as_uint(A) & 0x0f0f0f0fu);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed;
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
          "{%0, %1, %2, %3}, "
          "{%4, %5, %6, %7}, "
          "{%8, %9}, "
          "{%0, %1, %2, %3};\n"
          : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
            "r"(b0), "r"(b1));
    }

    unsigned int base = blockIdx.x * 16u * 8u + threadIdx.x * 4u;
    out[base + 0] = c0;
    out[base + 1] = c1;
    out[base + 2] = c2;
    out[base + 3] = c3;
}
