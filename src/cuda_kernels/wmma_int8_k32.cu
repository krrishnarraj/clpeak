// INT8 tensor-core throughput at the NVIDIA-native m16n8k32 tile via
// inline mma.sync PTX.  The wmma.h fragment-API path uses m16n16k16
// (K=16), which on NVIDIA effectively halves the natural INT8 tensor
// throughput because the hardware pipe accumulates 4 INT8 lanes per
// 32-bit slot at K=32.  This kernel exposes that K=32 path directly.
//
// Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major):
//   A: m16 x k32 = 512 bytes / 32 threads = 16 bytes/thread = 4 x .b32
//   B: k32 x  n8 = 256 bytes / 32 threads =  8 bytes/thread = 2 x .b32
//   C/D: m16 x n8 = 128 int32 / 32 threads = 4 int32/thread per accumulator
//
// 4 independent accumulator chains, packed into one non-volatile asm
// block (same fix as wmma_fp8_e4m3.cu).
//
// Per warp ops = 256 outer * 4 chains * (16*8*32*2) = 8,388,608;
// per thread = 262,144 (= 4 * COOPMAT_WORK_PER_WI).

extern "C" __global__ void wmma_int8_k32(int *out, int A)
{
    unsigned int packed = (A & 0xff)
                        | (((A + 1) & 0xff) << 8)
                        | (((A + 2) & 0xff) << 16)
                        | (((A + 3) & 0xff) << 24);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed;

    int c00=0,c01=0,c02=0,c03=0;
    int c10=0,c11=0,c12=0,c13=0;
    int c20=0,c21=0,c22=0,c23=0;
    int c30=0,c31=0,c32=0,c33=0;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%0,%1,%2,%3}, {%16,%17,%18,%19}, {%20,%21}, {%0,%1,%2,%3};\n"
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%4,%5,%6,%7}, {%16,%17,%18,%19}, {%20,%21}, {%4,%5,%6,%7};\n"
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%8,%9,%10,%11}, {%16,%17,%18,%19}, {%20,%21}, {%8,%9,%10,%11};\n"
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
              "{%12,%13,%14,%15}, {%16,%17,%18,%19}, {%20,%21}, {%12,%13,%14,%15};\n"
          : "+r"(c00),"+r"(c01),"+r"(c02),"+r"(c03),
            "+r"(c10),"+r"(c11),"+r"(c12),"+r"(c13),
            "+r"(c20),"+r"(c21),"+r"(c22),"+r"(c23),
            "+r"(c30),"+r"(c31),"+r"(c32),"+r"(c33)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
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
