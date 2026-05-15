// Binary (1-bit) BMMA tensor-core throughput at the m8n8k128 tile via
// inline mma.sync PTX with XOR-popcount semantics.  Available sm_75+
// (Turing).  Reported in TOPS following the conventional 2*m*n*k op
// count for binary tensor cores.
//
// Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major):
//   A: m8  x k128 = 1024 bits = 128 bytes / 32 threads = 4 bytes = 1 x .b32
//   B: k128 x n8  = same                                         = 1 x .b32
//   C/D: m8 x n8  = 64 int32 / 32 threads = 2 int32/thread per accumulator
//
// 4 independent accumulator chains, single non-volatile asm block.
//
// Per warp ops = 256 outer * 4 chains * (8*8*128*2) = 16,777,216;
// per thread = 524,288 (= 8 * COOPMAT_WORK_PER_WI).

extern "C" __global__ void wmma_bmma_b1(int *out, int A)
{
    unsigned int a0 = (unsigned int)A;
    unsigned int b0 = (unsigned int)A;

    int c00=0,c01=0;
    int c10=0,c11=0;
    int c20=0,c21=0;
    int c30=0,c31=0;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm(
          "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
              "{%0,%1}, {%8}, {%9}, {%0,%1};\n"
          "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
              "{%2,%3}, {%8}, {%9}, {%2,%3};\n"
          "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
              "{%4,%5}, {%8}, {%9}, {%4,%5};\n"
          "mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
              "{%6,%7}, {%8}, {%9}, {%6,%7};\n"
          : "+r"(c00),"+r"(c01),
            "+r"(c10),"+r"(c11),
            "+r"(c20),"+r"(c21),
            "+r"(c30),"+r"(c31)
          : "r"(a0), "r"(b0));
    }

    c00 += c10 + c20 + c30;
    c01 += c11 + c21 + c31;

    unsigned int base = blockIdx.x * 8u * 8u + threadIdx.x * 2u;
    out[base + 0] = c00;
    out[base + 1] = c01;
}
