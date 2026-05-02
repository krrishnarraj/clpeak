// INT4 (s4) tensor-core throughput at the m8n8k32 tile via inline mma.sync
// PTX.  s4 IMMA was added on Turing (sm_75) and kept through Ada (sm_89);
// removed on Hopper+ (sm_90+).  The runtime gate is set in CudaDevice::init.
//
// Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major):
//   A: m8  x k32 = 256 nibbles = 128 bytes / 32 threads = 4 bytes  = 1 x .b32
//   B: k32 x n8  = same                                            = 1 x .b32
//   C/D: m8 x n8 = 64 int32 / 32 threads = 2 int32/thread per accumulator
//
// 4 independent accumulator chains, single non-volatile asm block (same
// pattern as wmma_int8_k32.cu).
//
// Per warp ops = 256 outer * 4 chains * (8*8*32*2) = 4,194,304;
// per thread = 131,072 (= 2 * COOPMAT_WORK_PER_WI).  Reported in TOPS.

extern "C" __global__ void wmma_int4(int *out, int A)
{
    // Pack 8 nibbles into one int32 register (s4).
    unsigned int packed = ((A & 0xf) <<  0) | ((A & 0xf) <<  4)
                        | ((A & 0xf) <<  8) | ((A & 0xf) << 12)
                        | ((A & 0xf) << 16) | ((A & 0xf) << 20)
                        | ((A & 0xf) << 24) | ((A & 0xf) << 28);
    unsigned int a0 = packed;
    unsigned int b0 = packed;

    int c00=0,c01=0;
    int c10=0,c11=0;
    int c20=0,c21=0;
    int c30=0,c31=0;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm(
          "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
              "{%0,%1}, {%8}, {%9}, {%0,%1};\n"
          "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
              "{%2,%3}, {%8}, {%9}, {%2,%3};\n"
          "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
              "{%4,%5}, {%8}, {%9}, {%4,%5};\n"
          "mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 "
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
