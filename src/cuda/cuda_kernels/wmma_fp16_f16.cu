// FP16 tensor-core throughput with fp16 (NOT fp32) accumulator, via inline
// mma.sync PTX at m16n8k16.  Ampere+.
//
// Why a separate kernel: on GeForce (consumer Ada/Blackwell) fp16xfp16 with
// an fp32 accumulator runs at HALF rate -- that is what the "+fp32" wmma_fp16
// test measures (~42 TFLOPS on RTX 5060).  The full-rate path uses an fp16
// accumulator (f16.f16.f16.f16), which is what cuBLASLt's ~78 fp16 number
// uses.  This kernel measures that full-rate fp16-accumulate peak (~84).
//
// Accumulator differs from the +fp32 kernels: C/D is m16n8 in fp16 = 128
// halfs / 32 threads = 4 halfs = 2 x .b32 (f16x2 packed) per chain, so each
// chain has 2 accumulator registers, not 4.
//
// Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major):
//   A: m16 x k16 = 512 bytes / 32 threads = 4 x .b32 (2 halfs each)
//   B: k16 x  n8 = 256 bytes / 32 threads = 2 x .b32
//   C/D: m16 x n8 = 128 halfs / 32 threads = 2 x .b32 (f16x2) per accumulator
//
// EIGHT independent chains in a single non-volatile asm block.
//
// Per warp ops = 256 outer * 8 chains * (16*8*16*2) = 8,388,608;
// per thread = 262,144 (= 4 * COOPMAT_WORK_PER_WI).

#include <cuda_fp16.h>

extern "C" __global__ void wmma_fp16_f16(float *out, float A)
{
    unsigned short h = __half_as_ushort(__float2half(A));
    unsigned int packed = (unsigned int)h | ((unsigned int)h << 16);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed;

    // 8 chains, each C/D = 2 x .b32 holding an f16x2 pair.
    unsigned int c00=0,c01=0;
    unsigned int c10=0,c11=0;
    unsigned int c20=0,c21=0;
    unsigned int c30=0,c31=0;
    unsigned int c40=0,c41=0;
    unsigned int c50=0,c51=0;
    unsigned int c60=0,c61=0;
    unsigned int c70=0,c71=0;

    #pragma unroll 1
    for (int i = 0; i < 256; i++)
    {
        asm(
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
              "{%0,%1}, {%16,%17,%18,%19}, {%20,%21}, {%0,%1};\n"
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
              "{%2,%3}, {%16,%17,%18,%19}, {%20,%21}, {%2,%3};\n"
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
              "{%4,%5}, {%16,%17,%18,%19}, {%20,%21}, {%4,%5};\n"
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
              "{%6,%7}, {%16,%17,%18,%19}, {%20,%21}, {%6,%7};\n"
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
              "{%8,%9}, {%16,%17,%18,%19}, {%20,%21}, {%8,%9};\n"
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
              "{%10,%11}, {%16,%17,%18,%19}, {%20,%21}, {%10,%11};\n"
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
              "{%12,%13}, {%16,%17,%18,%19}, {%20,%21}, {%12,%13};\n"
          "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
              "{%14,%15}, {%16,%17,%18,%19}, {%20,%21}, {%14,%15};\n"
          : "+r"(c00),"+r"(c01), "+r"(c10),"+r"(c11),
            "+r"(c20),"+r"(c21), "+r"(c30),"+r"(c31),
            "+r"(c40),"+r"(c41), "+r"(c50),"+r"(c51),
            "+r"(c60),"+r"(c61), "+r"(c70),"+r"(c71)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
    }

    // Reduce every chain live: unpack each f16x2 accumulator and sum to fp32.
    unsigned int regs[16] = { c00,c01, c10,c11, c20,c21, c30,c31,
                              c40,c41, c50,c51, c60,c61, c70,c71 };
    float s = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; j++)
    {
        __half lo = __ushort_as_half((unsigned short)(regs[j] & 0xffffu));
        __half hi = __ushort_as_half((unsigned short)(regs[j] >> 16));
        s += __half2float(lo) + __half2float(hi);
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
