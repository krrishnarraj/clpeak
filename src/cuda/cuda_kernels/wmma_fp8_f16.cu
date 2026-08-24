// FP8 (E4M3) tensor-core throughput with an fp16 (NOT fp32) accumulator, via
// inline mma.sync PTX at m16n8k32.  Available on sm_89+ (Ada) and sm_90+
// (Hopper/Blackwell); PTX form `.f16.e4m3.e4m3.f16` -- D and C are fp16 while
// A and B stay e4m3.
//
// Why a separate kernel: the same reason wmma_fp16_f16.cu exists.  GeForce
// parts hold the fp32-accumulate tensor-core paths at half rate -- fp16+fp32
// measures 42.55 TFLOPS on RTX 5060 against 83.39 for fp16+fp16.  The dense
// FP8 test (wmma_fp8_e4m3.cu) accumulates in fp32 and lands at 85.13 there,
// while cuBLASLt's fp8 GEMM on the same part reports 153.33.
//
// The shape is unchanged from the fp32-accumulate kernel (m16n8k32), so the
// two rows are directly comparable: same instruction count, same nominal ops.
//
// MEASURED, RTX 5060 (sm_120, GeForce/consumer Blackwell): 166.81 TFLOPS,
// i.e. 1.96x the 85.13 of the fp32-accumulate form and ABOVE cuBLASLt's
// 153.33.  So the FP8 gap was the accumulator width, not the FP8 issue rate:
// the same half-rate-on-fp32-accumulate rule that governs fp16 governs fp8.
// Note this does NOT extend to the block-scaled FP4 path, which reaches its
// full ~328 with an fp32 accumulator (wmma_nvf4_e2m1.cu) -- the cap bites on
// fp16 and fp8, not on fp4.
//
// Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major):
//   A: m16 x k32 = 512 bytes / 32 threads = 16 bytes/thread = 4 x .b32
//   B: k32 x  n8 = 256 bytes / 32 threads =  8 bytes/thread = 2 x .b32
//   C/D: m16 x n8 = 128 halfs / 32 threads = 4 halfs = 2 x .b32 (f16x2 packed)
//        per accumulator -- half the accumulator registers of the fp32 form.
//
// EIGHT independent chains in a single non-volatile asm block, matching
// wmma_fp8_e4m3.cu: 4 chains left the FP8 pipeline latency-bound there, and
// a volatile block would force in-order emission and defeat the ILP.
//
// Per warp ops = 256 outer * 8 chains * (16*8*32*2) = 16,777,216;
// per thread = 524,288 (= 8 * COOPMAT_WORK_PER_WI) -- identical accounting to
// the fp32-accumulate kernel, so the two numbers can be compared directly.
//
// Only E4M3 is measured: E5M2 shares the data path (the dense pair reads 85.11
// vs 85.10 on RTX 5060), so a second row would add no information.

#include <cuda_fp16.h>

extern "C" __global__ void wmma_fp8_f16(float *out, float A)
{
    unsigned int packed = 0x3c3c3c3cu ^ (__float_as_uint(A) & 0x0f0f0f0fu);
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
          "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
              "{%0,%1}, {%16,%17,%18,%19}, {%20,%21}, {%0,%1};\n"
          "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
              "{%2,%3}, {%16,%17,%18,%19}, {%20,%21}, {%2,%3};\n"
          "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
              "{%4,%5}, {%16,%17,%18,%19}, {%20,%21}, {%4,%5};\n"
          "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
              "{%6,%7}, {%16,%17,%18,%19}, {%20,%21}, {%6,%7};\n"
          "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
              "{%8,%9}, {%16,%17,%18,%19}, {%20,%21}, {%8,%9};\n"
          "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
              "{%10,%11}, {%16,%17,%18,%19}, {%20,%21}, {%10,%11};\n"
          "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
              "{%12,%13}, {%16,%17,%18,%19}, {%20,%21}, {%12,%13};\n"
          "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
              "{%14,%15}, {%16,%17,%18,%19}, {%20,%21}, {%14,%15};\n"
          : "+r"(c00),"+r"(c01), "+r"(c10),"+r"(c11),
            "+r"(c20),"+r"(c21), "+r"(c30),"+r"(c31),
            "+r"(c40),"+r"(c41), "+r"(c50),"+r"(c51),
            "+r"(c60),"+r"(c61), "+r"(c70),"+r"(c71)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1));
    }

    // Reduce every chain live: unpack each f16x2 accumulator and sum to fp32,
    // so ptxas cannot dead-code any of the 8 chains.
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
