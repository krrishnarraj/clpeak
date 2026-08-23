// FP8 (E4M3) tensor-core throughput with 2:4 structured sparsity AND an fp16
// (NOT fp32) accumulator, via inline `mma.sp::ordered_metadata` PTX at
// m16n8k64.  The two-effects-at-once corner of the FP8 grid:
//
// MEASURED, RTX 5060 (sm_120, GeForce/consumer Blackwell) -- one run:
//
//                    +fp32     +fp16
//     dense  k32      85.14    166.78   (wmma_fp8_e4m3 / wmma_fp8_f16)
//     sparse k64     169.71    326.15   (wmma_fp8_sparse / here)
//
// The two effects are independent 2x multipliers and they compose: 3.83x the
// dense fp32-accumulate corner.  Sparsity doubles whatever rate the
// accumulator allows, which is why the fp32-accumulate sparse row only climbs
// back to the uncapped DENSE rate (169.71 ~ 166.78) while this one doubles
// past it.
//
// The result also agrees with INT8, whose int32 accumulator was never capped:
// wmma_int8_sparse.cu measures 327.27 at the same 8-bit width and the same 2:4
// -- within noise of the 326.15 here.  ~327 is the 8-bit 2:4 tensor rate on
// this part, reachable by any accumulator that is not artificially halved.
//
// ARCH FLOOR (why this kernel is not in the sm_89 FP8 group): ptxas accepts the
// instruction but refuses the target --
//
//   error: Instruction 'mma with F16 accumulator and F8 floating point type'
//          not supported on .target 'sm_89'
//
// That is a feature-vs-target rejection, not a parse failure, so the encoding
// exists; Ada simply does not implement it.  Note the asymmetry: Ada DOES do
// the fp16-accumulate FP8 mma dense (wmma_fp8_f16.cu builds for sm_89), so it
// is the combination of sparsity and the fp16 accumulator that Ada lacks.
// CUTLASS likewise wraps only f32-accumulate sparse FP8 (mma_sparse_sm89.h).
// The floor here is set to where the instruction is known to assemble
// (sm_120a/121a), NOT to a documented cutoff -- sm_90/sm_100 are untried and
// the group should be widened if they build.
//
// === Per-thread fragment layout (32 threads/warp, A=row-major, B=col-major) ===
//   A: m16 x k64 @ 2:4 (half non-zero) = 1024 bytes/2 / 32 = 16 B/thread = 4 x .b32
//   B: k64 x  n8                       =  512 bytes    / 32 = 16 B/thread = 4 x .b32
//   metadata: m16 x k64 2:4 selectors  =  128 bytes    / 32 =  4 B/thread = 1 x .b32
//             -- whole .b32 consumed at this shape, so sparsity_selector = 0x0.
//   C/D: m16 x n8 = 128 halfs / 32 threads = 4 halfs = 2 x .b32 (f16x2 packed)
//        per accumulator.  The output tile is m16n8 regardless of K, so this
//        matches wmma_fp8_f16.cu, not the 4 x fp32 of the sparse fp32 kernel.
//
// EIGHT independent chains in a single non-volatile asm block, matching both
// parents.
//
// === Ops accounting ===
// Nominal M*N*K*2 = 16*8*64*2 = 16384 ops per mma.sp, no sparsity multiplier
// on top -- the doubled K already encodes the 2x, exactly as in
// wmma_fp8_sparse.cu, so the two sparse rows are directly comparable.
//
// Per warp ops = 256 outer * 8 chains * (16*8*64*2) = 33,554,432;
// per thread = 1,048,576 (= 16 * COOPMAT_WORK_PER_WI).

#include <cuda_fp16.h>

extern "C" __global__ void wmma_fp8_sparse_f16(float *out, float A)
{
    unsigned int packed = 0x3c3c3c3cu ^ (__float_as_uint(A) & 0x0f0f0f0fu);
    unsigned int a0 = packed, a1 = packed, a2 = packed, a3 = packed;
    unsigned int b0 = packed, b1 = packed, b2 = packed, b3 = packed;
    unsigned int meta = 0xeeeeeeeeu;

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
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 "
              "{%0,%1}, {%16,%17,%18,%19}, {%20,%21,%22,%23}, {%0,%1}, %24, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 "
              "{%2,%3}, {%16,%17,%18,%19}, {%20,%21,%22,%23}, {%2,%3}, %24, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 "
              "{%4,%5}, {%16,%17,%18,%19}, {%20,%21,%22,%23}, {%4,%5}, %24, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 "
              "{%6,%7}, {%16,%17,%18,%19}, {%20,%21,%22,%23}, {%6,%7}, %24, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 "
              "{%8,%9}, {%16,%17,%18,%19}, {%20,%21,%22,%23}, {%8,%9}, %24, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 "
              "{%10,%11}, {%16,%17,%18,%19}, {%20,%21,%22,%23}, {%10,%11}, %24, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 "
              "{%12,%13}, {%16,%17,%18,%19}, {%20,%21,%22,%23}, {%12,%13}, %24, 0x0;\n"
          "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f16.e4m3.e4m3.f16 "
              "{%14,%15}, {%16,%17,%18,%19}, {%20,%21,%22,%23}, {%14,%15}, %24, 0x0;\n"
          : "+r"(c00),"+r"(c01), "+r"(c10),"+r"(c11),
            "+r"(c20),"+r"(c21), "+r"(c30),"+r"(c31),
            "+r"(c40),"+r"(c41), "+r"(c50),"+r"(c51),
            "+r"(c60),"+r"(c61), "+r"(c70),"+r"(c71)
          : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),"r"(b2),"r"(b3),
            "r"(meta));
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
