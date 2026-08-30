#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>

// ---------------------------------------------------------------------------
// WMMA + FP8 mma.sync umbrella -- mirrors vkPeak::runCoopMatrix.
// ---------------------------------------------------------------------------

int CudaPeak::runWmma(CudaDevice &dev, benchmark_config_t &cfg, Category category)
{
  // Shared geometry: one warp (32 threads) per block, m16n16k16 tile per
  // wmma fragment, 256 outer iters → COOPMAT_WORK_PER_WI per thread.
  const uint32_t warp = 32;
  const uint32_t outElems = 16 * 16; // M*N

  // ---------------------------------------------------------------------
  // FP cluster -- each variant opens its own <wmma_*> group with the
  // proper unit attribute via runComputeKernel; no umbrella tag here
  // (depth-5 nesting under one would break the v2 logger shim).
  // ---------------------------------------------------------------------
  if (category == Category::FpCompute)
  {
    // One scope for the whole family, opened here rather than per data type:
    // nineteen descs each opening the same test worked, but closed and
    // reopened it nineteen times, which no channel should have to stitch back
    // together.
    auto test = currentDeviceScope->beginTest(
      {"wmma", "Tensor cores (WMMA / mma.sync)", "tflops", Category::Unknown,
       "Peak speed of the tensor cores -- dedicated units that "
       "multiply whole blocks of numbers in one step rather than one value at "
       "a time.  Each reading is a different input format, and several are one "
       "format run a second way: with the running total kept narrower, or with "
       "half the values skipped as known zeros.  Which of them exist at all, "
       "and how much faster the narrow ones go, is most of what separates one "
       "generation of NVIDIA hardware from the next.",
       TestShape::Heterogeneous, "data type"});

    // FP16 WMMA
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "16-bit inputs with a 32-bit running total -- the "
                            "everyday precision of AI work.  Uses the "
                            "portable WMMA 16x16x16 tile.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp16";
      d.kernelName = "wmma_fp16";
      d.blob = &cuda_kernels::wmma_fp16;
      d.workPerWI = COOPMAT_WORK_PER_WI * 4; // 4 parallel chains per kernel
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = outElems;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported;
      d.skipMsg = "WMMA requires sm_70 or newer (Volta+)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // FP16 mma.sync m16n8k16 with fp16 accumulate (PTX) -- Ampere+.  Full-rate
    // fp16 path: on GeForce fp16+fp32-accum is half rate (the +fp32 test above
    // ~42), fp16+fp16-accum reaches the ~78 cuBLASLt fp16 peak.
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "The same 16-bit maths, keeping the running total "
                            "in 16 bits too.  On GeForce cards the 32-bit- "
                            "total form is deliberately capped at half rate, "
                            "so this is where the full tensor-core speed "
                            "shows up.  mma.sync m16n8k16.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp16 f16acc";
      d.kernelName = "wmma_fp16_f16";
      d.blob = &cuda_kernels::wmma_fp16_f16;
      d.workPerWI = COOPMAT_WORK_PER_WI * 4; // 8 chains * m16n8k16
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = warp; // one fp32 reduction per thread
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.bf16Supported; // sm_80+
      d.skipMsg = "FP16 mma.sync m16n8k16 requires sm_80 or newer (Ampere+)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // BF16 WMMA
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "bfloat16 -- 16 bits arranged for AI work, "
                            "trading digits of accuracy for the number range "
                            "of a full float, which makes training far more "
                            "forgiving.  WMMA 16x16x16.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "bf16";
      d.kernelName = "wmma_bf16";
      d.blob = &cuda_kernels::wmma_bf16;
      d.workPerWI = COOPMAT_WORK_PER_WI * 4;
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = outElems;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.bf16Supported;
      d.skipMsg = "bf16 WMMA requires sm_80 or newer (Ampere+)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // TF32 mma.sync m16n8k8 (native tile, PTX) -- Ampere+.  Replaces the old
    // wmma-fragment m16n16k8 path, which under-saturated Blackwell (~10.7); the
    // native tile reaches the ~20.6 cuBLASLt tf32 peak.
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "NVIDIA's trimmed-down stand-in for 32-bit float: "
                            "the full number range, with accuracy dropped to "
                            "fit the tensor cores.  mma.sync m16n8k8.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "tf32";
      d.kernelName = "wmma_tf32";
      d.blob = &cuda_kernels::wmma_tf32;
      d.workPerWI = COOPMAT_WORK_PER_WI * 2; // 8 chains * m16n8k8 (half K of fp16)
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.tf32GemmSupported;
      d.skipMsg = "TF32 mma.sync m16n8k8 requires sm_80 or newer (Ampere+)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // FP64 WMMA m8n8k4 -- Ampere+ DP tensor cores
    {
      double A = 1.3;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "Full 64-bit numbers, for scientific computing "
                            "rather than AI.  Only the datacenter parts carry "
                            "this hardware.  WMMA 8x8x4.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp64";
      d.kernelName = "wmma_fp64";
      d.blob = &cuda_kernels::wmma_fp64;
      d.workPerWI = COOPMAT_WORK_PER_WI; // 1024 outer iters bring this to par
      d.elemSize = sizeof(double);
      d.blockSize = warp;
      d.outElemsPerBlock = 8 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.dpTensorSupported;
      d.skipMsg = "FP64 WMMA requires sm_80 or newer (Ampere+)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // FP8 mma.sync E4M3 (PTX) - sm_89+
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "8-bit floats spending their bits on accuracy "
                            "rather than range.  Half the data of 16-bit per "
                            "value, so roughly twice the rate.  mma.sync "
                            "m16n8k32.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp8_e4m3";
      d.kernelName = "wmma_fp8_e4m3";
      d.blob = &cuda_kernels::wmma_fp8_e4m3;
      d.workPerWI = COOPMAT_WORK_PER_WI * 8; // 8 parallel chains for FP8
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp8MmaSupported;
      d.skipMsg = "FP8 mma.sync requires sm_89 or newer (Ada/Hopper+)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // FP8 mma.sync E5M2 (PTX) - sm_89+
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "The other 8-bit float, spending its bits on "
                            "range rather than accuracy -- the one that copes "
                            "with very large and very small values.  mma.sync "
                            "m16n8k32.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp8_e5m2";
      d.kernelName = "wmma_fp8_e5m2";
      d.blob = &cuda_kernels::wmma_fp8_e5m2;
      d.workPerWI = COOPMAT_WORK_PER_WI * 8;
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp8MmaSupported;
      d.skipMsg = "FP8 mma.sync requires sm_89 or newer (Ada/Hopper+)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // FP8 mma.sync E4M3 with fp16 accumulate (PTX) - sm_89+.  Same shape as the
    // +fp32 E4M3 row above, so the pair isolates the accumulator width: on
    // GeForce the fp32-accumulate tensor-core paths are held at half rate.
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "8-bit inputs with the running total kept in 16 "
                            "bits rather than 32.  As with fp16, GeForce caps "
                            "the 32-bit-total form at half rate, so this is "
                            "where the full 8-bit speed shows up.  mma.sync "
                            "m16n8k32.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp8_e4m3 f16acc";
      d.kernelName = "wmma_fp8_f16";
      d.blob = &cuda_kernels::wmma_fp8_f16;
      d.workPerWI = COOPMAT_WORK_PER_WI * 8; // 8 chains * m16n8k32, as the +fp32 row
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = warp; // one fp32 reduction per thread
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp8MmaSupported;
      d.skipMsg = "FP8 mma.sync with fp16 accumulate requires sm_89 or newer (Ada/Hopper+)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // FP8 mma.sp 2:4 structured sparsity E4M3 m16n8k64 (PTX) - sm_89+
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "8-bit floats with structured sparsity: half the "
                            "values in each group of four are known to be "
                            "zero and are skipped, so the hardware does twice "
                            "the useful work per step.  mma.sp m16n8k64.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp8_e4m3 2:4";
      d.kernelName = "wmma_fp8_sparse";
      d.blob = &cuda_kernels::wmma_fp8_sparse;
      d.workPerWI = COOPMAT_WORK_PER_WI * 16; // 8 chains * k64 = 2x dense k32
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp8MmaSparseSupported;
      d.skipMsg = "FP8 mma.sp 2:4 sparsity requires sm_89 or newer (Ada/Hopper+)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // FP8 mma.sp 2:4 sparsity E4M3 with fp16 accumulate (PTX) - sm_89+.  Both
    // effects at once: the row above is still held to 2x the CAPPED fp32-accum
    // rate, so lifting the accumulator is what should let sparsity double the
    // uncapped rate instead.
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "Both tricks at once -- zeros skipped and the "
                            "running total kept at 16 bits.  The fastest "
                            "arrangement these tensor cores offer for 8 bits. "
                            "mma.sp m16n8k64.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp8_e4m3 2:4 f16acc";
      d.kernelName = "wmma_fp8_sparse_f16";
      d.blob = &cuda_kernels::wmma_fp8_sparse_f16;
      d.workPerWI = COOPMAT_WORK_PER_WI * 16; // 8 chains * k64, as the +fp32 row
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = warp; // one fp32 reduction per thread
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp8MmaSparseF16Supported;
      d.skipMsg = "FP8 mma.sp 2:4 sparsity with fp16 accumulate requires Blackwell sm_120a or newer! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // FP4 mma.sync E2M1 (PTX) - Blackwell sm_120a+
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "4-bit floats, the narrowest format the hardware "
                            "handles, with no shared scale factor.  Blackwell "
                            "and newer only.  mma.sync m16n8k32.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp4_e2m1";
      d.kernelName = "wmma_fp4_e2m1";
      d.blob = &cuda_kernels::wmma_fp4_e2m1;
      d.workPerWI = COOPMAT_WORK_PER_WI * 8;
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp4MmaSupported;
      d.skipMsg = "FP4 mma.sync requires Blackwell sm_120a or newer! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // MXFP4 mma.sync E2M1 + UE8M0 block scale (PTX) - Blackwell sm_120a+
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "4-bit floats with a scale factor shared by each "
                            "block of 32 values, the open MX format.  The "
                            "scale is what makes 4 bits usable for real "
                            "models rather than a curiosity.  mma.sync "
                            "m16n8k64.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "mxf4_e2m1";
      d.kernelName = "wmma_mxf4_e2m1";
      d.blob = &cuda_kernels::wmma_mxf4_e2m1;
      d.workPerWI = COOPMAT_WORK_PER_WI * 16;
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp4MmaSupported;
      d.skipMsg = "MXFP4 mma.sync requires Blackwell sm_120a or newer! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // NVFP4 mma.sync E2M1 + UE4M3 block scale (PTX) - Blackwell sm_120a+
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "NVIDIA's own 4-bit block format, with a finer "
                            "scale shared by every 16 values instead of every "
                            "32 -- more accurate than MX at the same width. "
                            "mma.sync m16n8k64.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "nvf4_e2m1";
      d.kernelName = "wmma_nvf4_e2m1";
      d.blob = &cuda_kernels::wmma_nvf4_e2m1;
      d.workPerWI = COOPMAT_WORK_PER_WI * 16;
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp4MmaSupported;
      d.skipMsg = "NVFP4 mma.sync requires Blackwell sm_120a or newer! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // MXFP4 mma.sp 2:4 sparsity E2M1 + UE8M0 block scale (PTX) - Blackwell sm_120a+
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "The MX 4-bit path with zeros skipped as well, "
                            "two per group of four.  mma.sp m16n8k128.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "mxf4_e2m1 2:4";
      d.kernelName = "wmma_mxf4_sparse";
      d.blob = &cuda_kernels::wmma_mxf4_sparse;
      d.workPerWI = COOPMAT_WORK_PER_WI * 32; // 8 chains * k128 = 2x dense k64
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp4MmaSparseSupported;
      d.skipMsg = "MXFP4 mma.sp 2:4 sparsity requires Blackwell sm_120a or newer! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // NVFP4 mma.sp 2:4 sparsity E2M1 + UE4M3 block scale (PTX) - Blackwell sm_120a+
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.scope = &test;
      d.metricDescription = "NVIDIA's 4-bit block format with zeros skipped "
                            "too -- the fastest arrangement these tensor "
                            "cores offer.  mma.sp m16n8k128.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "nvf4_e2m1 2:4";
      d.kernelName = "wmma_nvf4_sparse";
      d.blob = &cuda_kernels::wmma_nvf4_sparse;
      d.workPerWI = COOPMAT_WORK_PER_WI * 32; // 8 chains * k128 = 2x dense k64
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp4MmaSparseSupported;
      d.skipMsg = "NVFP4 mma.sp 2:4 sparsity requires Blackwell sm_120a or newer! Skipped";
      runComputeKernel(dev, cfg, d);
    }
  }

  // ---------------------------------------------------------------------
  // Integer / binary cluster -- each variant opens its own <wmma_*> group
  // with the proper unit attribute via runComputeKernel; no umbrella tag.
  // The fp/int split is preserved by the per-variant unit -> category
  // derivation in the dump pipeline.
  // ---------------------------------------------------------------------

  if (category != Category::IntCompute)
    return 0;

  // The same test, reopened for the integer phase: the tensor cores do not
  // become different hardware because the numbers are whole.  Its readings are
  // measured in ops, so this opening declares that unit and each carries it.
  auto test = currentDeviceScope->beginTest(
    {"wmma", "Tensor cores (WMMA / mma.sync)", "tops", Category::Unknown,
     "Peak speed of the tensor cores -- dedicated units that "
       "multiply whole blocks of numbers in one step rather than one value at "
       "a time.  Each reading is a different input format, and several are one "
       "format run a second way: with the running total kept narrower, or with "
       "half the values skipped as known zeros.  Which of them exist at all, "
       "and how much faster the narrow ones go, is most of what separates one "
       "generation of NVIDIA hardware from the next.",
     TestShape::Heterogeneous, "data type"});

  // INT8 WMMA
  {
    int A = 3;
    cuda_compute_desc_t d = {};
    d.scope = &test;
    d.metricDescription = "8-bit whole numbers with a 32-bit running total -- "
                          "the format quantized neural networks use when they "
                          "are squeezed down to run fast on cheaper hardware. "
                          "Portable WMMA 16x16x16 tile.";
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "int8";
    d.metricUnit = "tops";
    d.kernelName = "wmma_int8";
    d.blob = &cuda_kernels::wmma_int8;
    d.workPerWI = COOPMAT_WORK_PER_WI * 4;
    d.elemSize = sizeof(int);
    d.blockSize = warp;
    d.outElemsPerBlock = outElems;
    d.scalarArg = &A;
    d.scalarSize = sizeof(A);
    d.skip = !dev.info.wmmaSupported || !dev.info.wmmaInt8Supported;
    d.skipMsg = "INT8 WMMA requires sm_72 or newer (Turing+)! Skipped";
    runComputeKernel(dev, cfg, d);
  }
  // INT8 mma.sync K=32 (NVIDIA-native tile via inline PTX)
  {
    int A = 3;
    cuda_compute_desc_t d = {};
    d.scope = &test;
    d.metricDescription = "The same 8-bit whole-number maths on NVIDIA's "
                          "native tile shape rather than the portable one, "
                          "which the hardware feeds more efficiently. "
                          "mma.sync m16n8k32.";
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "int8 k32";
    d.metricUnit = "tops";
    d.kernelName = "wmma_int8_k32";
    d.blob = &cuda_kernels::wmma_int8_k32;
    d.workPerWI = COOPMAT_WORK_PER_WI * 4;
    d.elemSize = sizeof(int);
    d.blockSize = warp;
    d.outElemsPerBlock = 16 * 8;
    d.scalarArg = &A;
    d.scalarSize = sizeof(A);
    d.skip = !dev.info.wmmaSupported || !dev.info.wmmaInt8Supported;
    d.skipMsg = "INT8 mma.sync K=32 requires sm_72 or newer (Turing+)! Skipped";
    runComputeKernel(dev, cfg, d);
  }
  // INT8 mma.sp 2:4 structured sparsity m16n8k64 -- sm_80+.  k64 is the
  // doubled-K counterpart of the dense m16n8k32 row above, which is what turns
  // 2:4 into throughput; see the shape note in the kernel.
  {
    int A = 3;
    cuda_compute_desc_t d = {};
    d.scope = &test;
    d.metricDescription = "8-bit whole numbers with zeros skipped, two per "
                          "group of four, so the hardware does twice the "
                          "useful work per step.  mma.sp m16n8k64.";
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "int8 2:4";
    d.metricUnit = "tops";
    d.kernelName = "wmma_int8_sparse";
    d.blob = &cuda_kernels::wmma_int8_sparse;
    d.workPerWI = COOPMAT_WORK_PER_WI * 16; // 8 chains * k64 = 2x dense k32
    d.elemSize = sizeof(int);
    d.blockSize = warp;
    d.outElemsPerBlock = 16 * 8;
    d.scalarArg = &A;
    d.scalarSize = sizeof(A);
    d.skip = !dev.info.wmmaSupported || !dev.info.int8MmaSparseSupported;
    d.skipMsg = "INT8 mma.sp 2:4 sparsity requires sm_80 or newer (Ampere+)! Skipped";
    runComputeKernel(dev, cfg, d);
  }
  // INT4 mma.sync m8n8k32 -- sm_75..sm_89
  {
    int A = 3;
    cuda_compute_desc_t d = {};
    d.scope = &test;
    d.metricDescription = "4-bit whole numbers.  Turing through Ada only -- "
                          "NVIDIA dropped the integer 4-bit path afterwards "
                          "in favour of the 4-bit float formats above. "
                          "mma.sync m8n8k32.";
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "int4";
    d.metricUnit = "tops";
    d.kernelName = "wmma_int4";
    d.blob = &cuda_kernels::wmma_int4;
    d.workPerWI = COOPMAT_WORK_PER_WI * 2; // 256 outer * 4 chains * 8*8*32*2 / 32
    d.elemSize = sizeof(int);
    d.blockSize = warp;
    d.outElemsPerBlock = 8 * 8;
    d.scalarArg = &A;
    d.scalarSize = sizeof(A);
    d.skip = !dev.info.wmmaSupported || !dev.info.int4MmaSupported;
    d.skipMsg = "INT4 mma.sync requires sm_75..sm_89 (Turing/Ampere/Ada)! Skipped";
    runComputeKernel(dev, cfg, d);
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Global bandwidth (CUDA)
// ---------------------------------------------------------------------------


#endif // ENABLE_CUDA
