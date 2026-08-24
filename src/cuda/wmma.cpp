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
    // FP16 WMMA
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.title = "WMMA fp16xfp16+fp32 16x16x16";
      d.resultTag = "wmma_fp16";
      d.description = "Peak speed of the tensor cores -- dedicated units that "
                      "multiply whole 16x16 blocks of numbers in one step rather "
                      "than one value at a time -- on 16-bit inputs with a 32-bit "
                      "running total.  This is the everyday precision of AI work.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "wmma_fp16";
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
      d.title = "FP16 mma.sync m16n8k16+fp16";
      d.resultTag = "wmma_fp16_f16";
      d.description = "The same 16-bit tensor-core maths, but keeping the running "
                      "total in 16 bits too.  On GeForce cards the 32-bit-total "
                      "form is deliberately capped at half rate, so this row is "
                      "where the full tensor-core speed shows up.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp16_f16acc";
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
      d.title = "WMMA bf16xbf16+fp32 16x16x16";
      d.resultTag = "wmma_bf16";
      d.description = "Tensor cores on bfloat16 -- 16 bits arranged for AI work, "
                      "trading digits of accuracy for the number range of a full "
                      "float, which makes training far more forgiving.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "wmma_bf16";
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
      d.title = "TF32 mma.sync m16n8k8+fp32";
      d.resultTag = "wmma_tf32";
      d.description = "Tensor cores on tf32, NVIDIA's trimmed-down stand-in for "
                      "32-bit float: it keeps the full number range but drops "
                      "accuracy to fit the tensor cores.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "wmma_tf32";
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
      d.title = "WMMA fp64xfp64+fp64 8x8x4";
      d.resultTag = "wmma_fp64";
      d.description = "Tensor cores on full 64-bit numbers, for scientific "
                      "computing.  Only the datacenter parts carry this hardware.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "wmma_fp64";
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
      d.title = "FP8(E4M3) mma.sync m16n8k32+fp32";
      d.resultTag = "wmma_fp8_e4m3";
      d.description = "Tensor cores on 8-bit numbers, in the variant that spends "
                      "its bits on accuracy rather than range.  Half the data of "
                      "16-bit per value, so it runs at roughly twice the rate.";
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
      d.title = "FP8(E5M2) mma.sync m16n8k32+fp32";
      d.resultTag = "wmma_fp8_e5m2";
      d.description = "The same 8-bit tensor-core path in the other variant, "
                      "which spends its bits on range rather than accuracy -- the "
                      "one that copes with very large and very small values.";
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
      d.title = "FP8(E4M3) mma.sync m16n8k32+fp16";
      d.resultTag = "wmma_fp8_f16";
      d.description = "The same 8-bit tensor-core maths, but keeping the running "
                      "total in 16 bits rather than 32.  On GeForce cards the "
                      "32-bit-total form is deliberately capped at half rate, so "
                      "this row is where the full 8-bit speed shows up.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp8_e4m3_f16acc";
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
      d.title = "FP8(E4M3) mma.sp 2:4 sparsity m16n8k64+fp32";
      d.resultTag = "wmma_fp8_sparse";
      d.description = "8-bit numbers with structured sparsity: half the values in "
                      "each group of four are known to be zero and are skipped, so "
                      "the hardware does twice the useful work per step.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp8_sparse";
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
      d.title = "FP8(E4M3) mma.sp 2:4 sparsity m16n8k64+fp16";
      d.resultTag = "wmma_fp8_sparse_f16";
      d.description = "8-bit numbers with both tricks at once: half the values in "
                      "each group of four are skipped as known zeros, and the "
                      "running total is kept in 16 bits rather than 32.  The "
                      "fastest arrangement these tensor cores offer for 8-bit.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp8_sparse_f16acc";
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
      d.title = "FP4(E2M1) mma.sync m16n8k32+fp32";
      d.resultTag = "wmma_fp4_e2m1";
      d.description = "Tensor cores on 4-bit numbers, the narrowest format the "
                      "hardware handles.  Blackwell and newer only.";
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
      d.title = "MXFP4(E2M1) mma.sync m16n8k64+fp32";
      d.resultTag = "wmma_mxf4_e2m1";
      d.description = "4-bit numbers with a shared scale factor per block of 32, "
                      "the open MX format.  The scale is what makes 4 bits usable "
                      "for real models rather than a curiosity.";
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
      d.title = "NVFP4(E2M1) mma.sync m16n8k64+fp32";
      d.resultTag = "wmma_nvf4_e2m1";
      d.description = "NVIDIA's own 4-bit block format, with a finer scale factor "
                      "shared by every 16 values instead of every 32 -- more "
                      "accurate than MX at the same width.";
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
      d.title = "MXFP4 mma.sp 2:4 sparsity m16n8k128+fp32";
      d.resultTag = "wmma_mxf4_sparse";
      d.description = "The MX 4-bit path with structured sparsity: half the values "
                      "in each group of four are known to be zero and are skipped, "
                      "so the hardware does twice the useful work per step.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "mxf4_sparse";
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
      d.title = "NVFP4 mma.sp 2:4 sparsity m16n8k128+fp32";
      d.resultTag = "wmma_nvf4_sparse";
      d.description = "NVIDIA's 4-bit block format with the same skip-the-zeros "
                      "trick -- the fastest arrangement these tensor cores offer.";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "nvf4_sparse";
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

  // INT8 WMMA
  {
    int A = 3;
    cuda_compute_desc_t d = {};
    d.title = "WMMA int8xint8+int32 16x16x16";
    d.resultTag = "wmma_int8";
    d.description = "Tensor cores on 8-bit whole numbers with a 32-bit running "
                    "total -- the format quantized neural networks use when they are "
                    "squeezed down to run fast on cheaper hardware.";
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "wmma_int8";
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
    d.title = "INT8 mma.sync m16n8k32+int32";
    d.resultTag = "wmma_int8_k32";
    d.description = "The same 8-bit whole-number maths on NVIDIA's own native "
                    "tile shape rather than the portable one, which the hardware "
                    "feeds more efficiently.";
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "int8_k32";
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
    d.title = "INT8 mma.sp 2:4 sparsity m16n8k64+int32";
    d.resultTag = "wmma_int8_sparse";
    d.description = "8-bit whole numbers with structured sparsity: half the values "
                    "in each group of four are known to be zero and are skipped, so "
                    "the hardware does twice the useful work per step.";
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "int8_sparse";
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
    d.title = "INT4 mma.sync m8n8k32+int32";
    d.resultTag = "wmma_int4";
    d.description = "Tensor cores on 4-bit whole numbers.  Turing through Ada "
                    "only -- NVIDIA dropped the integer 4-bit path afterwards in "
                    "favour of the 4-bit float formats above.";
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "int4";
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
