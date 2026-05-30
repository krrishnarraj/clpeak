#ifdef ENABLE_CUDA

#include <cuda/cuda_peak.h>
#include <common/common.h>
#include <cstdio>
#include <sstream>

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
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "wmma_fp16";
      d.kernelName = "wmma_fp16";
      d.src = cuda_kernels::wmma_fp16_src;
      d.srcName = cuda_kernels::wmma_fp16_name;
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
    // BF16 WMMA
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.title = "WMMA bf16xbf16+fp32 16x16x16";
      d.resultTag = "wmma_bf16";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "wmma_bf16";
      d.kernelName = "wmma_bf16";
      d.src = cuda_kernels::wmma_bf16_src;
      d.srcName = cuda_kernels::wmma_bf16_name;
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
    // TF32 WMMA m16n16k8 -- Ampere+
    {
      float A = 1.3f;
      cuda_compute_desc_t d = {};
      d.title = "WMMA tf32xtf32+fp32 16x16x8";
      d.resultTag = "wmma_tf32";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "wmma_tf32";
      d.kernelName = "wmma_tf32";
      d.src = cuda_kernels::wmma_tf32_src;
      d.srcName = cuda_kernels::wmma_tf32_name;
      d.workPerWI = COOPMAT_WORK_PER_WI * 2; // m16n16k8 = half the K of fp16
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = outElems;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.tf32GemmSupported;
      d.skipMsg = "TF32 WMMA requires sm_80 or newer (Ampere+)! Skipped";
      runComputeKernel(dev, cfg, d);
    }
    // FP64 WMMA m8n8k4 -- Ampere+ DP tensor cores
    {
      double A = 1.3;
      cuda_compute_desc_t d = {};
      d.title = "WMMA fp64xfp64+fp64 8x8x4";
      d.resultTag = "wmma_fp64";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "wmma_fp64";
      d.kernelName = "wmma_fp64";
      d.src = cuda_kernels::wmma_fp64_src;
      d.srcName = cuda_kernels::wmma_fp64_name;
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
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp8_e4m3";
      d.kernelName = "wmma_fp8_e4m3";
      d.src = cuda_kernels::wmma_fp8_e4m3_src;
      d.srcName = cuda_kernels::wmma_fp8_e4m3_name;
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
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp8_e5m2";
      d.kernelName = "wmma_fp8_e5m2";
      d.src = cuda_kernels::wmma_fp8_e5m2_src;
      d.srcName = cuda_kernels::wmma_fp8_e5m2_name;
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
    // FP4 mma.sync E2M1 (PTX) - Blackwell sm_120a+
    {
      float A = 1.3f;
      std::stringstream archOpt;
      archOpt << "--gpu-architecture=sm_" << dev.info.major << dev.info.minor << "a";
      std::string archOptStr = archOpt.str();
      const char *fp4Opts[] = {archOptStr.c_str()};
      cuda_compute_desc_t d = {};
      d.title = "FP4(E2M1) mma.sync m16n8k32+fp32";
      d.resultTag = "wmma_fp4_e2m1";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "fp4_e2m1";
      d.kernelName = "wmma_fp4_e2m1";
      d.src = cuda_kernels::wmma_fp4_e2m1_src;
      d.srcName = cuda_kernels::wmma_fp4_e2m1_name;
      d.workPerWI = COOPMAT_WORK_PER_WI * 8;
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp4MmaSupported;
      d.skipMsg = "FP4 mma.sync requires Blackwell sm_120a or newer! Skipped";
      d.extraNvrtcOpts = fp4Opts;
      d.numExtraNvrtcOpts = 1;
      runComputeKernel(dev, cfg, d);
    }
    // MXFP4 mma.sync E2M1 + UE8M0 block scale (PTX) - Blackwell sm_120a+
    {
      float A = 1.3f;
      std::stringstream archOpt;
      archOpt << "--gpu-architecture=sm_" << dev.info.major << dev.info.minor << "a";
      std::string archOptStr = archOpt.str();
      const char *fp4Opts[] = {archOptStr.c_str()};
      cuda_compute_desc_t d = {};
      d.title = "MXFP4(E2M1) mma.sync m16n8k64+fp32";
      d.resultTag = "wmma_mxf4_e2m1";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "mxf4_e2m1";
      d.kernelName = "wmma_mxf4_e2m1";
      d.src = cuda_kernels::wmma_mxf4_e2m1_src;
      d.srcName = cuda_kernels::wmma_mxf4_e2m1_name;
      d.workPerWI = COOPMAT_WORK_PER_WI * 16;
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp4MmaSupported;
      d.skipMsg = "MXFP4 mma.sync requires Blackwell sm_120a or newer! Skipped";
      d.extraNvrtcOpts = fp4Opts;
      d.numExtraNvrtcOpts = 1;
      runComputeKernel(dev, cfg, d);
    }
    // NVFP4 mma.sync E2M1 + UE4M3 block scale (PTX) - Blackwell sm_120a+
    {
      float A = 1.3f;
      std::stringstream archOpt;
      archOpt << "--gpu-architecture=sm_" << dev.info.major << dev.info.minor << "a";
      std::string archOptStr = archOpt.str();
      const char *fp4Opts[] = {archOptStr.c_str()};
      cuda_compute_desc_t d = {};
      d.title = "NVFP4(E2M1) mma.sync m16n8k64+fp32";
      d.resultTag = "wmma_nvf4_e2m1";
      d.unit = "tflops";
      d.unitDivider = 1e12;
      d.metricLabel = "nvf4_e2m1";
      d.kernelName = "wmma_nvf4_e2m1";
      d.src = cuda_kernels::wmma_nvf4_e2m1_src;
      d.srcName = cuda_kernels::wmma_nvf4_e2m1_name;
      d.workPerWI = COOPMAT_WORK_PER_WI * 16;
      d.elemSize = sizeof(float);
      d.blockSize = warp;
      d.outElemsPerBlock = 16 * 8;
      d.scalarArg = &A;
      d.scalarSize = sizeof(A);
      d.skip = !dev.info.wmmaSupported || !dev.info.fp4MmaSupported;
      d.skipMsg = "NVFP4 mma.sync requires Blackwell sm_120a or newer! Skipped";
      d.extraNvrtcOpts = fp4Opts;
      d.numExtraNvrtcOpts = 1;
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
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "wmma_int8";
    d.kernelName = "wmma_int8";
    d.src = cuda_kernels::wmma_int8_src;
    d.srcName = cuda_kernels::wmma_int8_name;
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
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "int8_k32";
    d.kernelName = "wmma_int8_k32";
    d.src = cuda_kernels::wmma_int8_k32_src;
    d.srcName = cuda_kernels::wmma_int8_k32_name;
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
  // INT8 mma.sp 2:4 structured sparsity m16n8k32 -- sm_80+
  {
    int A = 3;
    cuda_compute_desc_t d = {};
    d.title = "INT8 mma.sp 2:4 sparsity m16n8k32+int32";
    d.resultTag = "wmma_int8_sparse";
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "int8_sparse";
    d.kernelName = "wmma_int8_sparse";
    d.src = cuda_kernels::wmma_int8_sparse_src;
    d.srcName = cuda_kernels::wmma_int8_sparse_name;
    d.workPerWI = COOPMAT_WORK_PER_WI * 4;
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
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "int4";
    d.kernelName = "wmma_int4";
    d.src = cuda_kernels::wmma_int4_src;
    d.srcName = cuda_kernels::wmma_int4_name;
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
  // BMMA b1 mma.sync m8n8k128 (XOR-popc) -- sm_75+
  {
    int A = 3;
    cuda_compute_desc_t d = {};
    d.title = "BMMA b1 mma.sync m8n8k128+int32 xor.popc";
    d.resultTag = "wmma_bmma_b1";
    d.unit = "tops";
    d.unitDivider = 1e12;
    d.metricLabel = "bmma_b1";
    d.kernelName = "wmma_bmma_b1";
    d.src = cuda_kernels::wmma_bmma_b1_src;
    d.srcName = cuda_kernels::wmma_bmma_b1_name;
    d.workPerWI = COOPMAT_WORK_PER_WI * 8; // 256 outer * 4 chains * 8*8*128*2 / 32
    d.elemSize = sizeof(int);
    d.blockSize = warp;
    d.outElemsPerBlock = 8 * 8;
    d.scalarArg = &A;
    d.scalarSize = sizeof(A);
    d.skip = !dev.info.wmmaSupported || !dev.info.bmmaSupported;
    d.skipMsg = "BMMA b1 requires sm_75 or newer (Turing+)! Skipped";
    runComputeKernel(dev, cfg, d);
  }

  return 0;
}

// ---------------------------------------------------------------------------
// Global bandwidth (CUDA)
// ---------------------------------------------------------------------------


#endif // ENABLE_CUDA
