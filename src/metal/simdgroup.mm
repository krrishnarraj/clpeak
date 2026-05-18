#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// SIMD-group matrix multiply (Apple tensor cores)
// ---------------------------------------------------------------------------

int MetalPeak::runSimdgroupMatrixInt(MetalDevice &dev, benchmark_config_t &cfg)
{
    int A = 1;
    mtl_compute_desc_t d = {};
    d.title            = "simdgroup_matrix int8xint8+int32 8x8x8 (TOPS)";
    d.resultTag           = "simdgroup_matrix_int8";
    d.unit             = "tops";
    d.unitDivider      = 1e12;
    d.metricLabel      = "simdgroup_int8";
    d.kernelName       = "simdgroup_matrix_int8";
    d.src              = mtl_kernels::simdgroup_matrix_int8_src;
    d.srcName          = mtl_kernels::simdgroup_matrix_int8_name;
    d.workPerWI        = MTL_SIMDGROUP_WORK_PER_WI;
    d.elemSize         = sizeof(int);
    d.threadsPerGroup  = 32;
    d.outElemsPerGroup = 64;
    d.scalarArg        = &A;
    d.scalarSize       = sizeof(A);
    d.skip             = !dev.info.simdgroupMatrixInt8Supported;
    d.skipMsg          = "int8 simdgroup_matrix requires Apple9 (M3) or newer! Skipped";
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runSimdgroupMatrix(MetalDevice &dev, benchmark_config_t &cfg)
{
    {
        float A = 1.3f;
        mtl_compute_desc_t d = {};
        d.title            = "simdgroup_matrix fp16xfp16+fp32 8x8x8 (TFLOPS)";
        d.resultTag           = "simdgroup_matrix_fp16";
        d.unit             = "tflops";
        d.unitDivider      = 1e12;
        d.metricLabel      = "simdgroup_fp16";
        d.kernelName       = "simdgroup_matrix_fp16";
        d.src              = mtl_kernels::simdgroup_matrix_fp16_src;
        d.srcName          = mtl_kernels::simdgroup_matrix_fp16_name;
        d.workPerWI        = MTL_SIMDGROUP_WORK_PER_WI;
        d.elemSize         = sizeof(float);
        d.threadsPerGroup  = 32;
        d.outElemsPerGroup = 64;
        d.scalarArg        = &A;
        d.scalarSize       = sizeof(A);
        d.skip             = !dev.info.simdgroupMatrixFP16Supported;
        d.skipMsg          = "simdgroup_matrix requires Apple7 (M1) or newer! Skipped";
        runComputeKernel(dev, cfg, d);
    }
    {
        float A = 1.3f;
        mtl_compute_desc_t d = {};
        d.title            = "simdgroup_matrix bf16xbf16+fp32 8x8x8 (TFLOPS)";
        d.resultTag           = "simdgroup_matrix_bf16";
        d.unit             = "tflops";
        d.unitDivider      = 1e12;
        d.metricLabel      = "simdgroup_bf16";
        d.kernelName       = "simdgroup_matrix_bf16";
        d.src              = mtl_kernels::simdgroup_matrix_bf16_src;
        d.srcName          = mtl_kernels::simdgroup_matrix_bf16_name;
        d.workPerWI        = MTL_SIMDGROUP_WORK_PER_WI;
        d.elemSize         = sizeof(float);
        d.threadsPerGroup  = 32;
        d.outElemsPerGroup = 64;
        d.scalarArg        = &A;
        d.scalarSize       = sizeof(A);
        d.skip             = !dev.info.simdgroupMatrixFP16Supported || !dev.info.simdgroupMatrixBF16Supported;
        d.skipMsg          = "bf16 simdgroup_matrix requires Apple9 (M3) or newer! Skipped";
        runComputeKernel(dev, cfg, d);
    }
    return 0;
}


#endif // ENABLE_METAL
