#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// SIMD-group matrix multiply (Apple tensor cores)
//
// One test, one reading per data type.  They are separate measurements, not a
// sweep -- fp16 and bf16 are different formats the same instruction runs at
// different rates -- so the test is heterogeneous and every reading is shown.
// A device that supports only some of the types skips the rest as individual
// readings rather than losing the whole test.
// ---------------------------------------------------------------------------

int MetalPeak::runSimdgroupMatrix(MetalDevice &dev, benchmark_config_t &cfg)
{
    // bf16 arrived with Apple9 (M3); fp16 has been there since Apple7 (M1).
    // Not static: the gate is a runtime property of the device.
    const mtl_compute_variant_t variants[] = {
        { "fp16", "simdgroup_matrix_fp16",
          mtl_kernels::simdgroup_matrix_fp16_src,
          mtl_kernels::simdgroup_matrix_fp16_name,
          "Half-precision inputs with a 32-bit running total -- the format "
          "Apple's matrix hardware is built around.",
          nullptr,
          dev.info.simdgroupMatrixFP16Supported
              ? nullptr
              : "simdgroup_matrix requires Apple7 (M1) or newer" },

        { "bf16", "simdgroup_matrix_bf16",
          mtl_kernels::simdgroup_matrix_bf16_src,
          mtl_kernels::simdgroup_matrix_bf16_name,
          "bfloat16 inputs -- 16 bits arranged for AI work, trading digits of "
          "accuracy for a wider number range.",
          nullptr,
          (dev.info.simdgroupMatrixFP16Supported &&
           dev.info.simdgroupMatrixBF16Supported)
              ? nullptr
              : "bf16 simdgroup_matrix requires Apple9 (M3) or newer" },
    };

    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title            = "simdgroup_matrix 8x8x8";
    d.resultTag        = "simdgroup_matrix";
    d.unit             = "flops";
    d.description      = "Peak speed of Apple's matrix instruction, which multiplies "
                         "whole 8x8 blocks in one step instead of one value at a "
                         "time -- the GPU's answer to a tensor core.  Each reading "
                         "is a different input format; all of them accumulate into "
                         "32-bit floats.";
    d.shape            = TestShape::Heterogeneous;
    d.axis             = "data type";
    d.variants         = variants;
    d.numVariants      = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI        = MTL_SIMDGROUP_WORK_PER_WI;
    d.elemSize         = sizeof(float);
    d.threadsPerGroup  = 32;
    d.outElemsPerGroup = 64;
    d.scalarArg        = &A;
    d.scalarSize       = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}


#endif // ENABLE_METAL
