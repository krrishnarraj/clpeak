#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// Floating-point compute benchmarks
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

int MetalPeak::runComputeSP(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "float ", "compute_sp",  mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name },
        { "float2", "compute_sp2", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name },
        { "float4", "compute_sp4", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name },
        { "float8", "compute_sp8", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Single-precision compute";
    d.resultTag      = "single_precision_compute";
    d.unit        = "gflops";
    d.variants    = variants;
    d.numVariants = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
    d.elemSize    = sizeof(float);
    d.scalarArg   = &A;
    d.scalarSize  = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runComputeHP(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "half ", "compute_hp",  mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name },
        { "half2", "compute_hp2", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name },
        { "half4", "compute_hp4", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name },
        { "half8", "compute_hp8", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Half-precision compute";
    d.resultTag      = "half_precision_compute";
    d.unit        = "gflops";
    d.variants    = variants;
    d.numVariants = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
    d.elemSize    = sizeof(float);
    d.scalarArg   = &A;
    d.scalarSize  = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runComputeMP(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "mp ", "compute_mp",  mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name },
        { "mp2", "compute_mp2", mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name },
        { "mp4", "compute_mp4", mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Mixed-precision compute fp16xfp16+fp32";
    d.resultTag      = "mixed_precision_compute";
    d.unit        = "gflops";
    d.variants    = variants;
    d.numVariants = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
    d.elemSize    = sizeof(float);
    d.scalarArg   = &A;
    d.scalarSize  = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}


#endif // ENABLE_METAL
