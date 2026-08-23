#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// Floating-point compute benchmarks
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

int MetalPeak::runComputeSP(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "float ", "compute_sp",  mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name, mtlWidthNote(1),
          "compute_sp_alt" },
        { "float2", "compute_sp2", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name, mtlWidthNote(2),
          "compute_sp2_alt" },
        { "float4", "compute_sp4", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name, mtlWidthNote(4),
          "compute_sp4_alt" },
        { "float8", "compute_sp8", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name, mtlWidthNote(8) },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Single-precision compute";
    d.resultTag      = "single_precision_compute";
    d.unit        = "gflops";
    d.description = "Peak arithmetic speed of the GPU's shader cores on 32-bit "
                    "fractional numbers -- the ordinary float type.  Nothing "
                    "touches memory, so only the arithmetic units limit the rate.";
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
        { "half ", "compute_hp",  mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name, mtlWidthNote(1),
          "compute_hp_alt" },
        { "half2", "compute_hp2", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name, mtlWidthNote(2),
          "compute_hp2_alt" },
        { "half4", "compute_hp4", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name, mtlWidthNote(4),
          "compute_hp4_alt" },
        { "half8", "compute_hp8", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name, mtlWidthNote(8) },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Half-precision compute";
    d.resultTag      = "half_precision_compute";
    d.unit        = "gflops";
    d.description = "Peak arithmetic speed on 16-bit fractional numbers -- half the "
                    "size of a normal float, and what graphics and on-device AI "
                    "mostly run on.";
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
        { "mp ", "compute_mp",  mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name, mtlWidthNote(1) },
        { "mp2", "compute_mp2", mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name, mtlWidthNote(2) },
        { "mp4", "compute_mp4", mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name, mtlWidthNote(4) },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Mixed-precision compute fp16xfp16+fp32";
    d.resultTag      = "mixed_precision_compute";
    d.unit        = "gflops";
    d.description = "Peak speed when the GPU multiplies 16-bit numbers but keeps the "
                    "running total in 32 bits -- the accuracy-preserving pattern AI "
                    "code uses.";
    d.variants    = variants;
    d.numVariants = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
    d.elemSize    = sizeof(float);
    d.scalarArg   = &A;
    d.scalarSize  = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}


#endif // ENABLE_METAL
