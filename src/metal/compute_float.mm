#ifdef ENABLE_METAL
#include "mtl_internal.h"

using xpupeak::Dim; using xpupeak::DType;

// ---------------------------------------------------------------------------
// Floating-point compute benchmarks
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

int MetalPeak::runComputeSP(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "float ", "compute_sp",  mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name, mtlWidthNote(1),
          "compute_sp_alt", nullptr, {{Dim::Width, 1}} },
        { "float2", "compute_sp2", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name, mtlWidthNote(2),
          "compute_sp2_alt", nullptr, {{Dim::Width, 2}} },
        { "float4", "compute_sp4", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name, mtlWidthNote(4),
          "compute_sp4_alt", nullptr, {{Dim::Width, 4}} },
        { "float8", "compute_sp8", mtl_kernels::compute_sp_src, mtl_kernels::compute_sp_name, mtlWidthNote(8),
          nullptr, nullptr, {{Dim::Width, 8}} },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Single-precision compute";
    d.resultTag      = "single_precision_compute";
    // 2: its chain changed shape after it first shipped -- one accumulator
    // (2.1.0), raced shapes (2.1.3), 128-deep trips (3.0.2) -- which moved its
    // numbers.
    d.revision    = 2;
    d.unit        = "flops";
    d.description = "Peak fp32 arithmetic rate of the GPU's ALUs, with no memory "
                    "traffic.";
    d.shape       = TestShape::Homogeneous;
    d.axis        = "vector width";
    d.dims        = {{Dim::Dtype, DType::Fp32}};
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
          "compute_hp_alt", nullptr, {{Dim::Width, 1}} },
        { "half2", "compute_hp2", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name, mtlWidthNote(2),
          "compute_hp2_alt", nullptr, {{Dim::Width, 2}} },
        { "half4", "compute_hp4", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name, mtlWidthNote(4),
          "compute_hp4_alt", nullptr, {{Dim::Width, 4}} },
        { "half8", "compute_hp8", mtl_kernels::compute_hp_src, mtl_kernels::compute_hp_name, mtlWidthNote(8),
          nullptr, nullptr, {{Dim::Width, 8}} },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Half-precision compute";
    d.resultTag      = "half_precision_compute";
    // 2: its chain changed shape after it first shipped -- one accumulator
    // (2.1.0), raced shapes (2.1.3), 128-deep trips (3.0.2) -- which moved its
    // numbers.
    d.revision    = 2;
    d.unit        = "flops";
    d.description = "Peak fp16 arithmetic rate, with fp16 inputs and accumulator.";
    d.shape       = TestShape::Homogeneous;
    d.axis        = "vector width";
    d.dims        = {{Dim::Dtype, DType::Fp16}};
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
        { "mp ", "compute_mp",  mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name, mtlWidthNote(1),
          nullptr, nullptr, {{Dim::Width, 1}} },
        { "mp2", "compute_mp2", mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name, mtlWidthNote(2),
          nullptr, nullptr, {{Dim::Width, 2}} },
        { "mp4", "compute_mp4", mtl_kernels::compute_mp_src, mtl_kernels::compute_mp_name, mtlWidthNote(4),
          nullptr, nullptr, {{Dim::Width, 4}} },
    };
    float A = 1.3f;
    mtl_compute_desc_t d = {};
    d.title       = "Mixed-precision compute fp16xfp16+fp32";
    d.resultTag      = "mixed_precision_compute";
    // 2: its chain changed shape after it first shipped -- one accumulator
    // (2.1.0), raced shapes (2.1.3), 128-deep trips (3.0.2) -- which moved its
    // numbers.
    d.revision    = 2;
    d.unit        = "flops";
    d.description = "Peak rate of fp16 multiplies accumulated in fp32, "
                    "without the matrix engine.";
    d.shape       = TestShape::Homogeneous;
    d.axis        = "vector width";
    // The family's own accumulator is fp32, so the data type says it all.
    d.dims        = {{Dim::Dtype, DType::Fp16}};
    d.variants    = variants;
    d.numVariants = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI   = COMPUTE_FP_WORK_PER_WI;
    d.elemSize    = sizeof(float);
    d.scalarArg   = &A;
    d.scalarSize  = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}


#endif // ENABLE_METAL
