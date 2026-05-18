#ifdef ENABLE_METAL
#include "mtl_internal.h"

// ---------------------------------------------------------------------------
// Integer compute benchmarks
// ---------------------------------------------------------------------------

int MetalPeak::runComputeInt8DP(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "int8_dp ", "compute_int8_dp",  mtl_kernels::compute_int8_dp_src, mtl_kernels::compute_int8_dp_name },
        { "int8_dp2", "compute_int8_dp2", mtl_kernels::compute_int8_dp_src, mtl_kernels::compute_int8_dp_name },
        { "int8_dp4", "compute_int8_dp4", mtl_kernels::compute_int8_dp_src, mtl_kernels::compute_int8_dp_name },
    };
    int A = 4;
    mtl_compute_desc_t d = {};
    d.title          = "INT8 dot-product compute (emulated) (GOPS)";
    d.resultTag         = "integer_compute_int8_dp";
    d.unit           = "gops";
    d.variants       = variants;
    d.numVariants    = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI      = COMPUTE_INT8_DP_WORK_PER_WI;
    d.elemSize       = sizeof(int);
    d.scalarArg      = &A;
    d.scalarSize     = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}

int MetalPeak::runComputeInt4Packed(MetalDevice &dev, benchmark_config_t &cfg)
{
    static const mtl_compute_variant_t variants[] = {
        { "int4_packed ", "compute_int4_packed",  mtl_kernels::compute_int4_packed_src, mtl_kernels::compute_int4_packed_name },
        { "int4_packed2", "compute_int4_packed2", mtl_kernels::compute_int4_packed_src, mtl_kernels::compute_int4_packed_name },
        { "int4_packed4", "compute_int4_packed4", mtl_kernels::compute_int4_packed_src, mtl_kernels::compute_int4_packed_name },
    };
    int A = 3;
    mtl_compute_desc_t d = {};
    d.title          = "Packed INT4 compute (emulated) (GOPS)";
    d.resultTag         = "int4_packed_compute";
    d.unit           = "gops";
    d.variants       = variants;
    d.numVariants    = sizeof(variants) / sizeof(variants[0]);
    d.workPerWI      = COMPUTE_INT4_PACKED_WORK_PER_WI;
    d.elemSize       = sizeof(int);
    d.scalarArg      = &A;
    d.scalarSize     = sizeof(A);
    return runComputeKernel(dev, cfg, d);
}


#endif // ENABLE_METAL
