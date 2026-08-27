MSTRINGIFY(

// Mixed-precision MAC: fp16 inputs multiplied, fp32 accumulator.
// Dominant arithmetic path in LLM training/prefill -- distinct from
// compute_hp (fp16 accumulator).
//
// The chain is carried in fp32 and round-trips through fp16 once per MAD_128
// round, not once per MAC.  On shader cores fp16xfp16+fp32 lowers to an fp32
// FMA anyway, so the inner loop is the fp32 chain and the narrowing is what
// puts the fp16 type in the data path -- the same structure the CUDA, ROCm,
// Metal, Vulkan and oneAPI mp kernels use.
//
// It used to narrow after every MAC.  That conversion is an uncounted
// instruction -- it is not in the 4096-op budget -- and sitting in the
// dependent chain once per counted FMA it issued two instructions per FMA and
// capped the reading at half the fp32 rate.  An Arc A380 read mp at 2.78
// TFLOPS against 4.89 for fp32 (57%), where an RTX 5060 sat at 88%.  The old
// comment here claimed the downcast "does not distort the FMA measurement
// meaningfully"; on Alchemist it halved it.
//
// Not to be confused with the even earlier version that carried the fp16
// value itself through the loop and collapsed at vector widths (0.12x the
// squaring rate on Intel's CPU runtime from width 4 up).  Nothing in this loop
// is half; these are fp32 values periodically rounded to fp16 and back.
//
// Chain shape and why: see the MAD chain block in include/common/common.h.
//
// The seed carries a get_local_id term.  It is not decoration: without it
// every operand in this family is uniform across work-items, and a runtime
// that vectorises across work-items can compute the whole chain once and
// broadcast it.  Intel's CPU runtime does exactly that.  Adding the term
// dropped the reported figures by 7.96-8.02x at vector widths 2 through 16 --
// precisely the AVX2 fp32 lane count -- so every mp number this family
// produced on such a runtime had been eight times too high.  Scalar width 1
// was only inflated 1.37x, so the broadcast needed the vector types.  Every
// other backend's mp kernel already seeded from the thread or lane id, which
// is why this only ever affected OpenCL.

\n#if defined(cl_khr_fp16)
\n  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
\n  #define HALF_AVAILABLE
\n#endif

// Plain contracted expression instead of mad() -- see compute_sp_kernels.cl.

\n#undef MAD_4
\n#undef MAD_16
\n#undef MAD_128
\n#undef NARROW
\n
\n#define MAD_4(x, c)     x = (x*x) + c;      x = (x*x) + c;      x = (x*x) + c;      x = (x*x) + c;
\n#define MAD_16(x, c)    MAD_4(x, c);        MAD_4(x, c);        MAD_4(x, c);        MAD_4(x, c);
\n#define MAD_128(x, c)   MAD_16(x, c); MAD_16(x, c); MAD_16(x, c); MAD_16(x, c); \
                          MAD_16(x, c); MAD_16(x, c); MAD_16(x, c); MAD_16(x, c);
\n#define NARROW(HT, FT, x)  x = convert_##FT(convert_##HT(x));
\n

\n
\n#ifdef HALF_AVAILABLE
\n


__kernel void compute_mp_v1(__global float *ptr, float _B)
{
    float x = (float)((half)_B);
    float c = (float)((half)((float)get_local_id(0)));

    for(int i=0; i<16; i++)
    {
        MAD_128(x, c);
        NARROW(half, float, x);
    }

    ptr[get_global_id(0)] = x;
}

__kernel void compute_mp_v2(__global float *ptr, float _B)
{
    float2 x = convert_float2(convert_half2((float2)(_B, _B+1)));
    float2 c = convert_float2(convert_half2((float2)get_local_id(0)));

    for(int i=0; i<8; i++)
    {
        MAD_128(x, c);
        NARROW(half2, float2, x);
    }

    ptr[get_global_id(0)] = x.S0 + x.S1;
}

__kernel void compute_mp_v4(__global float *ptr, float _B)
{
    float4 x = convert_float4(convert_half4((float4)(_B, _B+1, _B+2, _B+3)));
    float4 c = convert_float4(convert_half4((float4)get_local_id(0)));

    for(int i=0; i<4; i++)
    {
        MAD_128(x, c);
        NARROW(half4, float4, x);
    }

    ptr[get_global_id(0)] = x.S0 + x.S1 + x.S2 + x.S3;
}

__kernel void compute_mp_v8(__global float *ptr, float _B)
{
    float8 x = convert_float8(convert_half8((float8)(_B, _B+1, _B+2, _B+3, _B+4, _B+5, _B+6, _B+7)));
    float8 c = convert_float8(convert_half8((float8)get_local_id(0)));

    for(int i=0; i<2; i++)
    {
        MAD_128(x, c);
        NARROW(half8, float8, x);
    }

    ptr[get_global_id(0)] = x.S0 + x.S1 + x.S2 + x.S3 + x.S4 + x.S5 + x.S6 + x.S7;
}

__kernel void compute_mp_v16(__global float *ptr, float _B)
{
    float16 x = convert_float16(convert_half16((float16)(_B, _B+1, _B+2, _B+3, _B+4, _B+5, _B+6, _B+7, _B+8, _B+9, _B+10, _B+11, _B+12, _B+13, _B+14, _B+15)));
    float16 c = convert_float16(convert_half16((float16)get_local_id(0)));

    for(int i=0; i<1; i++)
    {
        MAD_128(x, c);
        NARROW(half16, float16, x);
    }

    ptr[get_global_id(0)] = x.S0 + x.S1 + x.S2 + x.S3 + x.S4 + x.S5 + x.S6 + x.S7 + x.S8 + x.S9 + x.SA + x.SB + x.SC + x.SD + x.SE + x.SF;
}

// ---- affine-chain variants (generated; see mad_chain.cl) ----

__kernel void compute_mp_alt_v1(__global float *ptr, float _B)
{
    AF4_DECL(float, (float)((half)_B), (float)((half)((float)get_local_id(0))))

    for (int i = 0; i < 16; i++)
    {
        MP4_128
        MP4_NARROW(half, float)
    }

    float r = AF4_RES;
    ptr[get_global_id(0)] = r;
}

__kernel void compute_mp_alt_v2(__global float *ptr, float _B)
{
    AF2_DECL(float2, convert_float2(convert_half2((float2)(_B, _B+1))), convert_float2(convert_half2((float2)get_local_id(0))))

    for (int i = 0; i < 8; i++)
    {
        MP2_128
        MP2_NARROW(half2, float2)
    }

    float2 r = AF2_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1;
}

__kernel void compute_mp_alt_v4(__global float *ptr, float _B)
{
    AF1_DECL(float4, convert_float4(convert_half4((float4)(_B, _B+1, _B+2, _B+3))), convert_float4(convert_half4((float4)get_local_id(0))))

    for (int i = 0; i < 4; i++)
    {
        MP1_128
        MP1_NARROW(half4, float4)
    }

    float4 r = AF1_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1 + r.S2 + r.S3;
}

__kernel void compute_mp_alt_v8(__global float *ptr, float _B)
{
    AF1_DECL(float8, convert_float8(convert_half8((float8)(_B, _B+1, _B+2, _B+3, _B+4, _B+5, _B+6, _B+7))), convert_float8(convert_half8((float8)get_local_id(0))))

    for (int i = 0; i < 2; i++)
    {
        MP1_128
        MP1_NARROW(half8, float8)
    }

    float8 r = AF1_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1 + r.S2 + r.S3 + r.S4 + r.S5 + r.S6 + r.S7;
}

__kernel void compute_mp_alt_v16(__global float *ptr, float _B)
{
    AF1_DECL(float16, convert_float16(convert_half16((float16)(_B, _B+1, _B+2, _B+3, _B+4, _B+5, _B+6, _B+7, _B+8, _B+9, _B+10, _B+11, _B+12, _B+13, _B+14, _B+15))), convert_float16(convert_half16((float16)get_local_id(0))))

    for (int i = 0; i < 1; i++)
    {
        MP1_128
        MP1_NARROW(half16, float16)
    }

    float16 r = AF1_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1 + r.S2 + r.S3 + r.S4 + r.S5 + r.S6 + r.S7 + r.S8 + r.S9 + r.SA + r.SB + r.SC + r.SD + r.SE + r.SF;
}

\n#endif      // HALF_AVAILABLE
\n

)
