MSTRINGIFY(

// Mixed-precision MAC: fp16 inputs multiplied, fp32 accumulator.
// Dominant arithmetic path in LLM training/prefill -- distinct from
// compute_hp (fp16 accumulator).
//
// Each MAD_4 issues 4 mixed-precision multiply-accumulates:
//   a = convert_floatN(x) * convert_floatN(x) + a;
// and writes a back into x via an fp16 downcast, so the compiler can't CSE
// (float)x*(float)x across iterations. The downcast is ~1 cycle on every
// vendor we care about and does not distort the FMA measurement meaningfully.
//
// One live chain per lane (the fp32 accumulator a, with x as its fp16
// shadow), matching the MAD chain rules in include/common/common.h.  There is
// no separate loop invariant here because a is its own addend, and
// a = half(a)*half(a) + a is quadratic, so steps cannot be folded together.
//
// The MAC is a plain contracted expression rather than fma() for the same
// reason compute_sp_kernels.cl avoids mad() -- see the comment there.

\n#if defined(cl_khr_fp16)
\n  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
\n  #define HALF_AVAILABLE
\n#endif

\n#undef MAD_4
\n#undef MAD_16
\n
\n#define MAD_4(HT, FT, x, a) \
    a = convert_##FT(x) * convert_##FT(x) + a; x = convert_##HT(a); \
    a = convert_##FT(x) * convert_##FT(x) + a; x = convert_##HT(a); \
    a = convert_##FT(x) * convert_##FT(x) + a; x = convert_##HT(a); \
    a = convert_##FT(x) * convert_##FT(x) + a; x = convert_##HT(a);
\n#define MAD_16(HT, FT, x, a)  MAD_4(HT,FT,x,a); MAD_4(HT,FT,x,a); MAD_4(HT,FT,x,a); MAD_4(HT,FT,x,a);
\n

\n#ifdef HALF_AVAILABLE
\n

__kernel void compute_mp_v1(__global float *ptr, float _B)
{
    half x = (half)_B;
    float a = _B;

    for(int i=0; i<128; i++)
    {
        MAD_16(half, float, x, a);
    }

    ptr[get_global_id(0)] = a + (float)x;
}

__kernel void compute_mp_v2(__global float *ptr, float _B)
{
    half2 x = (half2)((half)_B, (half)(_B+1));
    float2 a = (float2)(_B, _B+1);

    for(int i=0; i<64; i++)
    {
        MAD_16(half2, float2, x, a);
    }

    float s = a.S0 + a.S1;
    ptr[get_global_id(0)] = s + (float)(x.S0);
}

__kernel void compute_mp_v4(__global float *ptr, float _B)
{
    half4 x = (half4)((half)_B, (half)(_B+1), (half)(_B+2), (half)(_B+3));
    float4 a = (float4)(_B, _B+1, _B+2, _B+3);

    for(int i=0; i<32; i++)
    {
        MAD_16(half4, float4, x, a);
    }

    float s = a.S0 + a.S1 + a.S2 + a.S3;
    ptr[get_global_id(0)] = s + (float)(x.S0);
}

__kernel void compute_mp_v8(__global float *ptr, float _B)
{
    half8 x = (half8)((half)_B, (half)(_B+1), (half)(_B+2), (half)(_B+3),
                      (half)(_B+4), (half)(_B+5), (half)(_B+6), (half)(_B+7));
    float8 a = (float8)(_B, _B+1, _B+2, _B+3, _B+4, _B+5, _B+6, _B+7);

    for(int i=0; i<16; i++)
    {
        MAD_16(half8, float8, x, a);
    }

    float s = a.S0 + a.S1 + a.S2 + a.S3 + a.S4 + a.S5 + a.S6 + a.S7;
    ptr[get_global_id(0)] = s + (float)(x.S0);
}

__kernel void compute_mp_v16(__global float *ptr, float _B)
{
    half16 x = (half16)((half)_B,     (half)(_B+1),  (half)(_B+2),  (half)(_B+3),
                        (half)(_B+4), (half)(_B+5),  (half)(_B+6),  (half)(_B+7),
                        (half)(_B+8), (half)(_B+9),  (half)(_B+10), (half)(_B+11),
                        (half)(_B+12),(half)(_B+13), (half)(_B+14), (half)(_B+15));
    float16 a = (float16)(_B,    _B+1,  _B+2,  _B+3,  _B+4,  _B+5,  _B+6,  _B+7,
                          _B+8,  _B+9,  _B+10, _B+11, _B+12, _B+13, _B+14, _B+15);

    for(int i=0; i<8; i++)
    {
        MAD_16(half16, float16, x, a);
    }

    float8 t = a.lo + a.hi;
    float4 t2 = t.lo + t.hi;
    float2 t3 = t2.lo + t2.hi;
    ptr[get_global_id(0)] = t3.S0 + t3.S1 + (float)(x.S0);
}

\n#endif      // HALF_AVAILABLE
\n

)
