MSTRINGIFY(

// Mixed-precision MAC: fp16 inputs multiplied, fp32 accumulator.
// Dominant arithmetic path in LLM training/prefill -- distinct from
// compute_hp (fp16 accumulator).
//
// Each MAD_4 issues 4 mixed-precision fma's:
//   a = fma(convert_floatN(x), convert_floatN(y), a);
// and writes a back into x (resp. y) via an fp16 downcast, so the
// compiler can't CSE (float)x*(float)y across iterations. The downcast
// is ~1 cycle on every vendor we care about and does not distort the
// FMA measurement meaningfully.

\n#if defined(cl_khr_fp16)
\n  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
\n  #define HALF_AVAILABLE
\n#endif

\n#undef MAD_4
\n#undef MAD_16
\n
\n#define MAD_4(HT, FT, x, y, a) \
    a = fma(convert_##FT(x), convert_##FT(y), a); x = convert_##HT(a); \
    a = fma(convert_##FT(y), convert_##FT(x), a); y = convert_##HT(a); \
    a = fma(convert_##FT(x), convert_##FT(y), a); x = convert_##HT(a); \
    a = fma(convert_##FT(y), convert_##FT(x), a); y = convert_##HT(a);
\n#define MAD_16(HT, FT, x, y, a)  MAD_4(HT,FT,x,y,a); MAD_4(HT,FT,x,y,a); MAD_4(HT,FT,x,y,a); MAD_4(HT,FT,x,y,a);
\n

\n#ifdef HALF_AVAILABLE
\n

__kernel void compute_mp_v1(__global float *ptr, float _B)
{
    half x = (half)_B;
    half y = (half)get_local_id(0);
    float a = _B;

    for(int i=0; i<128; i++)
    {
        MAD_16(half, float, x, y, a);
    }

    ptr[get_global_id(0)] = a + (float)x + (float)y;
}

__kernel void compute_mp_v2(__global float *ptr, float _B)
{
    half2 x = (half2)((half)_B, (half)(_B+1));
    half2 y = (half2)get_local_id(0);
    float2 a = (float2)(_B, _B+1);

    for(int i=0; i<64; i++)
    {
        MAD_16(half2, float2, x, y, a);
    }

    float s = a.S0 + a.S1;
    ptr[get_global_id(0)] = s + (float)(x.S0) + (float)(y.S0);
}

__kernel void compute_mp_v4(__global float *ptr, float _B)
{
    half4 x = (half4)((half)_B, (half)(_B+1), (half)(_B+2), (half)(_B+3));
    half4 y = (half4)get_local_id(0);
    float4 a = (float4)(_B, _B+1, _B+2, _B+3);

    for(int i=0; i<32; i++)
    {
        MAD_16(half4, float4, x, y, a);
    }

    float s = a.S0 + a.S1 + a.S2 + a.S3;
    ptr[get_global_id(0)] = s + (float)(x.S0) + (float)(y.S0);
}

__kernel void compute_mp_v8(__global float *ptr, float _B)
{
    half8 x = (half8)((half)_B, (half)(_B+1), (half)(_B+2), (half)(_B+3),
                      (half)(_B+4), (half)(_B+5), (half)(_B+6), (half)(_B+7));
    half8 y = (half8)get_local_id(0);
    float8 a = (float8)(_B, _B+1, _B+2, _B+3, _B+4, _B+5, _B+6, _B+7);

    for(int i=0; i<16; i++)
    {
        MAD_16(half8, float8, x, y, a);
    }

    float s = a.S0 + a.S1 + a.S2 + a.S3 + a.S4 + a.S5 + a.S6 + a.S7;
    ptr[get_global_id(0)] = s + (float)(x.S0) + (float)(y.S0);
}

__kernel void compute_mp_v16(__global float *ptr, float _B)
{
    half16 x = (half16)((half)_B,     (half)(_B+1),  (half)(_B+2),  (half)(_B+3),
                        (half)(_B+4), (half)(_B+5),  (half)(_B+6),  (half)(_B+7),
                        (half)(_B+8), (half)(_B+9),  (half)(_B+10), (half)(_B+11),
                        (half)(_B+12),(half)(_B+13), (half)(_B+14), (half)(_B+15));
    half16 y = (half16)get_local_id(0);
    float16 a = (float16)(_B,    _B+1,  _B+2,  _B+3,  _B+4,  _B+5,  _B+6,  _B+7,
                          _B+8,  _B+9,  _B+10, _B+11, _B+12, _B+13, _B+14, _B+15);

    for(int i=0; i<8; i++)
    {
        MAD_16(half16, float16, x, y, a);
    }

    float8 t = a.lo + a.hi;
    float4 t2 = t.lo + t.hi;
    float2 t3 = t2.lo + t2.hi;
    ptr[get_global_id(0)] = t3.S0 + t3.S1 + (float)(x.S0) + (float)(y.S0);
}

\n#endif      // HALF_AVAILABLE
\n

)
