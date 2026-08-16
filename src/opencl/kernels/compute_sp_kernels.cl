MSTRINGIFY(

// Avoiding auto-vectorize by using vector-width locked dependent code

// Chain shape (x = x*x + c, c loop-invariant, store x never c) and the reasons
// behind it: see the MAD chain block in include/common/common.h.

// The MAD is written as a plain contracted expression rather than the mad()
// builtin.  Both request the same thing under the -cl-mad-enable we always
// build with (mad() carries no accuracy guarantee either -- the spec permits
// it to be a multiply followed by an add), but some frontends treat the
// builtin as a slow precise path and never lower it to the hardware FMA:
// Apple's deprecated CL-on-Metal compiler halves peak fp32 that way.  This
// also matches the integer/char/short kernels here and the explicit fma
// intrinsics every other clpeak backend uses.  Do not "restore" mad().

\n#undef MAD_4
\n#undef MAD_16
\n#undef MAD_64
\n
\n#define MAD_4(x, c)     x = (x*x) + c;      x = (x*x) + c;      x = (x*x) + c;      x = (x*x) + c;
\n#define MAD_16(x, c)    MAD_4(x, c);        MAD_4(x, c);        MAD_4(x, c);        MAD_4(x, c);
\n#define MAD_64(x, c)    MAD_16(x, c);       MAD_16(x, c);       MAD_16(x, c);       MAD_16(x, c);
\n

__kernel void compute_sp_v1(__global float *ptr, float _A)
{
    float x = _A;
    float c = (float)get_local_id(0);

    for(int i=0; i<128; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = x;
}


__kernel void compute_sp_v2(__global float *ptr, float _A)
{
    float2 x = (float2)(_A, (_A+1));
    float2 c = (float2)get_local_id(0);

    for(int i=0; i<64; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1);
}

__kernel void compute_sp_v4(__global float *ptr, float _A)
{
    float4 x = (float4)(_A, (_A+1), (_A+2), (_A+3));
    float4 c = (float4)get_local_id(0);

    for(int i=0; i<32; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3);
}


__kernel void compute_sp_v8(__global float *ptr, float _A)
{
    float8 x = (float8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    float8 c = (float8)get_local_id(0);

    for(int i=0; i<16; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3) + (x.S4) + (x.S5) + (x.S6) + (x.S7);
}

__kernel void compute_sp_v16(__global float *ptr, float _A)
{
    float16 x = (float16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    float16 c = (float16)get_local_id(0);

    for(int i=0; i<8; i++)
    {
        MAD_16(x, c);
    }

    float2 t = (x.S01) + (x.S23) + (x.S45) + (x.S67) + (x.S89) + (x.SAB) + (x.SCD) + (x.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}


)
