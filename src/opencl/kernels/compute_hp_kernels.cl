MSTRINGIFY(

// Stringifying requires a new line after hash defines

\n#if defined(cl_khr_fp16)
\n  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
\n  #define HALF_AVAILABLE
\n#endif

// Plain contracted expression instead of mad() -- see compute_sp_kernels.cl.
// Chain shape and why: see the MAD chain block in include/common/common.h.

\n#undef MAD_4
\n#undef MAD_16
\n#undef MAD_64
\n
\n#define MAD_4(x, c)     x = (x*x) + c;      x = (x*x) + c;      x = (x*x) + c;      x = (x*x) + c;
\n#define MAD_16(x, c)    MAD_4(x, c);        MAD_4(x, c);        MAD_4(x, c);        MAD_4(x, c);
\n#define MAD_64(x, c)    MAD_16(x, c);       MAD_16(x, c);       MAD_16(x, c);       MAD_16(x, c);
\n

\n
\n#ifdef HALF_AVAILABLE
\n


__kernel void compute_hp_v1(__global half *ptr, float _B)
{
    half _A = (half)_B;
    half x = _A;
    half c = (half)get_local_id(0);

    for(int i=0; i<128; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = x;
}


__kernel void compute_hp_v2(__global half *ptr, float _B)
{
    half _A = (half)_B;
    half2 x = (half2)(_A, (_A+1));
    half2 c = (half2)get_local_id(0);

    for(int i=0; i<64; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1);
}

__kernel void compute_hp_v4(__global half *ptr, float _B)
{
    half _A = (half)_B;
    half4 x = (half4)(_A, (_A+1), (_A+2), (_A+3));
    half4 c = (half4)get_local_id(0);

    for(int i=0; i<32; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3);
}


__kernel void compute_hp_v8(__global half *ptr, float _B)
{
    half _A = (half)_B;
    half8 x = (half8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    half8 c = (half8)get_local_id(0);

    for(int i=0; i<16; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3) + (x.S4) + (x.S5) + (x.S6) + (x.S7);
}

__kernel void compute_hp_v16(__global half *ptr, float _B)
{
    half _A = (half)_B;
    half16 x = (half16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    half16 c = (half16)get_local_id(0);

    for(int i=0; i<8; i++)
    {
        MAD_16(x, c);
    }

    half2 t = (x.S01) + (x.S23) + (x.S45) + (x.S67) + (x.S89) + (x.SAB) + (x.SCD) + (x.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}

\n
\n#endif      // half_AVAILABLE
\n

)
