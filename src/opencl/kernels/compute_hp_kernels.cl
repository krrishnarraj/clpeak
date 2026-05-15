MSTRINGIFY(

// Stringifying requires a new line after hash defines

\n#if defined(cl_khr_fp16)
\n  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
\n  #define HALF_AVAILABLE
\n#endif

\n#undef MAD_4
\n#undef MAD_16
\n#undef MAD_64
\n
\n#define MAD_4(x, y)     x = mad(y, x, y);   y = mad(x, y, x);   x = mad(y, x, y);   y = mad(x, y, x);
\n#define MAD_16(x, y)    MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);
\n#define MAD_64(x, y)    MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);
\n

\n
\n#ifdef HALF_AVAILABLE
\n


__kernel void compute_hp_v1(__global half *ptr, float _B)
{
    half _A = (half)_B;
    half x = _A;
    half y = (half)get_local_id(0);

    for(int i=0; i<128; i++)
    {
        MAD_16(x, y);
    }

    ptr[get_global_id(0)] = y;
}


__kernel void compute_hp_v2(__global half *ptr, float _B)
{
    half _A = (half)_B;
    half2 x = (half2)(_A, (_A+1));
    half2 y = (half2)get_local_id(0);

    for(int i=0; i<64; i++)
    {
        MAD_16(x, y);
    }

    ptr[get_global_id(0)] = (y.S0) + (y.S1);
}

__kernel void compute_hp_v4(__global half *ptr, float _B)
{
    half _A = (half)_B;
    half4 x = (half4)(_A, (_A+1), (_A+2), (_A+3));
    half4 y = (half4)get_local_id(0);

    for(int i=0; i<32; i++)
    {
        MAD_16(x, y);
    }

    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3);
}


__kernel void compute_hp_v8(__global half *ptr, float _B)
{
    half _A = (half)_B;
    half8 x = (half8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    half8 y = (half8)get_local_id(0);

    for(int i=0; i<16; i++)
    {
        MAD_16(x, y);
    }

    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3) + (y.S4) + (y.S5) + (y.S6) + (y.S7);
}

__kernel void compute_hp_v16(__global half *ptr, float _B)
{
    half _A = (half)_B;
    half16 x = (half16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    half16 y = (half16)get_local_id(0);

    for(int i=0; i<8; i++)
    {
        MAD_16(x, y);
    }

    half2 t = (y.S01) + (y.S23) + (y.S45) + (y.S67) + (y.S89) + (y.SAB) + (y.SCD) + (y.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}

\n
\n#endif      // half_AVAILABLE
\n

)
