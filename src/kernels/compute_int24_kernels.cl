MSTRINGIFY(

// Avoiding auto-vectorize by using vector-width locked dependent code

\n#undef MAD_4INT
\n#undef MAD_16INT
\n#undef MAD_64INT
\n
\n#define MAD_4INT(x, y)  x = mad24(y,x,y);   y = mad24(x,y,x);   x = mad24(y,x,y);   y = mad24(x,y,x);
\n#define MAD_16INT(x, y) MAD_4INT(x, y);     MAD_4INT(x, y);     MAD_4INT(x, y);     MAD_4INT(x, y);
\n#define MAD_64INT(x, y) MAD_16INT(x, y);    MAD_16INT(x, y);    MAD_16INT(x, y);    MAD_16INT(x, y);
\n

__kernel void compute_intfast_v1(__global int *ptr, int _A)
{
    int x = _A;
    int y = (int)get_local_id(0);

    for(int i=0; i<64; i++)
    {
        MAD_16INT(x, y);
    }

    ptr[get_global_id(0)] = y;
}


__kernel void compute_intfast_v2(__global int *ptr, int _A)
{
    int2 x = (int2)(_A, (_A+1));
    int2 y = (int2)get_local_id(0);

    for(int i=0; i<32; i++)
    {
        MAD_16INT(x, y);
    }

    ptr[get_global_id(0)] = (y.S0) + (y.S1);
}

__kernel void compute_intfast_v4(__global int *ptr, int _A)
{
    int4 x = (int4)(_A, (_A+1), (_A+2), (_A+3));
    int4 y = (int4)get_local_id(0);

    for(int i=0; i<16; i++)
    {
        MAD_16INT(x, y);
    }

    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3);
}


__kernel void compute_intfast_v8(__global int *ptr, int _A)
{
    int8 x = (int8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    int8 y = (int8)get_local_id(0);

    for(int i=0; i<8; i++)
    {
        MAD_16INT(x, y);
    }

    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3) + (y.S4) + (y.S5) + (y.S6) + (y.S7);
}

__kernel void compute_intfast_v16(__global int *ptr, int _A)
{
    int16 x = (int16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    int16 y = (int16)get_local_id(0);

    for(int i=0; i<4; i++)
    {
        MAD_16INT(x, y);
    }

    int2 t = (y.S01) + (y.S23) + (y.S45) + (y.S67) + (y.S89) + (y.SAB) + (y.SCD) + (y.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}


)
