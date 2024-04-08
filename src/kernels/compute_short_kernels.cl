MSTRINGIFY(

// Avoiding auto-vectorize by using vector-width locked dependent code

\n#undef MAD_4
\n#undef MAD_16
\n#undef MAD_64
\n
\n#define MAD_4(x, y)     x = (y*x) + y;      y = (x*y) + x;      x = (y*x) + y;      y = (x*y) + x;
\n#define MAD_16(x, y)    MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);        MAD_4(x, y);
\n#define MAD_64(x, y)    MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);       MAD_16(x, y);
\n

__kernel void compute_short_v1(__global short *ptr, short _A)
{
    short x = _A;
    short y = (short)get_local_id(0);

    for(int i=0; i<64; i++)
    {
        MAD_16(x, y);
    }

    ptr[get_global_id(0)] = y;
}


__kernel void compute_short_v2(__global short *ptr, short _A)
{
    short2 x = (short2)(_A, (_A+1));
    short2 y = (short2)get_local_id(0);

    for(int i=0; i<32; i++)
    {
        MAD_16(x, y);
    }

    ptr[get_global_id(0)] = (y.S0) + (y.S1);
}

__kernel void compute_short_v4(__global short *ptr, short _A)
{
    short4 x = (short4)(_A, (_A+1), (_A+2), (_A+3));
    short4 y = (short4)get_local_id(0);

    for(int i=0; i<16; i++)
    {
        MAD_16(x, y);
    }

    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3);
}


__kernel void compute_short_v8(__global short *ptr, short _A)
{
    short8 x = (short8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    short8 y = (short8)get_local_id(0);

    for(int i=0; i<8; i++)
    {
        MAD_16(x, y);
    }

    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3) + (y.S4) + (y.S5) + (y.S6) + (y.S7);
}

__kernel void compute_short_v16(__global short *ptr, short _A)
{
    short16 x = (short16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    short16 y = (short16)get_local_id(0);

    for(int i=0; i<4; i++)
    {
        MAD_16(x, y);
    }

    short2 t = (y.S01) + (y.S23) + (y.S45) + (y.S67) + (y.S89) + (y.SAB) + (y.SCD) + (y.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}


)
