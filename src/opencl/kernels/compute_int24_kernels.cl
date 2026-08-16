MSTRINGIFY(

// Avoiding auto-vectorize by using vector-width locked dependent code.
// Chain shape and why: see the MAD chain block in include/common/common.h.
// mad24 is documented only for operands that fit in 24 bits; this chain (like
// the ping-pong it replaced) feeds its own result back in and leaves that
// range within a few steps, so the low bits are what is really being timed.

\n#undef MAD_4INT
\n#undef MAD_16INT
\n#undef MAD_64INT
\n
\n#define MAD_4INT(x, c)  x = mad24(x,x,c);   x = mad24(x,x,c);   x = mad24(x,x,c);   x = mad24(x,x,c);
\n#define MAD_16INT(x, c) MAD_4INT(x, c);     MAD_4INT(x, c);     MAD_4INT(x, c);     MAD_4INT(x, c);
\n#define MAD_64INT(x, c) MAD_16INT(x, c);    MAD_16INT(x, c);    MAD_16INT(x, c);    MAD_16INT(x, c);
\n

__kernel void compute_intfast_v1(__global int *ptr, int _A)
{
    int x = _A;
    int c = (int)get_local_id(0);

    for(int i=0; i<64; i++)
    {
        MAD_16INT(x, c);
    }

    ptr[get_global_id(0)] = x;
}


__kernel void compute_intfast_v2(__global int *ptr, int _A)
{
    int2 x = (int2)(_A, (_A+1));
    int2 c = (int2)get_local_id(0);

    for(int i=0; i<32; i++)
    {
        MAD_16INT(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1);
}

__kernel void compute_intfast_v4(__global int *ptr, int _A)
{
    int4 x = (int4)(_A, (_A+1), (_A+2), (_A+3));
    int4 c = (int4)get_local_id(0);

    for(int i=0; i<16; i++)
    {
        MAD_16INT(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3);
}


__kernel void compute_intfast_v8(__global int *ptr, int _A)
{
    int8 x = (int8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    int8 c = (int8)get_local_id(0);

    for(int i=0; i<8; i++)
    {
        MAD_16INT(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3) + (x.S4) + (x.S5) + (x.S6) + (x.S7);
}

__kernel void compute_intfast_v16(__global int *ptr, int _A)
{
    int16 x = (int16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    int16 c = (int16)get_local_id(0);

    for(int i=0; i<4; i++)
    {
        MAD_16INT(x, c);
    }

    int2 t = (x.S01) + (x.S23) + (x.S45) + (x.S67) + (x.S89) + (x.SAB) + (x.SCD) + (x.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}


)
