MSTRINGIFY(

// Avoiding auto-vectorize by using vector-width locked dependent code.
// Chain shape and why: see the MAD chain block in include/common/common.h.

\n#undef MAD_4
\n#undef MAD_16
\n#undef MAD_64
\n
\n#define MAD_4(x, c)     x = (x*x) + c;      x = (x*x) + c;      x = (x*x) + c;      x = (x*x) + c;
\n#define MAD_16(x, c)    MAD_4(x, c);        MAD_4(x, c);        MAD_4(x, c);        MAD_4(x, c);
\n#define MAD_64(x, c)    MAD_16(x, c);       MAD_16(x, c);       MAD_16(x, c);       MAD_16(x, c);
\n

__kernel void compute_short_v1(__global short *ptr, short _A)
{
    short x = _A;
    short c = (short)get_local_id(0);

    for(int i=0; i<64; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = x;
}


__kernel void compute_short_v2(__global short *ptr, short _A)
{
    short2 x = (short2)(_A, (_A+1));
    short2 c = (short2)get_local_id(0);

    for(int i=0; i<32; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1);
}

__kernel void compute_short_v4(__global short *ptr, short _A)
{
    short4 x = (short4)(_A, (_A+1), (_A+2), (_A+3));
    short4 c = (short4)get_local_id(0);

    for(int i=0; i<16; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3);
}


__kernel void compute_short_v8(__global short *ptr, short _A)
{
    short8 x = (short8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    short8 c = (short8)get_local_id(0);

    for(int i=0; i<8; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3) + (x.S4) + (x.S5) + (x.S6) + (x.S7);
}

__kernel void compute_short_v16(__global short *ptr, short _A)
{
    short16 x = (short16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    short16 c = (short16)get_local_id(0);

    for(int i=0; i<4; i++)
    {
        MAD_16(x, c);
    }

    short2 t = (x.S01) + (x.S23) + (x.S45) + (x.S67) + (x.S89) + (x.SAB) + (x.SCD) + (x.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}

// ---- affine-chain variants (generated; see mad_chain.cl) ----

__kernel void compute_short_alt_v1(__global short *ptr, short _A)
{
    RT4_DECL(short, _A, (short)get_local_id(0))

    for (int i = 0; i < 64; i++)
    {
        RT4_16
    }

    short r = RT4_RES;
    ptr[get_global_id(0)] = r;
}

__kernel void compute_short_alt_v2(__global short *ptr, short _A)
{
    RT2_DECL(short2, (short2)(_A, (_A + 1)), (short2)get_local_id(0))

    for (int i = 0; i < 32; i++)
    {
        RT2_16
    }

    short2 r = RT2_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1;
}

__kernel void compute_short_alt_v4(__global short *ptr, short _A)
{
    RT2_DECL(short4, (short4)(_A, (_A + 1), (_A + 2), (_A + 3)), (short4)get_local_id(0))

    for (int i = 0; i < 16; i++)
    {
        RT2_16
    }

    short4 r = RT2_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1 + r.S2 + r.S3;
}

__kernel void compute_short_alt_v8(__global short *ptr, short _A)
{
    RT2_DECL(short8, (short8)(_A, (_A + 1), (_A + 2), (_A + 3), (_A + 4), (_A + 5), (_A + 6), (_A + 7)), (short8)get_local_id(0))

    for (int i = 0; i < 8; i++)
    {
        RT2_16
    }

    short8 r = RT2_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1 + r.S2 + r.S3 + r.S4 + r.S5 + r.S6 + r.S7;
}

__kernel void compute_short_alt_v16(__global short *ptr, short _A)
{
    RT2_DECL(short16, (short16)(_A, (_A + 1), (_A + 2), (_A + 3), (_A + 4), (_A + 5), (_A + 6), (_A + 7), (_A + 8), (_A + 9), (_A + 10), (_A + 11), (_A + 12), (_A + 13), (_A + 14), (_A + 15)), (short16)get_local_id(0))

    for (int i = 0; i < 4; i++)
    {
        RT2_16
    }

    short16 r = RT2_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1 + r.S2 + r.S3 + r.S4 + r.S5 + r.S6 + r.S7 + r.S8 + r.S9 + r.SA + r.SB + r.SC + r.SD + r.SE + r.SF;
}

)
