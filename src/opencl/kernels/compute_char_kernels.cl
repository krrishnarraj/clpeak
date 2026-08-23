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

__kernel void compute_char_v1(__global char *ptr, char _A)
{
    char x = _A;
    char c = (char)get_local_id(0);

    for(int i=0; i<64; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = x;
}


__kernel void compute_char_v2(__global char *ptr, char _A)
{
    char2 x = (char2)(_A, (_A+1));
    char2 c = (char2)get_local_id(0);

    for(int i=0; i<32; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1);
}

__kernel void compute_char_v4(__global char *ptr, char _A)
{
    char4 x = (char4)(_A, (_A+1), (_A+2), (_A+3));
    char4 c = (char4)get_local_id(0);

    for(int i=0; i<16; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3);
}


__kernel void compute_char_v8(__global char *ptr, char _A)
{
    char8 x = (char8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    char8 c = (char8)get_local_id(0);

    for(int i=0; i<8; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3) + (x.S4) + (x.S5) + (x.S6) + (x.S7);
}

__kernel void compute_char_v16(__global char *ptr, char _A)
{
    char16 x = (char16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    char16 c = (char16)get_local_id(0);

    for(int i=0; i<4; i++)
    {
        MAD_16(x, c);
    }

    char2 t = (x.S01) + (x.S23) + (x.S45) + (x.S67) + (x.S89) + (x.SAB) + (x.SCD) + (x.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}

// ---- affine-chain variants (generated; see mad_chain.cl) ----

__kernel void compute_char_alt_v1(__global char *ptr, char _A)
{
    RT4_DECL(char, _A, (char)get_local_id(0))

    for (int i = 0; i < 64; i++)
    {
        RT4_16
    }

    char r = RT4_RES;
    ptr[get_global_id(0)] = r;
}

__kernel void compute_char_alt_v2(__global char *ptr, char _A)
{
    RT2_DECL(char2, (char2)(_A, (_A + 1)), (char2)get_local_id(0))

    for (int i = 0; i < 32; i++)
    {
        RT2_16
    }

    char2 r = RT2_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1;
}

__kernel void compute_char_alt_v4(__global char *ptr, char _A)
{
    RT2_DECL(char4, (char4)(_A, (_A + 1), (_A + 2), (_A + 3)), (char4)get_local_id(0))

    for (int i = 0; i < 16; i++)
    {
        RT2_16
    }

    char4 r = RT2_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1 + r.S2 + r.S3;
}

__kernel void compute_char_alt_v8(__global char *ptr, char _A)
{
    RT2_DECL(char8, (char8)(_A, (_A + 1), (_A + 2), (_A + 3), (_A + 4), (_A + 5), (_A + 6), (_A + 7)), (char8)get_local_id(0))

    for (int i = 0; i < 8; i++)
    {
        RT2_16
    }

    char8 r = RT2_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1 + r.S2 + r.S3 + r.S4 + r.S5 + r.S6 + r.S7;
}

__kernel void compute_char_alt_v16(__global char *ptr, char _A)
{
    RT2_DECL(char16, (char16)(_A, (_A + 1), (_A + 2), (_A + 3), (_A + 4), (_A + 5), (_A + 6), (_A + 7), (_A + 8), (_A + 9), (_A + 10), (_A + 11), (_A + 12), (_A + 13), (_A + 14), (_A + 15)), (char16)get_local_id(0))

    for (int i = 0; i < 4; i++)
    {
        RT2_16
    }

    char16 r = RT2_RES;
    ptr[get_global_id(0)] = r.S0 + r.S1 + r.S2 + r.S3 + r.S4 + r.S5 + r.S6 + r.S7 + r.S8 + r.S9 + r.SA + r.SB + r.SC + r.SD + r.SE + r.SF;
}

)
