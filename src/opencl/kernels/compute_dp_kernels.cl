MSTRINGIFY(

// Stringifying requires a new line after hash defines

\n#if defined(cl_khr_fp64)
\n  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
\n  #define DOUBLE_AVAILABLE
\n#elif defined(cl_amd_fp64)
\n  #pragma OPENCL EXTENSION cl_amd_fp64 : enable
\n  #define DOUBLE_AVAILABLE
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
\n#ifdef DOUBLE_AVAILABLE
\n


__kernel void compute_dp_v1(__global double *ptr, double _A)
{
    double x = _A;
    double c = (double)get_local_id(0);

    for(int i=0; i<128; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = x;
}


__kernel void compute_dp_v2(__global double *ptr, double _A)
{
    double2 x = (double2)(_A, (_A+1));
    double2 c = (double2)get_local_id(0);

    for(int i=0; i<64; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1);
}

__kernel void compute_dp_v4(__global double *ptr, double _A)
{
    double4 x = (double4)(_A, (_A+1), (_A+2), (_A+3));
    double4 c = (double4)get_local_id(0);

    for(int i=0; i<32; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3);
}


__kernel void compute_dp_v8(__global double *ptr, double _A)
{
    double8 x = (double8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    double8 c = (double8)get_local_id(0);

    for(int i=0; i<16; i++)
    {
        MAD_16(x, c);
    }

    ptr[get_global_id(0)] = (x.S0) + (x.S1) + (x.S2) + (x.S3) + (x.S4) + (x.S5) + (x.S6) + (x.S7);
}

__kernel void compute_dp_v16(__global double *ptr, double _A)
{
    double16 x = (double16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    double16 c = (double16)get_local_id(0);

    for(int i=0; i<8; i++)
    {
        MAD_16(x, c);
    }

    double2 t = (x.S01) + (x.S23) + (x.S45) + (x.S67) + (x.S89) + (x.SAB) + (x.SCD) + (x.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}

\n
\n#endif      // DOUBLE_AVAILABLE
\n

)
