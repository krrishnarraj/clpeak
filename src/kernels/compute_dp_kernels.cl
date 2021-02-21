MSTRINGIFY(

// Stringifying requires a new line after hash defines

\n#if defined(cl_khr_fp64)
\n  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
\n  #define DOUBLE_AVAILABLE
\n#elif defined(cl_amd_fp64)
\n  #pragma OPENCL EXTENSION cl_amd_fp64 : enable
\n  #define DOUBLE_AVAILABLE
\n#endif

\n#undef MAD_4
\n#undef MAD_16
\n#undef MAD_64
\n
\n#define MAD_4(x, y, z)     z += mad(y, x, y);  z += mad(x, y, x);  z += mad(y, x, y);  z += mad(x, y, x);
\n#define MAD_16(x, y, z)    MAD_4(x, y, z);     MAD_4(x, y, z);     MAD_4(x, y, z);     MAD_4(x, y, z);
\n#define MAD_64(x, y, z)    MAD_16(x, y, z);    MAD_16(x, y, z);    MAD_16(x, y, z);    MAD_16(x, y, z);
\n

\n
\n#ifdef DOUBLE_AVAILABLE
\n


__kernel void compute_dp_v1(__global double *ptr, double _A)
{
    double x = _A;
    double y = (double)get_local_id(0);
    double z = 0;

    for(int i=0; i<128; i++)
    {
        MAD_16(x, y, z);
    }

    ptr[get_global_id(0)] = z;
}


__kernel void compute_dp_v2(__global double *ptr, double _A)
{
    double2 x = (double2)(_A, (_A+1));
    double2 y = (double2)get_local_id(0);
    double2 z = 0;

    for(int i=0; i<64; i++)
    {
        MAD_16(x, y, z);
    }

    ptr[get_global_id(0)] = (z.S0) + (z.S1);
}

__kernel void compute_dp_v4(__global double *ptr, double _A)
{
    double4 x = (double4)(_A, (_A+1), (_A+2), (_A+3));
    double4 y = (double4)get_local_id(0);
    double4 z = 0;

    for(int i=0; i<32; i++)
    {
        MAD_16(x, y, z);
    }

    ptr[get_global_id(0)] = (z.S0) + (z.S1) + (z.S2) + (z.S3);
}


__kernel void compute_dp_v8(__global double *ptr, double _A)
{
    double8 x = (double8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    double8 y = (double8)get_local_id(0);
    double8 z = 0;

    for(int i=0; i<16; i++)
    {
        MAD_16(x, y, z);
    }

    ptr[get_global_id(0)] = (z.S0) + (z.S1) + (z.S2) + (z.S3) + (z.S4) + (z.S5) + (z.S6) + (z.S7);
}

__kernel void compute_dp_v16(__global double *ptr, double _A)
{
    double16 x = (double16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    double16 y = (double16)get_local_id(0);
    double16 z = 0;

    for(int i=0; i<8; i++)
    {
        MAD_16(x, y, z);
    }

    double2 t = (z.S01) + (z.S23) + (z.S45) + (z.S67) + (z.S89) + (z.SAB) + (z.SCD) + (z.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}

\n
\n#endif      // DOUBLE_AVAILABLE
\n

)
