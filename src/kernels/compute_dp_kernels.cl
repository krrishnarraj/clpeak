MSTRINGIFY(

// Stringifying requires a new line after hash defines 

\n#if defined(cl_khr_fp64)
\n  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
\n  #define DOUBLE_AVAILABLE
\n#elif defined(cl_amd_fp64)
\n  #pragma OPENCL EXTENSION cl_amd_fp64 : enable
\n  #define DOUBLE_AVAILABLE
\n#endif

\n#define MAD_4(A, B, x, y)     x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
\n#define MAD_16(A, B, x, y)    MAD_4(A, B, x, y);  MAD_4(A, B, x, y);  MAD_4(A, B, x, y);  MAD_4(A, B, x, y);
\n#define MAD_64(A, B, x, y)    MAD_16(A, B, x, y); MAD_16(A, B, x, y); MAD_16(A, B, x, y); MAD_16(A, B, x, y);
\n

\n
\n#ifdef DOUBLE_AVAILABLE
\n


__kernel void compute_dp_v1(__global double *ptr, double _A, double _B)
{
    double A = _A, B = _B;
    double x, y = (double)get_local_id(0);
    
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);

    ptr[get_global_id(0)] = y;
}


__kernel void compute_dp_v2(__global double *ptr, double _A, double _B)
{
    double2 A = (_A, (_A+1));
    double2 B = (_B, (_B+1));
    double2 x, y = (double2)get_local_id(0);
    
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);

    ptr[get_global_id(0)] = (y.S0) + (y.S1);
}

__kernel void compute_dp_v4(__global double *ptr, double _A, double _B)
{
    double4 A = (_A, (_A+1), (_A+2), (_A+3));
    double4 B = (_B, (_B+1), (_B+2), (_B+3));
    double4 x, y = (double4)get_local_id(0);
    
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    
    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3);
}


__kernel void compute_dp_v8(__global double *ptr, double _A, double _B)
{
    double8 A = (_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    double8 B = (_B, (_B+1), (_B+2), (_B+3), (_B+4), (_B+5), (_B+6), (_B+7));
    double8 x, y = (double8)get_local_id(0);
    
    MAD_64(A, B, x, y);

    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3) + (y.S4) + (y.S5) + (y.S6) + (y.S7);
}

__kernel void compute_dp_v16(__global double *ptr, double _A, double _B)
{
    double16 A = (_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    double16 B = (_B, (_B+1), (_B+2), (_B+3), (_B+4), (_B+5), (_B+6), (_B+7),
                    (_B+8), (_B+9), (_B+10), (_B+11), (_B+12), (_B+13), (_B+14), (_B+15));
    double16 x, y = (double16)get_local_id(0);

    MAD_16(A, B, x, y);
    MAD_16(A, B, x, y);

    double2 t = (y.S01) + (y.S23) + (y.S45) + (y.S67) + (y.S89) + (y.AB) + (y.CD) + (y.EF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}

\n
\n#endif      // DOUBLE_AVAILABLE
\n

)

