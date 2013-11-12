MSTRINGIFY(

// Stringifying requires a new line after hash defines 

\n#if defined(cl_khr_fp64)
\n  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
\n  #define DOUBLE_AVAILABLE
\n#elif defined(cl_amd_fp64)
\n  #pragma OPENCL EXTENSION cl_amd_fp64 : enable
\n  #define DOUBLE_AVAILABLE
\n#endif
\n
\n#ifdef DOUBLE_AVAILABLE
\n


__kernel void compute_dp_v1(__global double *ptr, double _A, double _B)
{
    double A = _A;
    double B = _B;
    double x;
    double y;
    
    x = mad(A, B, (A + (double)get_global_id(0)));
    y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);

    ptr[get_global_id(0)] = y;
}


__kernel void compute_dp_v2(__global double2 *ptr, double _A, double _B)
{
    double2 A = _A;
    double2 B = _B;
    double2 x;
    double2 y;
    
    x = mad(A, B, (A + (double2)get_global_id(0)));
    y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);

    ptr[get_global_id(0)] = y;
}

__kernel void compute_dp_v4(__global double4 *ptr, double _A, double _B)
{
    double4 A = _A;
    double4 B = _B;
    double4 x;
    double4 y;
    
    x = mad(A, B, (A + (double4)get_global_id(0)));
    y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);

    ptr[get_global_id(0)] = y;
}


__kernel void compute_dp_v8(__global double8 *ptr, double _A, double _B)
{
    double8 A = _A;
    double8 B = _B;
    double8 x;
    double8 y;
    
    x = mad(A, B, (A + (double8)get_global_id(0)));
    y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);

    ptr[get_global_id(0)] = y;
}

__kernel void compute_dp_v16(__global double16 *ptr, double _A, double _B)
{
    double16 A = _A;
    double16 B = _B;
    double16 x;
    double16 y;
    
    x = mad(A, B, (A + (double16)get_global_id(0)));
    y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);

    ptr[get_global_id(0)] = y;
}

\n
\n#endif      // DOUBLE_AVAILABLE
\n

)

