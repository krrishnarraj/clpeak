MSTRINGIFY(

// INT8 dot-product compute throughput using cl_khr_integer_dot_product.
// Each dot_acc_sat(char4, char4, int) computes 4 signed INT8 multiply-adds
// into a 32-bit accumulator and is the hardware DP4a / XDL / INT8-tensor-core
// path on modern GPUs (NVIDIA Turing+, AMD RDNA2+, Intel Xe+, Adreno, Mali).

\n#if defined(cl_khr_integer_dot_product)
\n  #pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
\n  #define INT8_DP_AVAILABLE
\n#endif

\n#ifdef INT8_DP_AVAILABLE

\n#undef MAD_DP_4
\n#undef MAD_DP_16
\n
\n#define MAD_DP_4(x, y, a)    a = dot_acc_sat(x, y, a);  a = dot_acc_sat(x, y, a);  a = dot_acc_sat(x, y, a);  a = dot_acc_sat(x, y, a);
\n#define MAD_DP_16(x, y, a)   MAD_DP_4(x, y, a); MAD_DP_4(x, y, a); MAD_DP_4(x, y, a); MAD_DP_4(x, y, a);
\n

__kernel void compute_int8_dp_v1(__global int *ptr, char _A)
{
    char4 x = (char4)(_A, _A+1, _A+2, _A+3);
    char4 y = (char4)((char)get_local_id(0));
    int a0 = 0;

    for (int i = 0; i < 64; i++)
    {
        MAD_DP_16(x, y, a0);
    }

    ptr[get_global_id(0)] = a0;
}

__kernel void compute_int8_dp_v2(__global int *ptr, char _A)
{
    char4 x = (char4)(_A, _A+1, _A+2, _A+3);
    char4 y = (char4)((char)get_local_id(0));
    int a0 = 0, a1 = 1;

    for (int i = 0; i < 32; i++)
    {
        MAD_DP_16(x, y, a0);
        MAD_DP_16(x, y, a1);
    }

    ptr[get_global_id(0)] = a0 + a1;
}

__kernel void compute_int8_dp_v4(__global int *ptr, char _A)
{
    char4 x = (char4)(_A, _A+1, _A+2, _A+3);
    char4 y = (char4)((char)get_local_id(0));
    int a0 = 0, a1 = 1, a2 = 2, a3 = 3;

    for (int i = 0; i < 16; i++)
    {
        MAD_DP_16(x, y, a0); MAD_DP_16(x, y, a1);
        MAD_DP_16(x, y, a2); MAD_DP_16(x, y, a3);
    }

    ptr[get_global_id(0)] = a0 + a1 + a2 + a3;
}

__kernel void compute_int8_dp_v8(__global int *ptr, char _A)
{
    char4 x = (char4)(_A, _A+1, _A+2, _A+3);
    char4 y = (char4)((char)get_local_id(0));
    int a0=0, a1=1, a2=2, a3=3, a4=4, a5=5, a6=6, a7=7;

    for (int i = 0; i < 8; i++)
    {
        MAD_DP_16(x, y, a0); MAD_DP_16(x, y, a1); MAD_DP_16(x, y, a2); MAD_DP_16(x, y, a3);
        MAD_DP_16(x, y, a4); MAD_DP_16(x, y, a5); MAD_DP_16(x, y, a6); MAD_DP_16(x, y, a7);
    }

    ptr[get_global_id(0)] = a0+a1+a2+a3+a4+a5+a6+a7;
}

__kernel void compute_int8_dp_v16(__global int *ptr, char _A)
{
    char4 x = (char4)(_A, _A+1, _A+2, _A+3);
    char4 y = (char4)((char)get_local_id(0));
    int a0=0, a1=1, a2=2, a3=3, a4=4, a5=5, a6=6, a7=7;
    int b0=0, b1=1, b2=2, b3=3, b4=4, b5=5, b6=6, b7=7;

    for (int i = 0; i < 4; i++)
    {
        MAD_DP_16(x, y, a0); MAD_DP_16(x, y, a1); MAD_DP_16(x, y, a2); MAD_DP_16(x, y, a3);
        MAD_DP_16(x, y, a4); MAD_DP_16(x, y, a5); MAD_DP_16(x, y, a6); MAD_DP_16(x, y, a7);
        MAD_DP_16(x, y, b0); MAD_DP_16(x, y, b1); MAD_DP_16(x, y, b2); MAD_DP_16(x, y, b3);
        MAD_DP_16(x, y, b4); MAD_DP_16(x, y, b5); MAD_DP_16(x, y, b6); MAD_DP_16(x, y, b7);
    }

    ptr[get_global_id(0)] = a0+a1+a2+a3+a4+a5+a6+a7+b0+b1+b2+b3+b4+b5+b6+b7;
}

\n#endif  // INT8_DP_AVAILABLE

)
