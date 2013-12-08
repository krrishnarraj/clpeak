MSTRINGIFY(

// Forcing compiler not to auto-vectorize using volatile

\n#define FETCH_2(sum, id, A)      sum += A[id];   id += get_global_size(0);   sum += A[id];   id += get_global_size(0);
\n#define FETCH_8(sum, id, A)      FETCH_2(sum, id, A);   FETCH_2(sum, id, A);   FETCH_2(sum, id, A);   FETCH_2(sum, id, A);
\n

__kernel void bandwidth_v1(__global volatile float *A, __global volatile float *B)
{
    int id = get_global_id(0);
    volatile float sum = 0;
    
    FETCH_8(sum, id, A);
    FETCH_8(sum, id, A);
    
    B[get_global_id(0)] = sum;
}


__kernel void bandwidth_v2(__global volatile float2 *A, __global volatile float *B)
{   
    int id = get_global_id(0);
    volatile float2 sum = 0;
    
    FETCH_8(sum, id, A);
    FETCH_8(sum, id, A);
    
    B[get_global_id(0)] = (sum.S0) + (sum.S1);
}


__kernel void bandwidth_v4(__global volatile float4 *A, __global volatile float *B)
{    
    int id = get_global_id(0);
    volatile float4 sum = 0;
    
    FETCH_8(sum, id, A);
    FETCH_8(sum, id, A);
    
    B[get_global_id(0)] = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3);
}


__kernel void bandwidth_v8(__global volatile float8 *A, __global volatile float *B)
{
    int id = get_global_id(0);
    volatile float8 sum = 0;
    
    FETCH_8(sum, id, A);
    FETCH_8(sum, id, A);
    
    B[get_global_id(0)] = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3) + (sum.S4) + (sum.S5) + (sum.S6) + (sum.S7);
}

__kernel void bandwidth_v16(__global volatile float16 *A, __global volatile float *B)
{
    int id = get_global_id(0);
    volatile float16 sum = 0;
    
    FETCH_8(sum, id, A);
    FETCH_8(sum, id, A);
    
    float t = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3) + (sum.S4) + (sum.S5) + (sum.S6) + (sum.S7);
    t += (sum.S8) + (sum.S9) + (sum.SA) + (sum.SB) + (sum.SC) + (sum.SD) + (sum.SE) + (sum.SF);
    B[get_global_id(0)] = t;
}

)

