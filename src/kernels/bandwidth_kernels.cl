MSTRINGIFY(


\n#define FETCH_2(sum, id, A)      sum += A[id];   id += get_local_size(0);   sum += A[id];   id += get_local_size(0);
\n#define FETCH_8(sum, id, A)      FETCH_2(sum, id, A);   FETCH_2(sum, id, A);   FETCH_2(sum, id, A);   FETCH_2(sum, id, A);
\n
\n
\n#define FETCH_PER_WI  16
\n

__kernel void bandwidth_v1(__global float *A, __global float *B)
{
    int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
    float sum = 0;
    
    FETCH_8(sum, id, A);
    FETCH_8(sum, id, A);
    
    B[get_global_id(0)] = sum;
}


__kernel void bandwidth_v2(__global float2 *A, __global float *B)
{   
    int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
    float2 sum = 0;
    
    FETCH_8(sum, id, A);
    FETCH_8(sum, id, A);
    
    B[get_global_id(0)] = (sum.S0) + (sum.S1);
}


__kernel void bandwidth_v4(__global float4 *A, __global float *B)
{    
    int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
    float4 sum = 0;
    
    FETCH_8(sum, id, A);
    FETCH_8(sum, id, A);
    
    B[get_global_id(0)] = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3);
}


__kernel void bandwidth_v8(__global float8 *A, __global float *B)
{
    int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
    float8 sum = 0;
    
    FETCH_8(sum, id, A);
    FETCH_8(sum, id, A);
    
    B[get_global_id(0)] = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3) + (sum.S4) + (sum.S5) + (sum.S6) + (sum.S7);
}

__kernel void bandwidth_v16(__global float16 *A, __global float *B)
{
    int id = (get_group_id(0) * get_local_size(0) * FETCH_PER_WI) + get_local_id(0);
    float16 sum = 0;
    
    FETCH_8(sum, id, A);
    FETCH_8(sum, id, A);
    
    float t = (sum.S0) + (sum.S1) + (sum.S2) + (sum.S3) + (sum.S4) + (sum.S5) + (sum.S6) + (sum.S7);
    t += (sum.S8) + (sum.S9) + (sum.SA) + (sum.SB) + (sum.SC) + (sum.SD) + (sum.SE) + (sum.SF);
    B[get_global_id(0)] = t;
}

)

