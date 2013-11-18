MSTRINGIFY(

// Forcing compiler not to auto-vectorize using volatile


__kernel void bandwidth_v1(__global float *A, __global float *B)
{
    volatile float x = A[get_global_id(0)];
    volatile float y = B[get_global_id(0)];
}


__kernel void bandwidth_v2(__global float2 *A, __global float2 *B)
{   
    volatile float2 x = A[get_global_id(0)];
    volatile float2 y = B[get_global_id(0)];
}


__kernel void bandwidth_v4(__global float4 *A, __global float4 *B)
{    
    volatile float4 x = A[get_global_id(0)];
    volatile float4 y = B[get_global_id(0)];
}


__kernel void bandwidth_v8(__global float8 *A, __global float8 *B)
{
    volatile float8 x = A[get_global_id(0)];
    volatile float8 y = B[get_global_id(0)];
}

__kernel void bandwidth_v16(__global float16 *A, __global float16 *B)
{
    volatile float16 x = A[get_global_id(0)];
    volatile float16 y = B[get_global_id(0)];
}

)

