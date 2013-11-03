MSTRINGIFY(

// Forcing compiler not to auto-vectorize using volatile


__kernel void bandwidth_v1(__global float *arr)
{
    volatile float x = arr[get_global_id(0)];
}


__kernel void bandwidth_v2(__global float *arr)
{
    __global float2 *ptr = (__global float2*)arr;
    
    volatile float2 x = ptr[get_global_id(0)];
}


__kernel void bandwidth_v4(__global float *arr)
{
    __global float4 *ptr = (__global float4*)arr;
    
    volatile float4 x = ptr[get_global_id(0)];
}


__kernel void bandwidth_v8(__global float *arr)
{
    __global float8 *ptr = (__global float8*)arr;
    
    volatile float8 x = ptr[get_global_id(0)];
}

__kernel void bandwidth_v16(__global float *arr)
{
    __global float16 *ptr = (__global float16*)arr;
    
    volatile float16 x = ptr[get_global_id(0)];
}

)

