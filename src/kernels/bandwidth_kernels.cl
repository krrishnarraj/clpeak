MSTRINGIFY(

// Forcing compiler not to auto-vectorize using volatile


__kernel void bandwidth_v1(__global float *arr, __global float *out)
{
    volatile float x = arr[get_global_id(0)];
    out[get_global_id(0)] = x + 1.0f;
}


__kernel void bandwidth_v2(__global float2 *arr, __global float2 *out)
{   
    volatile float2 x = arr[get_global_id(0)];
    out[get_global_id(0)] = x + 1.0f;
}


__kernel void bandwidth_v4(__global float4 *arr, __global float4 *out)
{    
    volatile float4 x = arr[get_global_id(0)];
    out[get_global_id(0)] = x + 1.0f;
}


__kernel void bandwidth_v8(__global float8 *arr, __global float8 *out)
{
    volatile float8 x = arr[get_global_id(0)];
    out[get_global_id(0)] = x + 1.0f;
}

__kernel void bandwidth_v16(__global float16 *arr, __global float16 *out)
{
    volatile float16 x = arr[get_global_id(0)];
    out[get_global_id(0)] = x + 1.0f;
}

)

