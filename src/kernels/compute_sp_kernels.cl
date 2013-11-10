MSTRINGIFY(


__kernel void compute_sp_v1(__global float *ptr, float _A, float _B)
{
    float A = _A;
    float B = _B;
    float x;
    float y;
    
    x = mad(A, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
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


__kernel void compute_sp_v2(__global float2 *ptr, float _A, float _B)
{
    float2 A = _A;
    float2 B = _B;
    float2 x;
    float2 y;
    
    x = mad(A, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
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

__kernel void compute_sp_v4(__global float4 *ptr, float _A, float _B)
{
    float4 A = _A;
    float4 B = _B;
    float4 x;
    float4 y;
    
    x = mad(A, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
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


__kernel void compute_sp_v8(__global float8 *ptr, float _A, float _B)
{
    float8 A = _A;
    float8 B = _B;
    float8 x;
    float8 y;
    
    x = mad(A, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);

    ptr[get_global_id(0)] = y;
}

__kernel void compute_sp_v16(__global float16 *ptr, float _A, float _B)
{
    float16 A = _A;
    float16 B = _B;
    float16 x;
    float16 y;
    
    x = mad(A, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
    x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);

    ptr[get_global_id(0)] = y;
}


)

