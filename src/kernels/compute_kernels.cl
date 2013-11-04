MSTRINGIFY(

// Forcing compiler not to auto-vectorize using volatile


__kernel void compute_sp_v1(float _A, float _B)
{
    float A = _A;
    float B = _B;
    
    volatile float x;
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
}


__kernel void compute_sp_v2(float _A, float _B)
{
    float2 A = _A;
    float2 B = _B;
    
    volatile float2 x;
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
}

__kernel void compute_sp_v4(float _A, float _B)
{
    float4 A = _A;
    float4 B = _B;
    
    volatile float4 x;
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
}


__kernel void compute_sp_v8(float _A, float _B)
{
    float8 A = _A;
    float8 B = _B;
    
    volatile float8 x;
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
}

__kernel void compute_sp_v16(float _A, float _B)
{
    float16 A = _A;
    float16 B = _B;
    
    volatile float16 x;
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
    
    x = mad(A, B, A);
    x = mad(A, A, B);
    x = mad(B, A, B);
    x = mad(B, B, A);
}


)

