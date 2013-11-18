MSTRINGIFY(

// Avoiding auto-vectorize by using vector-width locked dependent code

\n#define MAD_4(A, B, x, y)     x = mad(y, B, A);   y = mad(A, x, B);   x = mad(y, A, B);   y = mad(x, B, A);
\n#define MAD_16(A, B, x, y)    MAD_4(A, B, x, y);  MAD_4(A, B, x, y);  MAD_4(A, B, x, y);  MAD_4(A, B, x, y);
\n#define MAD_64(A, B, x, y)    MAD_16(A, B, x, y); MAD_16(A, B, x, y); MAD_16(A, B, x, y); MAD_16(A, B, x, y);
\n

__kernel void compute_sp_v1(__global float *ptr, float _A, float _B)
{
    float A = _A, B = _B;
    float x, y = (float)get_local_id(0);
    
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


__kernel void compute_sp_v2(__global float *ptr, float _A, float _B)
{
    float2 A = (float2)(_A, (_A+1));
    float2 B = (float2)(_B, (_B+1));
    float2 x, y = (float2)get_local_id(0);
    
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);

    ptr[get_global_id(0)] = (y.S0) + (y.S1);
}

__kernel void compute_sp_v4(__global float *ptr, float _A, float _B)
{
    float4 A = (float4)(_A, (_A+1), (_A+2), (_A+3));
    float4 B = (float4)(_B, (_B+1), (_B+2), (_B+3));
    float4 x, y = (float4)get_local_id(0);
    
    MAD_64(A, B, x, y);
    MAD_64(A, B, x, y);
    
    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3);
}


__kernel void compute_sp_v8(__global float *ptr, float _A, float _B)
{
    float8 A = (float8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    float8 B = (float8)(_B, (_B+1), (_B+2), (_B+3), (_B+4), (_B+5), (_B+6), (_B+7));
    float8 x, y = (float8)get_local_id(0);
    
    MAD_64(A, B, x, y);

    ptr[get_global_id(0)] = (y.S0) + (y.S1) + (y.S2) + (y.S3) + (y.S4) + (y.S5) + (y.S6) + (y.S7);
}

__kernel void compute_sp_v16(__global float *ptr, float _A, float _B)
{
    float16 A = (float16)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7),
                    (_A+8), (_A+9), (_A+10), (_A+11), (_A+12), (_A+13), (_A+14), (_A+15));
    float16 B = (float16)(_B, (_B+1), (_B+2), (_B+3), (_B+4), (_B+5), (_B+6), (_B+7),
                    (_B+8), (_B+9), (_B+10), (_B+11), (_B+12), (_B+13), (_B+14), (_B+15));
    float16 x, y = (float16)get_local_id(0);

    MAD_16(A, B, x, y);
    MAD_16(A, B, x, y);

    float2 t = (y.S01) + (y.S23) + (y.S45) + (y.S67) + (y.S89) + (y.SAB) + (y.SCD) + (y.SEF);
    ptr[get_global_id(0)] = t.S0 + t.S1;
}


)

