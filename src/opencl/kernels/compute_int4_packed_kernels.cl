MSTRINGIFY(

// Packed INT4 MAD throughput (emulated, not a hardware peak).
//
// Two signed 4-bit values are packed into a signed 8-bit char
// (low nibble + high nibble).  The emulation models the ALU cost of
// "dequant-on-the-fly" inside a 4-bit weight kernel: unpack, multiply,
// accumulate, repack.  Native hardware INT4 (where it exists) is only
// reachable via cooperative matrix -- covered by the Vulkan coopmat
// tests.  Numbers from this test are EXPECTED to be lower than
// compute_char / compute_int8_dp on the same device.
//
// Each INT4_MAC counts as 2 int4 multiply-adds (low + high nibble)
// = 4 int4 ops.  MAD_4 performs 4 MACs = 16 int4 ops per char lane,
// MAD_16 = 64 int4 ops per char lane.  Iteration counts are chosen so
// every variant performs COMPUTE_INT4_PACKED_WORK_PER_WI (4096) int4
// ops per work-item (matches FP compute_fp work budget).

\n#undef MAD_4
\n#undef MAD_16
\n#undef INT4_MAC
\n

// CV = char vector type (char, char2, ...); IV = matching int vector.
// Sign-extend each nibble via arithmetic shifts, do MAC on both lanes,
// then repack by masking back to 4 bits.  Accumulator feeds x back in,
// so the compiler cannot hoist the invariant multiplies.
\n#define INT4_MAC(CV, IV, dst, src) { \
    IV _d = convert_##IV(dst); \
    IV _s = convert_##IV(src); \
    IV _dl = (_d << 28) >> 28; \
    IV _dh = _d >> 4; \
    IV _sl = (_s << 28) >> 28; \
    IV _sh = _s >> 4; \
    _dl = _sl * _dl + _sl; \
    _dh = _sh * _dh + _sh; \
    dst = convert_##CV( (_dl & 0x0F) | ((_dh & 0x0F) << 4) ); \
}

\n#define MAD_4(CV, IV, x, y)   INT4_MAC(CV,IV,x,y); INT4_MAC(CV,IV,y,x); INT4_MAC(CV,IV,x,y); INT4_MAC(CV,IV,y,x);
\n#define MAD_16(CV, IV, x, y)  MAD_4(CV,IV,x,y); MAD_4(CV,IV,x,y); MAD_4(CV,IV,x,y); MAD_4(CV,IV,x,y);
\n

__kernel void compute_int4_packed_v1(__global char *ptr, char _A)
{
    char x = _A;
    char y = (char)get_local_id(0);

    for(int i=0; i<64; i++)
    {
        MAD_16(char, int, x, y);
    }

    ptr[get_global_id(0)] = y;
}

__kernel void compute_int4_packed_v2(__global char *ptr, char _A)
{
    char2 x = (char2)(_A, (_A+1));
    char2 y = (char2)get_local_id(0);

    for(int i=0; i<32; i++)
    {
        MAD_16(char2, int2, x, y);
    }

    ptr[get_global_id(0)] = y.S0 + y.S1;
}

__kernel void compute_int4_packed_v4(__global char *ptr, char _A)
{
    char4 x = (char4)(_A, (_A+1), (_A+2), (_A+3));
    char4 y = (char4)get_local_id(0);

    for(int i=0; i<16; i++)
    {
        MAD_16(char4, int4, x, y);
    }

    ptr[get_global_id(0)] = y.S0 + y.S1 + y.S2 + y.S3;
}

__kernel void compute_int4_packed_v8(__global char *ptr, char _A)
{
    char8 x = (char8)(_A, (_A+1), (_A+2), (_A+3), (_A+4), (_A+5), (_A+6), (_A+7));
    char8 y = (char8)get_local_id(0);

    for(int i=0; i<8; i++)
    {
        MAD_16(char8, int8, x, y);
    }

    ptr[get_global_id(0)] = y.S0 + y.S1 + y.S2 + y.S3 + y.S4 + y.S5 + y.S6 + y.S7;
}

__kernel void compute_int4_packed_v16(__global char *ptr, char _A)
{
    char16 x = (char16)(_A,     (_A+1),  (_A+2),  (_A+3),
                        (_A+4), (_A+5),  (_A+6),  (_A+7),
                        (_A+8), (_A+9),  (_A+10), (_A+11),
                        (_A+12),(_A+13), (_A+14), (_A+15));
    char16 y = (char16)get_local_id(0);

    for(int i=0; i<4; i++)
    {
        MAD_16(char16, int16, x, y);
    }

    char2 t = y.S01 + y.S23 + y.S45 + y.S67 + y.S89 + y.SAB + y.SCD + y.SEF;
    ptr[get_global_id(0)] = t.S0 + t.S1;
}

)
