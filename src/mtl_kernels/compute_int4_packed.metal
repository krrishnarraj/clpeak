// Packed-INT4 emulated MAD throughput.  Two signed 4-bit lanes per signed
// 8-bit container; sign-extend, multiply-accumulate in int, mask and
// repack.  Models the dequant-on-the-fly cost in 4-bit weight kernels;
// not a hardware peak.

#include <metal_stdlib>
using namespace metal;

#define INT4_MAC(dst, src) { \
    int _d = (int)(dst); \
    int _s = (int)(src); \
    int _dl = (_d << 28) >> 28; \
    int _dh = _d >> 4; \
    int _sl = (_s << 28) >> 28; \
    int _sh = _s >> 4; \
    _dl = _sl * _dl + _sl; \
    _dh = _sh * _dh + _sh; \
    dst = ((_dl & 0x0F) | ((_dh & 0x0F) << 4)); \
}

#define MAD_4(x, y)  INT4_MAC(x, y) INT4_MAC(y, x) INT4_MAC(x, y) INT4_MAC(y, x)
#define MAD_16(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y) MAD_4(x, y)

kernel void compute_int4_packed(device int* out [[buffer(0)]],
                                constant int& A [[buffer(1)]],
                                uint tid [[thread_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]])
{
    int x = A;
    int y = (int)lid;

    for (int i = 0; i < 64; i++)
    {
        MAD_16(x, y)
    }

    out[tid] = x + y;
}

// Vectorised INT4_MAC.  MSL applies <<, >>, *, |, & componentwise on
// integer vectors, so the same bit pattern works for int2 / int4.
#define INT4_MAC_V(dst, src) { \
    int4 _d = int4(dst); \
    int4 _s = int4(src); \
    int4 _dl = (_d << 28) >> 28; \
    int4 _dh = _d >> 4; \
    int4 _sl = (_s << 28) >> 28; \
    int4 _sh = _s >> 4; \
    _dl = _sl * _dl + _sl; \
    _dh = _sh * _dh + _sh; \
    dst = ((_dl & 0x0F) | ((_dh & 0x0F) << 4)); \
}
#define INT4_MAC_V2(dst, src) { \
    int2 _d = int2(dst); \
    int2 _s = int2(src); \
    int2 _dl = (_d << 28) >> 28; \
    int2 _dh = _d >> 4; \
    int2 _sl = (_s << 28) >> 28; \
    int2 _sh = _s >> 4; \
    _dl = _sl * _dl + _sl; \
    _dh = _sh * _dh + _sh; \
    dst = ((_dl & 0x0F) | ((_dh & 0x0F) << 4)); \
}

#define MAD_4_V2(x, y)  INT4_MAC_V2(x, y) INT4_MAC_V2(y, x) INT4_MAC_V2(x, y) INT4_MAC_V2(y, x)
#define MAD_16_V2(x, y) MAD_4_V2(x, y) MAD_4_V2(x, y) MAD_4_V2(x, y) MAD_4_V2(x, y)

#define MAD_4_V4(x, y)  INT4_MAC_V(x, y) INT4_MAC_V(y, x) INT4_MAC_V(x, y) INT4_MAC_V(y, x)
#define MAD_16_V4(x, y) MAD_4_V4(x, y) MAD_4_V4(x, y) MAD_4_V4(x, y) MAD_4_V4(x, y)

// int2 vectorised packed-int4: 32 outer * 16 fmas * (per-lane ops) * 2 lanes.
kernel void compute_int4_packed2(device int* out [[buffer(0)]],
                                 constant int& A [[buffer(1)]],
                                 uint tid [[thread_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]])
{
    int2 x = int2(A, A);
    int2 y = int2((int)lid, (int)lid);

    for (int i = 0; i < 32; i++)
    {
        MAD_16_V2(x, y)
    }

    int2 r = x + y;
    out[tid] = r.x + r.y;
}

// int4 vectorised: 16 outer * 16 fmas * (per-lane ops) * 4 lanes.
kernel void compute_int4_packed4(device int* out [[buffer(0)]],
                                 constant int& A [[buffer(1)]],
                                 uint tid [[thread_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]])
{
    int4 x = int4(A);
    int4 y = int4((int)lid);

    for (int i = 0; i < 16; i++)
    {
        MAD_16_V4(x, y)
    }

    int4 r = x + y;
    out[tid] = r.x + r.y + r.z + r.w;
}
