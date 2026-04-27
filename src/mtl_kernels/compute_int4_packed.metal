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
