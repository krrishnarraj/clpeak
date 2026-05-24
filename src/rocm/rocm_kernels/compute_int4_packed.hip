#include <hip/hip_runtime.h>

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

extern "C" __global__ void compute_int4_packed(int *out, int A)
{
    int x = A;
    int y = (int)threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < 64; i++)
    {
        MAD_16(x, y)
    }

    out[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}
