// Packed-INT4 emulated MAD throughput.  Two signed 4-bit lanes per signed
// 8-bit container; each MAC sign-extends both lanes, multiplies and
// accumulates in int32, then masks back to 4 bits and repacks.  Models the
// dequant-on-the-fly cost in 4-bit weight kernels; clearly NOT a hardware
// peak.  Numbers are expected to be lower than INT8-DP on the same device.
//
// 64 outer iters * 16 MACs * 4 int4 ops = 4096 int4 ops per thread
// (= COMPUTE_INT4_PACKED_WORK_PER_WI).

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
