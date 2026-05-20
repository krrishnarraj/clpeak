#include <opencl/cl_utils.h>
#include <cstring>

uint64_t roundToMultipleOf(uint64_t number, uint64_t base, uint64_t maxValue)
{
    uint64_t n = (number > maxValue) ? maxValue : number;
    return (n / base) * base;
}

void populate(float *ptr, uint64_t N)
{
    // Use pseudo-random data to defeat hardware memory compression (some GPUs
    // transparently compress buffers, inflating apparent bandwidth when the
    // content is predictable/compressible).
    uint32_t state = 0xDEADBEEF;
    for (uint64_t i = 0; i < N; i++)
    {
        // xorshift32
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Reinterpret bits as float; mask off sign+exponent high bit to avoid
        // NaN/Inf (keep exponent in [1,127] range so values are finite).
        uint32_t bits = (state & 0x7F7FFFFF) | 0x00800000;
        float val;
        memcpy(&val, &bits, sizeof(val));
        ptr[i] = val;
    }
}

void trimString(std::string &str)
{
    size_t pos = str.find('\0');
    if (pos != std::string::npos)
    {
        str.erase(pos);
    }
}
