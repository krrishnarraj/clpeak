#include <opencl/cl_utils.h>

uint64_t roundToMultipleOf(uint64_t number, uint64_t base, uint64_t maxValue)
{
    uint64_t n = (number > maxValue) ? maxValue : number;
    return (n / base) * base;
}

void populate(float *ptr, uint64_t N)
{
    for (uint64_t i = 0; i < N; i++)
    {
        ptr[i] = (float)i;
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
