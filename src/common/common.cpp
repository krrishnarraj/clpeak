#include <common.h>
#include <benchmark_constants.h>
#include <calibrate.h>

benchmark_config_t benchmark_config_t::forDevice(DeviceType type)
{
    benchmark_config_t cfg;
    if (type == DeviceType::Cpu) {
        cfg.globalBWMaxSize   = 1 << 27;
        cfg.computeWgsPerCU   = 512;
        cfg.computeDPWgsPerCU = 256;
        cfg.transferBWMaxSize = 1 << 27;
    } else {  // Gpu / Accelerator
        cfg.globalBWMaxSize   = 1 << 29;
        cfg.computeWgsPerCU   = 2048;
        cfg.computeDPWgsPerCU = 512;
        cfg.transferBWMaxSize = 1 << 29;
    }
    cfg.targetTimeUs       = DEFAULT_TARGET_TIME_US;
    cfg.kernelLatencyIters = 2000;
    return cfg;
}

void Timer::start()
{
    tick = std::chrono::high_resolution_clock::now();
}

float Timer::stopAndTime()
{
    tock = std::chrono::high_resolution_clock::now();
    return (float)(std::chrono::duration_cast<std::chrono::microseconds>(tock - tick).count());
}

void populate(float *ptr, uint64_t N)
{
    for (uint64_t i = 0; i < N; i++)
    {
        ptr[i] = (float)i;
    }
}

void populate(double *ptr, uint64_t N)
{
    for (uint64_t i = 0; i < N; i++)
    {
        ptr[i] = (double)i;
    }
}

uint64_t roundToMultipleOf(uint64_t number, uint64_t base, uint64_t maxValue)
{
    uint64_t n = (number > maxValue) ? maxValue : number;
    return (n / base) * base;
}

void trimString(std::string &str)
{
    size_t pos = str.find('\0');

    if (pos != std::string::npos)
    {
        str.erase(pos);
    }
}
