#include <common/common.h>
#include <cstring>

namespace clpeak {
static bool g_verbose = false;
bool verboseEnabled()   { return g_verbose; }
void setVerbose(bool on) { g_verbose = on; }
}

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

unsigned int pickIters(double per_iter_us, unsigned int target_us,
                       unsigned int forced, unsigned int max_iters)
{
  if (forced) return forced;
  if (target_us == 0) target_us = 5000000; // 5s legacy default
  if (per_iter_us < 1.0) per_iter_us = 1.0;
  double want = (double)target_us / per_iter_us;
  if (want < 1.0)               want = 1.0;
  if (want > (double)max_iters) want = (double)max_iters;
  return (unsigned int)want;
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
