#include <common/common.h>

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

unsigned int pickIters(double per_iter_us, unsigned int target_us, unsigned int forced)
{
  if (forced) return forced;
  if (target_us == 0) target_us = 5000000; // 5s legacy default
  if (per_iter_us < 1.0) per_iter_us = 1.0;
  double want = (double)target_us / per_iter_us;
  if (want < 1.0)     want = 1.0;
  if (want > 10000.0) want = 10000.0;
  return (unsigned int)want;
}
