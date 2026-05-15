#include <common/calibrate.h>

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
