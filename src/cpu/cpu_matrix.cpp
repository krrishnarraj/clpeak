#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

// CPU matrix engine (tensor-core analog): Intel AMX (x86) / SMMLA + BFMMLA
// (ARM).  The variant is compiled per-ISA and selected at runtime; run in both
// the fp (bf16) and int (int8) phases like the GPU tensor tests.

using clpeak_cpu::kernels;

int CpuPeak::runCpuMatrix(benchmark_config_t &cfg, Category category)
{
  if (category == Category::FpCompute)
  {
    auto test = currentDeviceScope->beginTest(
      {"cpu_matrix_fp", "CPU matrix engine (bf16)", "gflops"});
    const auto &v = kernels().mat_fp;
    if (v.fn) emitCompute(*this, test, "matrix_bf16", v.opsPerIter, v.fn, cfg);
    else      test.skip("matrix_bf16", ResultStatus::Unsupported,
                        "no CPU bf16 matrix engine (AMX / BFMMLA) on this CPU");
    return 0;
  }

  auto test = currentDeviceScope->beginTest(
    {"cpu_matrix_int", "CPU matrix engine (int8)", "gops"});
  const auto &v = kernels().mat_int8;
  if (v.fn) emitCompute(*this, test, "matrix_int8", v.opsPerIter, v.fn, cfg);
  else      test.skip("matrix_int8", ResultStatus::Unsupported,
                      "no CPU int8 matrix engine (AMX / I8MM) on this CPU");
  return 0;
}

#endif // ENABLE_CPU
