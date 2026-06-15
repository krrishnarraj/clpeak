#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

// CPU matrix engine (tensor-core analog): Intel AMX (x86) / SMMLA + BFMMLA
// (ARM).  The variant is compiled per-ISA and selected at runtime; run in both
// the fp (bf16) and int (int8) phases like the GPU tensor tests.

using clpeak_cpu::kernelMenu;

int CpuPeak::runCpuMatrix(benchmark_config_t &cfg, Category category)
{
  if (category == Category::FpCompute)
  {
    emitVariants(*this, {"cpu_matrix_fp", "CPU matrix engine (bf16)", "gflops"},
                 "matrix_bf16", kernelMenu().mat_fp,
                 "no CPU bf16 matrix engine (AMX / BFMMLA) on this CPU", cfg);
    return 0;
  }

  emitVariants(*this, {"cpu_matrix_int", "CPU matrix engine (int8)", "gops"},
               "matrix_int8", kernelMenu().mat_int8,
               "no CPU int8 matrix engine (AMX / I8MM) on this CPU", cfg);
  return 0;
}

#endif // ENABLE_CPU
