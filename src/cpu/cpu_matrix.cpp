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
                 "no CPU bf16 matrix engine (AMX / BFMMLA / SME) on this CPU", cfg);
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) || \
    defined(__aarch64__) || defined(_M_ARM64)
    // fp16 matrix exists on both architectures: AMX-FP16 (Granite Rapids) and
    // SME widening FMOPA (Apple M4+, Oryon Gen 3).  Skip the row elsewhere
    // (armv7 / unknown arch), where no fp16 tile engine can exist.
    emitVariants(*this, {"cpu_matrix_fp16", "CPU matrix engine (fp16)", "gflops"},
                 "matrix_fp16", kernelMenu().mat_fp16,
                 "no CPU fp16 matrix engine (AMX-FP16 / SME) on this CPU", cfg);
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // tf32/fp8 matrix are AMX-only (Diamond Rapids) -- x86 exclusive, so only
    // emit them (incl. the Unsupported row) on an x86 build.  On ARM there is
    // no tile instruction for these dtypes today, so the rows would be noise.
    emitVariants(*this, {"cpu_matrix_tf32", "CPU matrix engine (tf32)", "gflops"},
                 "matrix_tf32", kernelMenu().mat_tf32,
                 "no CPU tf32 matrix engine (AMX-TF32) on this CPU", cfg);
    emitVariants(*this, {"cpu_matrix_fp8", "CPU matrix engine (fp8)", "gflops"},
                 "matrix_fp8", kernelMenu().mat_fp8,
                 "no CPU fp8 matrix engine (AMX-FP8) on this CPU", cfg);
#elif defined(__aarch64__) || defined(_M_ARM64)
    // fp32/fp64 matrix are SME-only (fp32 FMOPA is base SME; fp64 needs
    // FEAT_SME_F64F64, which Apple M4+ has) -- ARM exclusive: no x86 engine
    // does fp32/fp64 tiles, so only emit these rows on an arm64 build.
    emitVariants(*this, {"cpu_matrix_fp32", "CPU matrix engine (fp32)", "gflops"},
                 "matrix_fp32", kernelMenu().mat_fp32,
                 "no CPU fp32 matrix engine (SME) on this CPU", cfg);
    emitVariants(*this, {"cpu_matrix_fp64", "CPU matrix engine (fp64)", "gflops"},
                 "matrix_fp64", kernelMenu().mat_fp64,
                 "no CPU fp64 matrix engine (SME F64F64) on this CPU", cfg);
#endif
    return 0;
  }

  emitVariants(*this, {"cpu_matrix_int", "CPU matrix engine (int8)", "gops"},
               "matrix_int8", kernelMenu().mat_int8,
               "no CPU int8 matrix engine (AMX / I8MM / SME) on this CPU", cfg);
  return 0;
}

#endif // ENABLE_CPU
