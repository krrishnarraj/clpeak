#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

// CPU matrix engine (tensor-core analog): Intel AMX (x86) / SMMLA + BFMMLA
// (ARM).  The variant is compiled per-ISA and selected at runtime.
//
// One test, one reading per data type -- the engine is a single unit and the
// interesting question is which formats it runs and how fast, which six
// near-identical one-reading tests answered only by their names.  int8 runs in
// the integer phase and reopens the same test to append its readings, carrying
// its own unit (ops, not flops) since it is the one row that is not floating
// point.

using clpeak_cpu::kernelMenu;

int CpuPeak::runCpuMatrix(benchmark_config_t &cfg, Category category)
{
  (void)category;
  const logger::TestSpec base = {
      "cpu_matrix", "CPU matrix engine", "gflops", Category::Unknown,
      "Peak speed of the CPU's built-in matrix engine -- Intel AMX tiles or "
      "Arm SMMLA/BFMMLA/SME, a small tensor unit beside the ordinary vector "
      "units.  Each reading is a different input format; which of them the "
      "engine supports is the main thing that separates one generation from "
      "the next.",
      TestShape::Heterogeneous, "data type"};

  {
    std::vector<FamilyRow> rows = {
        { "bf16",
          "bfloat16 -- the AI-oriented 16-bit format, and the one every matrix "
          "engine supports first.",
          "no CPU bf16 matrix engine (AMX / BFMMLA / SME) on this CPU",
          &kernelMenu().mat_fp },
    };

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86) || \
    defined(__aarch64__) || defined(_M_ARM64)
    // fp16 matrix exists on both architectures: AMX-FP16 (Granite Rapids) and
    // SME widening FMOPA (Apple M4+, Oryon Gen 3).  Skip the row elsewhere
    // (armv7 / unknown arch), where no fp16 tile engine can exist.
    rows.push_back(
        { "fp16",
          "Ordinary half precision, which keeps more digits than bf16 over a "
          "narrower range.",
          "no CPU fp16 matrix engine (AMX-FP16 / SME) on this CPU",
          &kernelMenu().mat_fp16 });
#endif
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // fp8 matrix is AMX-only (Diamond Rapids) -- x86 exclusive, so only emit it
    // (incl. the Unsupported row) on an x86 build.  On ARM there is no tile
    // instruction for this dtype today, so the row would be noise.
    rows.push_back(
        { "fp8",
          "8-bit floats, the narrowest format the engine multiplies directly.",
          "no CPU fp8 matrix engine (AMX-FP8) on this CPU",
          &kernelMenu().mat_fp8 });
#elif defined(__aarch64__) || defined(_M_ARM64)
    // fp32/fp64 matrix are SME-only (fp32 FMOPA is base SME; fp64 needs
    // FEAT_SME_F64F64, which Apple M4+ has) -- ARM exclusive: no x86 engine
    // does fp32/fp64 tiles, so only emit these rows on an arm64 build.
    rows.push_back(
        { "fp32",
          "Full 32-bit precision, which only Arm's SME engine offers -- x86 "
          "tiles stop at 16 bits.",
          "no CPU fp32 matrix engine (SME) on this CPU",
          &kernelMenu().mat_fp32 });
    rows.push_back(
        { "fp64",
          "64-bit precision, for scientific work rather than AI.",
          "no CPU fp64 matrix engine (SME F64F64) on this CPU",
          &kernelMenu().mat_fp64 });
#endif

    emitFamily(*this, base, rows, cfg);
  }

  // Integer row -- same test, reopened; carries its own unit (ops).
  emitFamily(*this, base,
             {{ "int8",
                "8-bit whole numbers, the format quantized neural networks "
                "run on -- and the fastest thing the engine does.",
                "no CPU int8 matrix engine (AMX / I8MM / SME) on this CPU",
                &kernelMenu().mat_int8,
                "gops" }},
             cfg);
  return 0;
}

#endif // ENABLE_CPU
