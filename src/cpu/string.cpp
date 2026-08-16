#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

// String-processing throughput tests (Category::String): memchr-style byte
// scan and UTF-8 validation, in GB/s -- the most-executed primitives of
// parsers, JS engines, grep and every C string API.  Kernel bodies live in
// kernels/string_compute.h (SVE scan in kernels/sve_compute.h); per-ISA
// variants come from kernelMenu() like the compute tests.  The inputs are
// 16 KB thread-local buffers -- L1-resident, so the numbers measure the
// scan/classify machinery, not memory bandwidth.  The unit is "gbps" but the
// category is passed explicitly: categoryFromUnit() would otherwise file
// these under Bandwidth.

using clpeak_cpu::kernelMenu;

int CpuPeak::runStringScan(benchmark_config_t &cfg)
{
  emitVariants(*this, {"string_scan", "String scan", "gbps", Category::String,
                       "How fast the CPU hunts through memory for one particular "
                       "byte -- what memchr, strlen and every text search do.  The "
                       "data fits in L1 cache, so this is the scanning machinery, "
                       "not memory speed."},
               "byte_scan", kernelMenu().strscan,
               "no SIMD byte-scan kernel for this CPU", cfg);
  return 0;
}

int CpuPeak::runUtf8Validate(benchmark_config_t &cfg)
{
  emitVariants(*this, {"utf8_validate", "UTF-8 validate", "gbps", Category::String,
                       "How fast the CPU checks that text is well-formed UTF-8 -- "
                       "the first thing every parser, browser and web server does to "
                       "incoming text.  Also run on L1-resident data."},
               "utf8", kernelMenu().utf8,
               "no SIMD table-lookup (PSHUFB/TBL) kernel for this CPU", cfg);
  return 0;
}

#endif // ENABLE_CPU
