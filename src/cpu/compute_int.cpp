#ifdef ENABLE_CPU

#include <cpu/cpu_peak.h>
#include <common/common.h>
#include "cpu_kernels.h"
#include "compute_common.h"

using clpeak_cpu::kernelMenu;
using clpeak_cpu::kernels;

int CpuPeak::runComputeInt32(benchmark_config_t &cfg)
{
  emitVariants(*this, {"integer_compute", "Integer compute", "ops",
                       Category::Unknown,
                       "Peak speed on 32-bit whole numbers -- the arithmetic behind "
                       "array indexing, hashing, compression and address math."},
               "int", kernelMenu().int32, "no SIMD int32 path for this CPU", cfg);
  return 0;
}

// The integer dot-product instructions, as one test.  int8 (VPDPBUSD / SDOT)
// and int16 (VPDPWSSD) are the same idea at two widths on the same unit, and
// a CPU either has the VNNI family or it has neither -- so which of them exist
// is one fact about the chip, not two tests.
//
// Both entry points below run the same family; whichever the run reaches first
// opens the test and the other reopens it.  int16 is x86-only (NEON and SVE
// have no 16-bit dot), so on ARM the family is the int8 row alone rather than
// a permanently unsupported int16 row nobody can fix.
namespace {

std::vector<FamilyRow> intDotRows()
{
  std::vector<FamilyRow> rows = {
      { "int8_dp",
        "Four 8-bit products summed in one instruction -- the workhorse of "
        "quantized (compressed) neural networks running on the CPU.",
        "no int8 dot instruction on this CPU",
        &clpeak_cpu::kernelMenu().int8dp },
  };
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  // VPDPWSSD is in every VNNI CPU since Cascade Lake; AVX-VNNI-INT16 adds the
  // mixed-sign forms on Diamond Rapids / Nova Lake.
  rows.push_back(
      { "int16_dp",
        "Two 16-bit products summed in one instruction -- the wider cousin, "
        "for work where 8 bits loses too much accuracy.",
        "no int16 dot instruction on this CPU",
        &clpeak_cpu::kernelMenu().int16dp });
#endif
  return rows;
}

const logger::TestSpec kIntDotSpec = {
    "integer_dot_product", "Integer dot-product compute", "ops",
    Category::Unknown,
    "Peak speed of the CPU's integer dot-product instructions, which multiply "
    "several pairs of small whole numbers and sum them in one step.  Each "
    "reading is a different input width; having them at all is what makes a "
    "CPU worth running a quantized model on.",
    TestShape::Heterogeneous, "data type"};

} // namespace

int CpuPeak::runComputeInt8DP(benchmark_config_t &cfg)
{
  emitFamily(*this, kIntDotSpec, intDotRows(), cfg);
  return 0;
}

int CpuPeak::runComputeInt16DP(benchmark_config_t &cfg)
{
  // Same family, same test.  Reached only when --int8-dot-product-compute was
  // deselected and this one was not; otherwise the call above already ran it,
  // and running it twice would double every reading.
  if (isAllowed(Benchmark::ComputeInt8DP))
    return 0;
  emitFamily(*this, kIntDotSpec, intDotRows(), cfg);
  return 0;
}

int CpuPeak::runComputeIntDiv(benchmark_config_t &cfg)
{
  // Scalar u64 divide: there is no SIMD integer divide on x86 or NEON, so this
  // is a single test with no per-ISA variants -- the kernel is identical in
  // every TU and read straight from kernels().intdiv (always present via the
  // generic floor).  The number is the scalar DIV unit's throughput in Gops.
  const auto &v = kernels().intdiv;
  logger::TestSpec spec{"integer_divide_compute", "Integer divide compute u64", "ops",
                        Category::Unknown,
                        "How many 64-bit whole-number divisions the CPU sustains per "
                        "second.  No CPU has a vector integer divide, so this is one "
                        "narrow scalar unit and usually the slowest instruction on "
                        "the chip.",
                        TestShape::Homogeneous, "threads"};
  auto test = currentDeviceScope->beginTest(spec);
  if (v.fn)
    emitCompute(*this, test, "u64", v.opsPerIter, v.fn, cfg);
  else
    test.skip("u64 ST", ResultStatus::Unsupported, "no integer divide kernel");
  return 0;
}

#endif // ENABLE_CPU
