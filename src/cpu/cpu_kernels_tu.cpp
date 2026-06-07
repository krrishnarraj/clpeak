#ifdef ENABLE_CPU

// One feature TU.  CMake compiles this file once per ISA variant (different
// -m/-arch flags + a unique CLPEAK_ISA_TAG), producing a uniquely-named
// table accessor that cpu_dispatch.cpp wires up at runtime.  All the kernel
// bodies (and the per-TU table) live in cpu_kernels_impl.h.

#include "cpu_kernels_impl.h"

#define CLPEAK_CAT2(a, b) a##b
#define CLPEAK_CAT(a, b) CLPEAK_CAT2(a, b)

#ifndef CLPEAK_ISA_TAG
#define CLPEAK_ISA_TAG generic
#endif

extern "C" const clpeak_cpu::CpuKernelTable *CLPEAK_CAT(clpeak_table_, CLPEAK_ISA_TAG)()
{
  return clpeak_cpu::tuTable();
}

#endif // ENABLE_CPU
