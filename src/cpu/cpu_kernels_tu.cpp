#ifdef ENABLE_CPU

// One feature TU.  CMake compiles this file once per ISA variant (different
// -m/-arch flags + a unique XPUPEAK_ISA_TAG), producing a uniquely-named
// table accessor that cpu_dispatch.cpp wires up at runtime.  The per-TU table
// builder is in cpu_kernels_impl.h; the kernel bodies it references live in the
// kernels/ sub-headers that header pulls in.

#include "cpu_kernels_impl.h"

#define XPUPEAK_CAT2(a, b) a##b
#define XPUPEAK_CAT(a, b) XPUPEAK_CAT2(a, b)

#ifndef XPUPEAK_ISA_TAG
#define XPUPEAK_ISA_TAG generic
#endif

extern "C" const xpupeak_cpu::CpuKernelTable *XPUPEAK_CAT(xpupeak_table_, XPUPEAK_ISA_TAG)()
{
  return xpupeak_cpu::tuTable();
}

#endif // ENABLE_CPU
