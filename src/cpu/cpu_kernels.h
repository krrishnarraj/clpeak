#ifndef CPU_KERNELS_H
#define CPU_KERNELS_H

#ifdef ENABLE_CPU

#include <cstddef>
#include <cstdint>
#include <vector>

// ===========================================================================
// Runtime ISA dispatch.  The compute / read kernels are compiled once per
// feature TU (cpu_kernels_tu.cpp built N times with different -m/-arch flags;
// see CMakeLists.txt).  At runtime cpu_dispatch.cpp probes CPUID / HWCAP and
// assembles `kernels()` by picking, per kernel, the best variant whose full
// feature set the *running* CPU supports.  Taking the address of a higher-ISA
// function is safe at baseline; it is only ever *called* once guarded.
// ===========================================================================

namespace clpeak_cpu {

// Runtime CPU feature flags (filled by detectFeatures() in cpu_dispatch.cpp).
struct CpuFeatures {
  // x86
  bool sse42 = false, avx2 = false, fma = false;
  bool avx512f = false, avx512bw = false, avx512vl = false, avx512dq = false;
  bool avx512vnni = false, avx512bf16 = false, avx512fp16 = false;
  bool amx_tile = false, amx_int8 = false, amx_bf16 = false;
  // ARM
  bool neon = false, fp16 = false, fp16fml = false, dotprod = false, bf16 = false, i8mm = false;
};

const CpuFeatures &cpuFeatures();   // cached runtime probe
const char       *isaName();        // widest active compute ISA, e.g. "AVX-512" / "NEON"

// A compute chain runs `outer` outer-iterations and returns a sink value.
using ChainFn = double (*)(uint64_t outer);
// A streaming read returns a checksum sink over `iters` passes of `M` floats.
using ReadFn = uint64_t (*)(const float *p, size_t M, uint64_t iters);

struct ChainVariant {
  ChainFn fn = nullptr;
  double  opsPerIter = 0.0;   // flops/ops one thread performs per outer-iteration
};

// One feature TU's offered kernels (null entries = not provided by that ISA).
struct CpuKernelTable {
  ChainVariant fp32, fp64, int32, fp16, bf16, mp, int8dp, mat_int8, mat_fp;
  ReadFn       readsum = nullptr;
  const char  *isaName = "";
};

// The best kernels for THIS host (assembled once on first call).  Used by the
// bandwidth path (readsum), which is memory-bound and only needs one variant.
const CpuKernelTable &kernels();

// One compute variant tagged with its canonical ISA label (e.g. "AVX-512").
// The label lives here, not in ChainVariant, so it can differ per kernel slot
// for the same TU (a feature TU's int8dp is "AVX-512 VNNI" while its base fp32
// would be "AVX-512").
struct IsaVariant {
  ChainVariant v;
  const char  *isa = "";
};

// ALL supported variants of each compute kernel for THIS host, baseline-first
// (low ISA -> high ISA).  The compute tests run every entry so the user can
// compare instruction sets, rather than only the widest one (see kernels()).
struct CpuKernelMenu {
  std::vector<IsaVariant> fp32, fp64, int32, fp16, bf16, mp, int8dp, mat_int8, mat_fp;
};

// Assembled once on first call.
const CpuKernelMenu &kernelMenu();

} // namespace clpeak_cpu

#endif // ENABLE_CPU
#endif // CPU_KERNELS_H
