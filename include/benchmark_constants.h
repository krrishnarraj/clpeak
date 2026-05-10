#ifndef BENCHMARK_CONSTANTS_H
#define BENCHMARK_CONSTANTS_H

#include <algorithm>
#include <cstdint>

// Centralized tuning constants that MUST match the corresponding values
// hard-coded in the OpenCL kernel source files (.cl).
//
// If you change a value here, update the matching kernel too (and vice versa).

// global_bandwidth_kernels.cl & kernel_latency.cpp
static const unsigned int FETCH_PER_WI = 16;

// local_bandwidth_kernels.cl
static const unsigned int LMEM_REPS = 64;

// atomic_throughput_kernels.cl
// Was 512.  Cut to 256 because float atomicAdd on AMD/RADV (and likely other
// vendors lacking native fp32 atomic add) is emitted as a shader CAS loop:
// each "atomic add" can run 5-20x slower than int_atomic, which at the old
// 512 reps * 33M WIs * 8 iters pushed the dispatch past the GPU watchdog
// (RX 9070 XT was hard-recovering on the float_global variant).  256 keeps
// the per-dispatch window long enough that GPU frequency scaling can ramp
// to peak (cutting further to 64 under-measured M1 by ~20%) while still
// giving 2x headroom against TDR on the slowest atomic_float path.
// Hardcoded inside each shader/kernel -- keep all sites in sync.
static const unsigned int ATOMIC_REPS = 256;

// image_bandwidth_kernels.cl
static const unsigned int IMAGE_FETCH_PER_WI = 16;

// compute_sp/hp/dp_kernels.cl  (128 iters * MAD_16 * 2 ops per MAD = 4096)
static const unsigned int COMPUTE_FP_WORK_PER_WI = 4096;

// fp64 runs at 1/16-1/64 of fp32 on most consumer GPUs, so the same per-WI
// budget as fp32 produces a kernel that's long enough to trip the GPU
// watchdog on some drivers (RDNA4 + RADV was hard-recovering on dvec2/dvec4
// fma loops at the fp32 budget).  Vulkan compute_dp_v* shaders use this.
static const unsigned int COMPUTE_DP_WORK_PER_WI = 512;

// compute_integer/intfast/char/short_kernels.cl  (64 iters * MAD_16 * 2 = 2048)
static const unsigned int COMPUTE_INT_WORK_PER_WI = 2048;

// compute_int4_packed_kernels.cl
// Two int4 lanes per char.  Every variant performs 4096 int4 ops per WI
// (iter count scaled by vector width).  Labeled emulated in the UI/CLI.
static const unsigned int COMPUTE_INT4_PACKED_WORK_PER_WI = 4096;

// compute_int8_dp_kernels.cl
// Each dot_acc_sat(char4, char4, int) is 4 INT8 multiply-adds = 8 ops.
// v1: 64 iters * MAD_DP_16 (16 dots) * 8 ops = 8192 per WI (all variants equal).
static const unsigned int COMPUTE_INT8_DP_WORK_PER_WI = 8192;

// coopmat_*.comp: 16x16x16 tile, 256 iters per subgroup, one subgroup
// (32 threads) per work-group.  Per subgroup: M*N*K*2*ITERS = 2,097,152 ops;
// per work-item: 2,097,152 / 32 = 65,536 ops.
static const unsigned int COOPMAT_WORK_PER_WI = 65536;

// Max work-group size cap.  Hardware may report higher (1024 on most NVIDIA
// GPUs), but we clamp to 256 because v16 kernels hold a float16/double16
// accumulator (~50-64 registers per thread).  At localSize=1024 this exceeds
// the SM register file on e.g. RTX 5060 (65536 regs/SM), causing
// clEnqueueNDRangeKernel to fail with CL_OUT_OF_RESOURCES.  256 matches
// clpeak's historical cap and leaves broad headroom across all devices.
static const unsigned int MAX_WG_SIZE = 256;

// Scale per-launch global thread count to the device's compute-unit count so
// modern high-CU GPUs (H100 132 SMs, MI300X 304 CUs, M3 Ultra 80 cores, etc.)
// don't get under-saturated by a fixed dispatch.  Mirrors the OpenCL backend's
// numCUs * computeWgsPerCU(=2048) * MAX_WG_SIZE(=256) formula.
//
// Floor = 32M to (1) preserve historical behavior on small/low-CU devices and
// (2) keep a safe target when CU count is unknown (e.g. Vulkan on Intel /
// MoltenVK where no vendor property extension is advertised -- pass 0 and the
// floor takes over).  Realized dispatches are still clamped from above by
// per-test buffer / heap budgets.
static inline uint64_t targetGlobalThreads(uint32_t numCUs)
{
  const uint64_t kFloor = 32ULL << 20;            // 32M
  const uint64_t scaled = (uint64_t)numCUs * 2048ULL * (uint64_t)MAX_WG_SIZE;
  return std::max(kFloor, scaled);
}

#endif // BENCHMARK_CONSTANTS_H
