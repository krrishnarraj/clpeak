#ifndef COMMON_H
#define COMMON_H

#if defined(__APPLE__) || defined(__MACOSX) || defined(__FreeBSD__)
#include <sys/types.h>
#endif

#include <stdlib.h>
#include <chrono>
#include <string>
#include <cstdint>
#include <algorithm>
#include <common/benchmark_enums.h>

#define TAB             "  "
#define NEWLINE         "\n"

#if defined(__APPLE__) || defined(__MACOSX)
#define OS_NAME         "Macintosh"
#elif defined(__ANDROID__)
#define OS_NAME         "Android"
#elif defined(_WIN32)
  #if defined(_WIN64)
  #define OS_NAME     "Win64"
  #else
  #define OS_NAME     "Win32"
  #endif
#elif defined(__linux__)
  #if defined(__x86_64__)
  #define OS_NAME     "Linux x64"
  #elif defined(__i386__)
  #define OS_NAME     "Linux x86"
  #elif defined(__arm__)
  #define OS_NAME     "Linux ARM"
  #elif defined(__aarch64__)
  #define OS_NAME     "Linux ARM64"
  #else
  #define OS_NAME     "Linux unknown"
  #endif
#elif defined(__FreeBSD__)
#define OS_NAME     "FreeBSD"
#else
#define OS_NAME     "Unknown"
#endif

// ---------------------------------------------------------------------------
// Benchmark tuning constants
//
// These MUST match the hard-coded values in kernel / shader source files.
// If you change a value here, update every matching kernel too (and vice versa).
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

// Default --max-time budget (microseconds).  500 ms is comfortably above
// the empirical M1 clock-ramp window (220-440 ms) so peak-frequency steady
// state is reached, while still leaving usable headroom under Adreno's
// 500 ms hangcheck.  This is the single source of truth -- CliOptions,
// benchmark_config_t::forDevice, and the backend constructors all read it.
// Keep the "500 ms" mention in the --help text in src/common/options.cpp in sync.
static const unsigned int DEFAULT_TARGET_TIME_US = 500000;

// Pick an iteration count from a measured per-iter time and a per-test
// time budget.  Used by every backend's runKernel/runDispatches helper to
// size the timed batch so it lands at ~target_us regardless of device
// speed (avoids GPU watchdog hits on slow paths and clock-ramp
// under-measurement on fast paths).
//
//   per_iter_us  measured time per dispatch from a calibration run
//   target_us    per-test budget (cfg.targetTimeUs); 0 => fall back to
//                a 5 s budget (matches the legacy BLAS pickIters
//                behaviour)
//   forced       if non-zero, short-circuit and return this value (the
//                user passed --iters)
//
// Result is clamped to [1, 10000] so a single dispatch/copy can be used when
// one iteration already exceeds the target budget, while still bounding
// command-buffer / event-pool size on fast paths.
unsigned int pickIters(double per_iter_us, unsigned int target_us, unsigned int forced);

// ---------------------------------------------------------------------------
// Benchmark data initialisation
// ---------------------------------------------------------------------------

// Fill an array with xorshift32 pseudo-random bit patterns.  Used to defeat
// transparent hardware memory compression that inflates apparent bandwidth
// when buffer content is predictable (sequential, zero-filled, or constant).
void populate(float *ptr, uint64_t N);

// ---------------------------------------------------------------------------
// Per-device benchmark tuning knobs
// ---------------------------------------------------------------------------

struct benchmark_config_t {
  uint64_t globalBWMaxSize;
  unsigned int computeWgsPerCU;
  unsigned int computeDPWgsPerCU;
  unsigned int targetTimeUs;          // per-test budget for the timed phase
  unsigned int kernelLatencyIters;    // separately-submitted dispatch count
  uint64_t transferBWMaxSize;

  static benchmark_config_t forDevice(DeviceType type);
};

#endif  // COMMON_H
