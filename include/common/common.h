#ifndef COMMON_H
#define COMMON_H

#if defined(__APPLE__) || defined(__MACOSX) || defined(__FreeBSD__)
#include <sys/types.h>
#endif

#include <stdlib.h>
#include <cstdio>
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

// image_bandwidth_kernels.cl
static const unsigned int IMAGE_FETCH_PER_WI = 16;

// ---------------------------------------------------------------------------
// The MAD chain, shared by every compute kernel in every backend
//
//   MAD_4(x, c):  x = x*x + c;  four times over.
//
// Every backend spells this in its own language, but the shape is fixed and
// the rules below are what keep the numbers meaningful.  Change one and you
// change what every compute row measures.
//
//  - One live value per lane.  c is a per-thread loop invariant, so vector
//    width W keeps W+1 values live, not 2W.  This replaced a ping-pong form
//    (x = c*x + c; c = x*c + x) that kept 2W live: on Apple GPUs that halved
//    fp32 throughput at W=5..8 and fp16 at W>=9, with an identical instruction
//    count -- the same MADs simply issued at half rate.  Fewer live values
//    also keeps the wide variants off the spill cliff on CPU-backed OpenCL
//    and Vulkan devices, where vector registers are architectural and scarce.
//
//  - One MAD per statement.  MAD_16 is 16 MADs = 32 ops, so the per-work-item
//    totals below are the same as they were under the ping-pong form and the
//    numbers stay comparable in units.
//
//  - Quadratic, never affine.  x = c*x + c is an affine recurrence: two steps
//    compose into c*c*x + c*c + c, so a compiler is free to hoist the
//    coefficient and halve the loop.  Squaring raises the polynomial degree,
//    so folding steps always costs more operations than it saves.  No
//    compiler in the toolchain set does the affine fold today, but quadratic
//    is safe by construction rather than by luck.  The Vulkan backend's
//    second shape (below) is the one exception, and it is why that exception
//    is worth revisiting.
//
// Two shapes, raced.  No single recurrence reaches peak on every vendor: the
// two dominant register files have opposite constraints.
//
//   x = x*x + c      reads {x, x, c}.  Intel Alchemist (Xe-HPG) halves any
//                    three-source mad whose operands are not all distinct
//                    registers, so this lands at 0.496 instructions per lane
//                    per clock on an Arc A380 -- at every vector width, every
//                    chain count and every work-group size.  Full rate on
//                    NVIDIA, Apple, Adreno and pre-Xe Intel.
//
//   x = a*x + b      reads three distinct registers.  Full rate on Alchemist;
//                    NVIDIA is the mirror image and halves it at one chain,
//                    which four independent chains restore.
//
// Vulkan, OpenCL, oneAPI and Metal carry both shapes, time both, and report
// the faster -- landing within 3% of the best measured shape on every device
// tested.  That covers every backend that can run on Alchemist.
//
// CUDA and ROCm deliberately do not race.  Each targets a single vendor, and
// racing costs roughly 2x the compute-test budget: on NVIDIA the squaring
// chain is measured optimal at one chain (1.00 against 0.52 for a single
// affine chain, on both a 5060 and a 4060), so the second shape would be pure
// cost.  AMD is simply unmeasured -- measure both shapes on an AMD GPU before
// deciding, rather than paying for insurance nobody has priced.
//
// Integer families use a third shape rather than the affine one, because an
// integer affine recurrence is legally foldable (integer multiply and add are
// associative and distributive) and Apple's OpenCL compiler does fold it --
// int16/char16/short16 came back 15.5x inflated.  Their second shape rotates
// the multiplier through the other accumulators
//
//   x_k = x_k * x_(k+1) + c
//
// keeping three distinct source registers and instruction-level parallelism
// while staying quadratic, so no closed form exists to fold to.  Floating
// point keeps the affine shape: the same fold there needs FP reassociation,
// which nothing in the toolchain set does by default.
//
//  - The kernel must store x, never c.  c is loop-invariant; storing it lets
//    the entire chain be dead-coded away and produces an absurd number.
//
//  - No two scalar chains may start on the same value.  Independent chains
//    under the same recurrence stay bitwise identical forever, and a compiler
//    that scalarises vectors then CSEs one of them away -- the reading comes
//    out inflated by chains/(chains-1), too small for MAX_ALT_CHAIN_RATIO to
//    catch.  A width-W vector seed already spans (A, A+1, ... A+W-1) across
//    its own components, so chain k must start at least W past chain k-1, not
//    1 past.  This is why the affine width-2 shapes read 4/3 high on NVIDIA
//    fp64 (Vulkan double2 423 GFLOPS on a 5060 whose FP64 units cap near 335).
//    It does not apply to the rotating integer shape, which rewrites x_k from
//    another accumulator, so equal seeds diverge on the first instruction.
//
// Narrow integer types (char/short) drive x to a fixed point within a few
// squarings.  That is fine -- integer multiply is fixed-latency on every
// target here -- but it is why the terminal store matters.
// ---------------------------------------------------------------------------

// Reject an alt-chain reading more than this many times the squaring reading:
// past this it is a compiler that folded the chain, not silicon that liked the
// shape.  Measured legitimate gains top out near 4x (Intel's CPU OpenCL runtime
// goes 402 -> 1633 GFLOPS purely on the independent chains); the one observed
// fold was 15.5x.
static const float MAX_ALT_CHAIN_RATIO = 6.0f;

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
// GPUs), but we clamp to 256 because the v16 kernels hold a float16/double16
// accumulator.  Under the ping-pong chain that was ~50-64 registers per thread
// and at localSize=1024 it exceeded the SM register file on e.g. RTX 5060
// (65536 regs/SM), causing clEnqueueNDRangeKernel to fail with
// CL_OUT_OF_RESOURCES.  The single-accumulator chain roughly halves that, but
// the cap stays at 256: it matches clpeak's historical cap, leaves broad
// headroom across all devices, and the higher setting has not been re-tested
// on the hardware that originally failed.
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

// The native CPU backend has no GPU watchdog to dodge, and its per-test timed
// phases complete much faster, so a longer budget steadies the numbers against
// turbo / scheduler jitter.  Selectable separately via --max-time-cpu.
static const unsigned int DEFAULT_CPU_TARGET_TIME_US = 2000000;  // 2000 ms

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
// Result is clamped to [1, max_iters].  max_iters defaults to 10000 so a single
// dispatch/copy can be used when one iteration already exceeds the target
// budget, while still bounding command-buffer / event-pool size on the GPU
// backends' fast paths.  The CPU backend has no such per-dispatch limit and
// passes a much larger cap so a cheap kernel actually fills its time budget
// instead of stopping at 10000 iterations.
unsigned int pickIters(double per_iter_us, unsigned int target_us,
                       unsigned int forced, unsigned int max_iters = 10000);

// ---------------------------------------------------------------------------
// Benchmark data initialisation
// ---------------------------------------------------------------------------

// Fill an array with xorshift32 pseudo-random bit patterns.  Used to defeat
// transparent hardware memory compression that inflates apparent bandwidth
// when buffer content is predictable (sequential, zero-filled, or constant).
void populate(float *ptr, uint64_t N);

// ---------------------------------------------------------------------------
// String helpers
// ---------------------------------------------------------------------------

// Escape a string for embedding in a JSON string literal (quotes, backslash,
// \n \r \t, and \u%04x for remaining control chars).  Shared by the result
// dump and the device-inventory JSON emitters.
std::string jsonEscape(const std::string &s);

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

// ---------------------------------------------------------------------------
// Verbose diagnostics gate (--verbose).  Backend build logs, kernel launch /
// API errors and similar debug spam are suppressed by default and only
// emitted when verbose is enabled.  A process-global flag is used because the
// gated sites include free functions and error macros in the *_device.cpp
// files that have no access to the Peak object or the logger.
// ---------------------------------------------------------------------------
namespace clpeak {
bool verboseEnabled();
void setVerbose(bool on);
}

// ---------------------------------------------------------------------------
// Cooperative run cancellation.  A process-global atomic flag observed at
// test boundaries: Peak::isAllowed() returns false once cancellation is
// requested, so every remaining test silently no-ops and runAll() unwinds
// quickly.  Backends additionally break out of their device loops.  Used by
// the GUI (via clpeak_ffi) — the CLI never sets it.  Reset at the start of
// each embedded launch.
// ---------------------------------------------------------------------------
namespace clpeak {
void requestCancel();
bool cancelRequested();
void resetCancel();
}

// Gated stderr diagnostic — no-op unless --verbose was passed.
#define CLPEAK_VLOG(...) \
    do { if (::clpeak::verboseEnabled()) fprintf(stderr, __VA_ARGS__); } while (0)

#endif  // COMMON_H
