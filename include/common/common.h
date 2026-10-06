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

// The image read must never go through a sampler the shader compiler cannot
// see.  Vulkan's VkSampler is a descriptor bound at dispatch time, so a shader
// that samples through it compiles to OpImageSampleExplicitLod and the compiler
// cannot fold away filter and address-mode resolution it will not know until
// the descriptor is bound -- the texture unit redoes that on every request.
// NVIDIA is indifferent and hid this for a long time; while Vulkan sampled, a
// Mali G615 read 6.25 GBPS against the 14.4 OpenCL got through the same memory
// path, and llvmpipe split 20.1 vs 32.7 the same way.  texelFetch is the fix,
// and is what OpenCL's read_imagef(int2) and SYCL's unsampled read() already
// compile to.  The samplers the other backends use are compile-time constants
// -- Metal's constexpr sampler, OpenCL's inline sampler_t, CUDA's
// CU_TR_FILTER_MODE_POINT texture descriptor -- so those compilers do know the
// filter mode and none of them pays this.
//
// The filtered path is worth measuring, but it is a different quantity and has
// its own row -- Metal's texture_sample_rate, where the texture is deliberately
// cache-resident so the filter units, not DRAM, are the limiter.
//
// Image bandwidth races walk shapes, the same way the compute tests race two
// MAD-chain shapes and for the same reason: no single shape is best on every
// vendor.  Here the variable is the image's memory layout, which is the
// driver's choice and not visible through any of these APIs.
//
// Two of the shapes are 1D runs.  A row-major walk gives a warp 32 texels along
// x -- ideal for a linear surface, but 8 scattered chunks of a block-linear one
// -- and the transposed walk is the mirror image.  On an RTX 5060 that is worth
// 270 vs 419 GBPS through CUDA (whose CUarray is always block-linear) while
// Vulkan reads ~415 either way, so a row-major-only test reported the same
// texture path 1.5x apart across APIs.
//
// A 1D run of either orientation still loses half of a *swizzled* layout.  Mali
// stores an optimal-tiled image in 16x16 u-order blocks, where a 64-byte line
// holds a 2x2 quad of RGBA32F texels, so a walk along x takes two texels from
// each line and drops the other two -- and the transpose fails identically on
// the other axis.  Racing only those two hid it: a Mali G615 read 7.29 and 6.53
// GBPS against a 14.2 GBPS global roof.  So the Vulkan backend also races
// blocked shapes, where a run of TILE_W*TILE_H consecutive lanes covers a 2D
// block and one line is consumed by one warp.  That is worth 2x on Mali (14.9)
// and 14% on an M1 Pro (154 -> 176, its global roof), and Apple turns out to
// swizzle too.  Which block size wins is the swizzle granularity and the two
// disagree -- Mali climbs to 16x16, Apple peaks at 2x2 -- so both ends run.
//
// The other five backends carry the two 1D shapes alone.  On an M1 Pro that
// leaves OpenCL at 164 and Metal at 160 where Vulkan's blocked walk reaches
// 176, so porting it is worth real bandwidth on Apple; Mali's OpenCL images are
// evidently linear, since they sit at the roof on the row-major walk already.
//
// Unlike the MAD chains, no walk can flatter the result and no ratio guard is
// needed: every shape reads every pixel exactly once, so the byte count is
// identical and only the access order differs.
//
// Global bandwidth does NOT race, and that is deliberate.  OpenCL carries two
// shapes -- `_local_offset`, where each work-group owns a contiguous
// FETCH_PER_WI*local_size block, and `_global_offset`, a grid-stride sweep --
// and reports the faster; the other five backends implement the local-offset
// one alone.  Measured across an RTX 5060, pocl and Intel OpenCL on a
// Threadripper, and an M1 Pro, grid-stride never wins by more than 0.6% (all of
// it on the 5060, at or below run-to-run noise) while local-offset is up to
// 9.5% ahead on the CPU runtimes.  So the shape every backend already has is
// the one that wins where it matters, and porting the twin to the other five
// would double their global-bandwidth budget to chase 0.6%.  The OpenCL race
// stays as the canary: --verbose prints both rates, and if a device ever shows
// grid-stride ahead by a real margin, that is the signal to port it.

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
//  - The squaring shape runs 128 chain instructions per loop trip (MAD_128).
//    The trip counter's add, compare and branch are ops the per-work-item
//    budget does not count, and Apple's compilers keep the loop as written:
//    on an M1 Pro the identical chain read 4.45 TFLOPS at 16 per trip and
//    5.13 at 128, against a 5.31 peak, and a runtime trip count read the
//    same.  That 16% is why mixed precision, already 128 deep, read above
//    fp32 there.  Compilers that unroll lose nothing: Intel's OpenCL compiler
//    turns both depths into the same 256-instruction trip.  16-wide vectors
//    keep MAD_16 -- already 256 lane-FMAs a trip -- because as one
//    straight-line trip Intel's CPU OpenCL runtime stopped vectorising
//    float16 across work-items, 1962 -> 829 GFLOPS on a Threadripper 3955WX.
//    The affine shape keeps 16 per trip in Vulkan, OpenCL and Metal: deeper,
//    it gained nothing on any GPU measured and cost llvmpipe's emulated fp16
//    26-43%.  oneAPI keeps both shapes as they were and races a third, the
//    uniform-b affine chain at 128 a trip: its loops are pinned rolled
//    (#pragma unroll 1) on every device, Intel's GPUs included, where the
//    affine shape is the one that wins, but in place of its affine shape that
//    one cost Intel's CPU runtime up to 63% at widths 2-4 (compute_float.cpp
//    there has the numbers).  fp64 stays at 16: Vulkan's and
//    oneAPI's 512-op fp64 budget cannot fill a 128-deep trip at width 4, and
//    Apple GPUs, where the counter costs, have no fp64.
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
// Two recurrences, raced.  No single recurrence reaches peak on every vendor:
// the two dominant register files have opposite constraints.
//
//   x = x*x + c      reads {x, x, c}.  Intel Alchemist (Xe-HPG) halves any
//                    three-source mad whose operands are not all distinct
//                    registers, so this lands at 0.496 instructions per lane
//                    per clock on an Arc A380 -- at every vector width, every
//                    chain count and every work-group size.  Full rate on
//                    NVIDIA, Apple, Adreno and pre-Xe Intel.
//
//   x = a*x + b      a per-lane: three distinct registers.  NVIDIA is the
//                    mirror image and halves it at one chain, which four
//                    independent chains restore.
//
// On Alchemist three distinct registers are not enough either.  At SIMD32 a
// 32-lane fp32 value spans four GRFs, two in each register bank in the same
// order, and a mad that reads all three sources from one bank pays for the
// conflict -- which the allocator's natural layout makes the common case.  At
// SIMD16 each value sits in one bank and the conflict mostly goes away, but
// packed fp16 needs SIMD32 for its double rate.  So Vulkan and OpenCL time the
// float families' affine chain at two sub-group widths: Vulkan at the width
// its pipelines are pinned to and at half of it, OpenCL at the compiler's
// choice and pinned to 16 (intel_reqd_sub_group_size, where it is offered).
// On an Arc A380 (driver 8993), whose fp32 peak is ~5.0 TFLOPS, fp32 read
// 4.78-4.83 in Vulkan and 3.92-3.95 in OpenCL at SIMD32 against 4.87-4.88 and
// 4.93-4.98 at 16; mixed precision 3.44-3.55 and 3.98-4.88 against 4.60-4.80
// and 4.68-4.89; fp16 9.48-9.56 at SIMD32 against 4.92-4.95.  The two widths
// race as a clpeak::FormRace (form_race.h) that drops only a clear loser: a
// tie at one vector width does not predict the next, because the allocator
// lays each width out afresh.
//
// A uniform addend is no substitute.  b = the kernel's scalar + 2 caps the
// conflict at one half of a mad, and IGC's own listing (ocloc -device
// acm-g11) showed it conflict-free at SIMD32 -- yet the same A380 read it at
// 4.12 in Vulkan fp32 and 6.60 in Vulkan fp16, against 4.81 and 9.50 with b
// per-lane.  So b is per-lane (a + 2), and a listing is no substitute for
// timing.  oneAPI still races a uniform-b chain as its third kernel (see the
// loop-trip rule above).
//
// Vulkan, OpenCL and oneAPI race these and report the fastest.  Metal, which
// never runs on Alchemist, races the two shapes alone.  That covers every
// backend that can run on Alchemist.
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

// compute_sp/hp/mp_kernels.cl  (16 iters * MAD_128 * 2 ops per MAD = 4096)
static const unsigned int COMPUTE_FP_WORK_PER_WI = 4096;

// fp64 runs at 1/16-1/64 of fp32 on most consumer GPUs, so the same per-WI
// budget as fp32 produces a kernel that's long enough to trip the GPU
// watchdog on some drivers (RDNA4 + RADV was hard-recovering on dvec2/dvec4
// fma loops at the fp32 budget).  Vulkan compute_dp_v* shaders use this.
static const unsigned int COMPUTE_DP_WORK_PER_WI = 512;

// compute_integer/intfast/char/short_kernels.cl  (64 iters * MAD_16 * 2 = 2048)
static const unsigned int COMPUTE_INT_WORK_PER_WI = 2048;

// The int8 dot-product kernels (compute_int8_dp_kernels.cl,
// compute_int8_dp.cu / .hip, compute_int8_dp_v*.comp, and the CPU backend).
// oneAPI has no such test -- see the Gotchas in src/oneapi/AGENTS.md.
// Each dot is 4 INT8 multiply-adds = 8 ops.  All of them spell the chain the
// same way -- one STEP is two dots into a pair of accumulators that feed each
// other -- and every variant issues 512 STEPs, so 1024 dots * 8 ops = 8192 per
// WI (1 chain = 64 iters * 8 steps, 8 chains = 64 * 1 * 8; OpenCL's v16 is
// 32 * 1 * 16).  Vulkan also races a second shape, four accumulators in a
// cycle, over the same 1024 dots.  Read the comment at the top of any of those
// files (Vulkan's is shaders/dp4a_chain.glsl) for why the chain has to be
// shaped that way; a different shape silently reports a wrong number.
static const unsigned int COMPUTE_INT8_DP_WORK_PER_WI = 8192;

// coopmat_*.comp: 16x16x16 tile, 256 MulAdds per subgroup, one subgroup
// (32 threads) per work-group.  Per subgroup: M*N*K*2*MulAdds = 2,097,152 ops;
// per work-item: 2,097,152 / 32 = 65,536 ops.
static const unsigned int COOPMAT_WORK_PER_WI = 65536;

// MulAdds in one trip of the coopmat inner loop.  Must match CM_MMA_PER_TRIP
// in src/vulkan/shaders/coopmat_chain.glsl: the host pushes the trip count,
// the shader runs this many MulAdds per trip, and the product is the MulAdd
// budget above.  Every tile seen so far -- 8x8x8, 8x8x16, 8x8x32, 16x16x16,
// 16x16x32, at subgroup 32 and 64 -- divides exactly by it, so the budget is
// hit on the nose rather than rounded.
static const unsigned int COOPMAT_MMA_PER_TRIP = 16;

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
// System memory
// ---------------------------------------------------------------------------

namespace clpeak {

// Total physical RAM in bytes, or 0 if it cannot be determined.
//
// Benchmarks that size their own buffers need this: a ceiling that suits a
// workstation is a crash on a phone, and the difference between them is two
// orders of magnitude.  Prefer `memoryBudget()` over using this directly.
uint64_t systemMemoryBytes();

// The largest allocation a test should attempt, being `fraction` of physical
// memory, never more than `ceiling`.  When physical memory is unknown the
// ceiling is used, which is why the ceiling should be a figure that is safe
// on modest hardware rather than the most a big machine could manage.
uint64_t memoryBudget(uint64_t ceiling, unsigned fraction = 4);

// How many file descriptors this process holds open, and its soft limit
// (0 when unlimited or unknown).  False where the count cannot be read
// (Windows).  A vendor runtime that leaks descriptors -- LiteRT's WebGPU
// accelerator on Linux exhausts them and is lost -- takes every later
// backend in the process down with it, and this is how a run says so
// instead of leaving the later failures to look like their own.
bool openFileDescriptors(unsigned long &used, unsigned long &limit);

} // namespace clpeak

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

  // `lastLevelCacheBytes` is the device's last-level cache -- OpenCL's
  // CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, CUDA/HIP's L2 size, SYCL's
  // global_mem_cache_size.  Pass 0 when the API has no such query (Vulkan,
  // Metal) or the driver leaves it unset.
  //
  // `dedicatedMemBytes` is a stand-in for the cache levels no API reports, and
  // is used only when it asks for a larger working set than the reported cache
  // does.  Pass it ONLY for memory that is the device's own: on an iGPU or a
  // unified-memory part it is system RAM and the ratio forDevice() applies is
  // meaningless.  Both can only grow globalBWMaxSize, never shrink it.  See
  // forDevice() for why the working set has to outgrow the cache at all.
  static benchmark_config_t forDevice(DeviceType type,
                                      uint64_t lastLevelCacheBytes = 0,
                                      uint64_t dedicatedMemBytes = 0);
};

// ---------------------------------------------------------------------------
// Diagnostics.  Everything clpeak has to say outside a reading -- a missing
// library, a failed kernel build, a calibration decision -- is one message
// at one level, routed through clpeak::logMessage():
//
//   Error    something failed that was expected to work
//   Warning  why something is absent or partial (the old logger::note())
//   Info     a fact worth keeping in every dump, not worth printing
//   Debug    the trace a maintainer reads when a number looks wrong
//
// Info and above are always recorded on the run document's `log` (see
// run_log.h); Debug only when --verbose is on.  The CLI prints its own
// warnings inline with the results, and everything else to stderr under
// --verbose; the GUI gets every entry as an event and keeps it in the file
// a user exports.  A process-global route, like the verbose flag, because
// the emitting sites include free functions and error macros in the
// *_device.cpp files that have no access to the Peak object or the logger.
//
// `source` names the library a message came from when clpeak merely relayed
// it: "onnxruntime", "vulkan", "opencl", or "console" for output captured
// off stdout/stderr.  Empty for clpeak's own messages.
// ---------------------------------------------------------------------------
namespace clpeak {
bool verboseEnabled();
void setVerbose(bool on);

enum class LogLevel { Error, Warning, Info, Debug };
const char *logLevelString(LogLevel level);
LogLevel    logLevelFromString(const std::string &s);

// Route one diagnostic.  Trailing whitespace is dropped; embedded newlines
// (a compiler's build log) are kept -- the line structure is the message.
void logMessage(LogLevel level, const std::string &source, std::string message);

// printf-style form of the same.
void logf(LogLevel level, const char *fmt, ...);

// Where a message goes.  Installed for the duration of a run by RunLog
// (run_log.h); with nothing installed -- `--list-devices`, the GUI's
// catalog enumeration -- messages fall through to stderr, Debug ones only
// under --verbose.
class LogSink {
public:
  virtual ~LogSink() = default;
  virtual void onLog(LogLevel level, const std::string &source,
                     const std::string &message) = 0;
};
void    setLogSink(LogSink *sink);
LogSink *logSink();

// Write straight to the process's real stderr, bypassing any console capture
// in progress (console_mute.h).  What renders a captured line must not write
// into the capture, or the line would be captured again; and stderr is
// unbuffered, so a line that reaches here is on screen before a driver-side
// crash can take the process down.
void stderrWrite(const std::string &text);
// The fd stderrWrite() uses while a capture is redirecting fd 2, or -1.
// Captures nest (an ONNX attach mute inside the inventory's), so a scope
// reads the current one on entry and puts it back on exit.
void setRealStderrFd(int fd);
int  realStderrFd();
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

// One diagnostic at an explicit level: CLPEAK_LOG(Error, "cuInit failed: %s", s).
#define CLPEAK_LOG(level, ...) \
    ::clpeak::logf(::clpeak::LogLevel::level, __VA_ARGS__)

// Debug-level diagnostic -- a no-op unless --verbose was passed.  Gated at
// the call site, not inside logf(), so the arguments are never evaluated
// when it is off: some of them are expensive (an OpenCL build-log query).
// --verbose is the only tool we have for locating a fault inside someone
// else's shader compiler, so every line is flushed as it is written, and
// with -o it is also on disk before the next line runs (run_log.h).
#define CLPEAK_VLOG(...) \
    do { if (::clpeak::verboseEnabled()) \
             ::clpeak::logf(::clpeak::LogLevel::Debug, __VA_ARGS__); } while (0)

#endif  // COMMON_H
