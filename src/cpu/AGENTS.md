# src/cpu — Native CPU Backend Implementation

`CpuPeak` class: a plain-C++ / `std::thread` backend that benchmarks the host
CPU. No external dependencies (only a threading library). Built as the
`peak_cpu` static library and compiled with aggressive flags
(`-O3 -ffast-math -march=native` / `-mcpu=native` on Apple/ARM; `/O2 /fp:fast
/arch:AVX2` on MSVC) so the kernels reach CPU peak.

The CPU is modelled as a single device (index 0). The GPU mental model maps
across: SIMD lane ↔ work-item, thread/core ↔ work-group, cache hierarchy ↔
local memory, DRAM ↔ global memory, thread-dispatch ↔ kernel launch.

## Quick Lookups

- Main class / orchestrator / `runAll()` / `runWorkload()`? → `cpu_peak.cpp`
- CPU detection (name, cores, cache sizes, ISA flags)? → `cpu_device.cpp`
- Pinned barrier thread pool? → `thread_pool.cpp`
- SIMD abstraction (per-ISA vector wrappers)? → `cpu_simd.h`
- Shared 1T/NT compute runner + gflops/gops emit? → `compute_common.h`
- FP compute (fp32/fp64/fp16/bf16/mixed)? → `compute_float.cpp`
- INT compute (int32, int8 dot) + atomic throughput? → `compute_int.cpp`
- CPU matrix engine (AMX / SMMLA / BFMMLA)? → `cpu_matrix.cpp`
- DRAM / cache / memcpy bandwidth? → `bandwidth.cpp`
- Memory (pointer-chase) + thread-dispatch latency? → `latency.cpp`

## Key Files

| File | Purpose |
|------|---------|
| `cpu_peak.cpp` | `CpuPeak`: ctor, `applyOptions`, `runAll` (category-ordered dispatch), `runWorkload` (warmup + probe + `pickIters` timed batch via the pool), `enumerate`, `printInventory` |
| `cpu_device.cpp` | `detectCpuInfo()` — brand/vendor (CPUID / sysctl / `/proc/cpuinfo`), core counts (incl. P/E split), L1d/L2/L3 per-instance **and aggregate** (`l3TotalBytes`, from `index3/shared_cpu_list` on Linux / summed on Windows) sizes (sysfs / `GetLogicalProcessorInformationEx` / CPUID; on Apple, `hw.perflevel0.*` for the P-core L1/L2 with a fallback to the generic `hw.*` keys), and ISA flags from compiler feature macros (host-accurate under `-march`/`-mcpu=native`) |
| `thread_pool.cpp` | `CpuThreadPool`: persistent workers parked on a CV, `run(n, body)` barrier dispatch, per-core pinning (`pthread_setaffinity_np` / `SetThreadAffinityMask`; advisory no-op on macOS) |
| `cpu_simd.h` | Per-ISA `f32v`/`f64v`/`i32v` wrappers (AVX-512 / AVX2+FMA / NEON / scalar) with `set`/`load`/`fma`/`add`/`hsum` + a per-ISA accumulator count (`*_NACC`) |
| `compute_common.h` | `emitCompute()` — runs a chain single-threaded (`ST`) and across all cores (`MT`), emits both metrics |
| `compute_float.cpp` | `runComputeSP/DP` (FMA chains), `runComputeHP` (native fp16 FMA), `runComputeBF16` (bf16 dot), `runComputeMP` (conversion-free fp16-mul→fp32-acc widening FMLA where the ISA supports it) |
| `compute_int.cpp` | `runComputeInt32` (int madd chain), `runComputeInt8DP` (VNNI / dotprod), `runAtomicThroughput` (uncontended / contended / sharded) |
| `cpu_matrix.cpp` | `runCpuMatrix` — Intel AMX tile matmul (int8 + bf16, Linux) / ARM SMMLA (int8) / BFMMLA (bf16); `Benchmark::Amx`, run in both fp and int phases |
| `bandwidth.cpp` | `runDramBandwidth` (STREAM read/copy/triad), `runCacheBandwidth` (per-level L1/L2/L3 read, 1T + NT, shared-cache MT sets split across threads, load + integer checksum so FP adds do not bottleneck reads), `runMemcpyBandwidth`. DRAM/memcpy arrays are sized off the **aggregate** L3 (`pickStreamFloats`) and first-touched in parallel for NUMA-local placement |
| `latency.cpp` | `runMemoryLatency` (random pointer-chase per cache level, ns), `runThreadLatency` (pool round-trip, us) |

## Build

- Built by default (`CLPEAK_ENABLE_CPU=ON`); the one backend with no external
  dependency, so it is always ENABLED.
- Optimization flags are scoped to `peak_cpu` only (see `CMakeLists.txt`).
- `peak_common` is compiled with `ENABLE_CPU` too, because `options.cpp` /
  help text gate the CPU flags (`--cpu`, `--amx`, `--cache-bandwidth`,
  `--memory-latency`) on that macro.

## ISA strategy

Two build modes, selected by `CLPEAK_CPU_NATIVE_ARCH` (default OFF):

- **Portable (default, OFF)** — baseline ISA only: SSE4.2 on x86, armv8-a on ARM
  (`-msse4.2` / toolchain default; MSVC x64 default). Safe to distribute across
  machines (no illegal instructions on older CPUs). A startup `log->note()`
  advises rebuilding native for best local numbers. *(Caveat: AppleClang on
  macOS implicitly targets the host CPU even with no `-mcpu`, so a macOS
  "portable" build is still host-tuned — Linux is where the baseline downgrade
  actually takes effect. The runtime-dispatch work below makes this uniform.)*
- **Native (ON)** — `-march=native` (x86) / `-mcpu=native` (Apple/ARM): tuned for
  the build host, **not portable**. Fastest for a local build/run.

Today each advanced kernel is `#if`-guarded on a compile feature macro
(`__AVX512F__` / `__ARM_FEATURE_*`) with a runtime `info.has*` check, recording a
clean `Unsupported` row when absent. `cpu_device.cpp` fills `info.has*` from those
compile macros — correct for a native (single-target) build.

**Planned: per-TU runtime ISA dispatch.** To get one *portable* binary that also
uses AVX2/AVX-512/VNNI/… on capable hosts, the compute/read kernels will be
compiled once per **feature** TU (each with its own `-m…`/`/arch:` flags) and
selected at runtime via CPUID (`__builtin_cpu_supports` / `__cpuid`) on x86 and
HWCAP (`getauxval`) / `sysctlbyname` on ARM. AVX-512 is per-feature (F/BW/VL vs
VNNI vs BF16 vs FP16 vs AMX are separate TUs, each entered only when its full
feature set is present). When that lands, `info.has*` moves from compile macros
to the runtime probe and drives both dispatch and the Unsupported rows.

## Gotchas

- **Compute kernels must carry a real loop-carried dependency** or `-O3
  -ffast-math` deletes the work and reports a fabricated peak. The FMA chains
  use `acc = acc*b + c` with `b<1` (converges to a finite fixed point, no
  inf/denormal) and a *runtime* trip count.
- **fp16/bf16 constants must survive narrowing.** `b=0.999999` rounds to exactly
  `1.0` in fp16, making `acc=acc*1+0` invariant → the loop is deleted and fp16
  reports hundreds of TFLOPS. Use values distinct from `1.0`/`0.0` after fp16
  rounding (e.g. `0.9995`, `0.001`).
- **Loop-invariant operands get hoisted.** The mixed-precision kernel multiplies
  a value *derived from the accumulator* (not a constant) so the fp16 multiply
  isn't lifted out of the loop (which would measure only the fp32 adds).
- **Reduce EVERY accumulator, not just `acc[0]`.** If the final reduction reads
  only `acc[0]`, `-O3` dead-code-eliminates the other `NACC-1` chains, leaving a
  single latency-bound chain — and because the op count still assumes all `NACC`
  chains ran, the reported throughput is fabricated (it happened to land near
  "peak" when `NACC ≈ pipes × latency`). int8-dot dropped ~22% once this was
  fixed; fp16/bf16/matrix were affected too. Sum all accumulators (see the fp32
  chain's reduction loop for the pattern).
- **macOS has no hard thread affinity**, so single-thread (`ST`) numbers vary
  run-to-run as the kernel lands on a P- or E-core. `MT` numbers are stable.
  Pinning is real on Linux/Windows.
- **DRAM bandwidth must beat the AGGREGATE LLC, not one L3 slice.** On multi-CCX/
  CCD AMD, `cpu0`'s L3 is one instance (e.g. 16 MB) but the chip total is
  `instances × that` (e.g. 64 MB). `detectCpuInfo` derives `l3TotalBytes` from
  `index3/shared_cpu_list` (Linux) / summed cache entries (Windows); sizing the
  STREAM arrays off `l3CacheBytes` instead let a 64 MB array sit entirely in the
  64 MB aggregate L3 and report ~550 GB/s "DRAM" read (> the DDR ceiling).
- **First-touch the STREAM arrays in parallel.** `std::vector<float> a(N)`
  zero-fills on the calling thread, so every page lands on one NUMA node and
  multi-socket/CCD bandwidth collapses. Use `new float[N]` (untouched) and have
  each worker `populate()` its own chunk so pages are NUMA-local.
- **Unroll the iteration loop** (`CPU_UNROLL_K`) around every compute chain: the
  per-FMA-group loop-control branch is otherwise a scheduling bubble. On
  Firestorm this is ~13% on fp32 and **~4×** on the cheap int32 madd (where the
  branch dominated). `CPU_UNROLL_FULL` on the accumulator loop keeps the NACC
  chains in registers. Verified via `otool -tv`: the hot loop should be N
  back-to-back FMA/dot ops with zero loads/stores.

## Metrics

Compute / cache-bandwidth / thread-latency tests emit an `ST` (single-thread,
one pinned core) and an `MT` (all logical cores) variant — `ST`/`MT` rather than
literal thread counts so results are comparable across machines with different
core counts. Memory-latency is `ST` only (pointer-chase); DRAM bandwidth emits
`read`/`copy`/`triad`; atomics emit `uncontended`/`contended MT`/`sharded MT`.

## Reaching peak (investigation notes)

The dependent FMA chains generate optimal code (verified: back-to-back
`fmla`/`fmadd`, no spills). **`NACC` must hide the FMA latency**: throughput is
`min(num_pipes, NACC / latency)`, so NACC needs to be ≥ `pipes × latency` to
saturate. On Apple M1 Pro (Firestorm: 4 FP pipes) the fp32/fp64 FMLA latency is
~6 cycles, so NACC=16 only reached ~62% of peak — **NACC=24 lifts fp32 from
~545 to ~745 GFLOPS MT (~84% of the ~880 GFLOPS theoretical)** and fp64 from
~272 to ~375. fp16 saturates at NACC=16 (wider lanes / lower effective latency),
which is why fp16 looked like ~3× fp32 before the fix instead of the expected
~2×. The NEON fp32/fp64 NACC is therefore 24, and AVX-512 fp32/fp64/int32 are
also 24 (32 ZMM registers give the headroom). AVX2 stays 12 — only 16 YMM
registers, and fp64 beating oneAPI at 12 confirms it's sufficient there; the
small fp32 gap to Intel's vectorizer on AVX2 is codegen, not accumulator count.
Re-measure a NACC sweep when validating on a new x86 host. The residual gap to
100% is all-core frequency throttling + (on macOS) no hard pinning (ST swings
P↔E core).

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate file, the `runAll()`
  dispatch + the `CpuPeak` interface (`include/cpu/cpu_peak.h`), `CMakeLists.txt`,
  and this file. New CPU-specific tests also need a `Benchmark` enum value +
  `categoryOf()` entry in `include/common/benchmark_enums.h` and a flag in
  `src/common/options.cpp`.
- If you add a new ISA capability gate → set it in `cpu_device.cpp::detectIsa()`
  and document it under `cpu_device_info_t`.
