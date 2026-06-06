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
| `cpu_device.cpp` | `detectCpuInfo()` — brand/vendor (CPUID / sysctl / `/proc/cpuinfo`), core counts (incl. P/E split), L1d/L2/L3 sizes (sysfs / sysctl / `GetLogicalProcessorInformationEx` / CPUID), and ISA flags from compiler feature macros (host-accurate under `-march`/`-mcpu=native`) |
| `thread_pool.cpp` | `CpuThreadPool`: persistent workers parked on a CV, `run(n, body)` barrier dispatch, per-core pinning (`pthread_setaffinity_np` / `SetThreadAffinityMask`; advisory no-op on macOS) |
| `cpu_simd.h` | Per-ISA `f32v`/`f64v`/`i32v` wrappers (AVX-512 / AVX2+FMA / NEON / scalar) with `set`/`load`/`fma`/`add`/`hsum` + a per-ISA accumulator count (`*_NACC`) |
| `compute_common.h` | `emitCompute()` — runs a chain single-threaded (`ST`) and across all cores (`MT`), emits both metrics |
| `compute_float.cpp` | `runComputeSP/DP` (FMA chains), `runComputeHP` (native fp16 FMA), `runComputeBF16` (bf16 dot), `runComputeMP` (fp16-mul→fp32-acc dependent chain) |
| `compute_int.cpp` | `runComputeInt32` (int madd chain), `runComputeInt8DP` (VNNI / dotprod), `runAtomicThroughput` (uncontended / contended / sharded) |
| `cpu_matrix.cpp` | `runCpuMatrix` — Intel AMX tile matmul (int8 + bf16, Linux) / ARM SMMLA (int8) / BFMMLA (bf16); `Benchmark::Amx`, run in both fp and int phases |
| `bandwidth.cpp` | `runDramBandwidth` (STREAM read/copy/triad), `runCacheBandwidth` (per-level L1/L2/L3 read, 1T + NT), `runMemcpyBandwidth` |
| `latency.cpp` | `runMemoryLatency` (random pointer-chase per cache level, ns), `runThreadLatency` (pool round-trip, us) |

## Build

- Built by default (`CLPEAK_ENABLE_CPU=ON`); the one backend with no external
  dependency, so it is always ENABLED.
- Optimization flags are scoped to `peak_cpu` only (see `CMakeLists.txt`).
- `peak_common` is compiled with `ENABLE_CPU` too, because `options.cpp` /
  help text gate the CPU flags (`--cpu`, `--amx`, `--cache-bandwidth`,
  `--memory-latency`) on that macro.

## ISA strategy

The binary is built `-march=native` (x86) / `-mcpu=native` (Apple/ARM), so the
compiler's `__AVX512F__` / `__ARM_FEATURE_*` macros reflect exactly the build
host's capabilities — `cpu_device.cpp` reads those into the `info` flags, and
each advanced kernel is `#if`-guarded on the same macro with a runtime
`info.has*` check, recording a clean `Unsupported` row when the feature is
absent (e.g. bf16 / i8mm on Apple M1). Note: AppleClang's `-march=native`
under-enables fp16/bf16/i8mm — hence `-mcpu=native` is preferred there.

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
- **macOS has no hard thread affinity**, so single-thread (`ST`) numbers vary
  run-to-run as the kernel lands on a P- or E-core. `MT` numbers are stable.
  Pinning is real on Linux/Windows.
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
`fmla`/`fmadd`, no spills). On Apple M1 Pro fp32 lands ~525 GFLOPS `MT`
(~60% of the ~860 GFLOPS NEON theoretical); the gap is all-core frequency
throttling + Firestorm's sustained dependent-FMA rate, not a codegen defect —
raising `NACC` beyond 16 does not help (measured). On x86 with hard pinning the
chains should sit closer to peak. `NACC` is tuned per ISA to the register file
(AVX2=12, AVX-512/NEON=16) — larger spills and regresses.

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate file, the `runAll()`
  dispatch + the `CpuPeak` interface (`include/cpu/cpu_peak.h`), `CMakeLists.txt`,
  and this file. New CPU-specific tests also need a `Benchmark` enum value +
  `categoryOf()` entry in `include/common/benchmark_enums.h` and a flag in
  `src/common/options.cpp`.
- If you add a new ISA capability gate → set it in `cpu_device.cpp::detectIsa()`
  and document it under `cpu_device_info_t`.
