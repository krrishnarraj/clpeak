# src/cpu — Native CPU Backend Implementation

`CpuPeak` class: a plain-C++ / `std::thread` backend that benchmarks the host
CPU. No external dependencies (only a threading library). Built as the
`peak_cpu` static library and compiled with aggressive flags
(`-O3 -ffast-math -march=native` / `-mcpu=native` on Apple/ARM; `/O2 /fp:fast
/arch:AVX2` on MSVC) so the kernels reach CPU peak.

The CPU is modelled as a single device (index 0). The GPU mental model maps
across: SIMD lane ↔ work-item, thread/core ↔ work-group, cache hierarchy ↔
local memory, DRAM ↔ global memory.

## Quick Lookups

- Main class / orchestrator / `runAll()` / `runWorkload()`? → `cpu_peak.cpp`
- CPU detection (name, cores, cache sizes, ISA flags)? → `cpu_device.cpp`
- Pinned barrier thread pool? → `thread_pool.cpp`
- SIMD abstraction (per-ISA vector wrappers)? → `cpu_simd.h`
- Shared 1T/NT compute runner + gflops/gops emit? → `compute_common.h`
- FP compute (fp32/fp64/fp16/bf16/mixed)? → `compute_float.cpp`
- INT compute (int32, int8 dot)? → `compute_int.cpp`
- CPU matrix engine (AMX / SMMLA / BFMMLA)? → `cpu_matrix.cpp`
- DRAM / cache bandwidth? → `bandwidth.cpp`
- Memory (pointer-chase) latency? → `latency.cpp`

## Key Files

| File | Purpose |
|------|---------|
| `cpu_peak.cpp` | `CpuPeak`: ctor, `applyOptions`, `runAll` (category-ordered dispatch), `runWorkload` (warmup + probe + `pickIters` timed batch via the pool), `enumerate`, `printInventory` |
| `cpu_device.cpp` | `detectCpuInfo()` — brand/vendor (CPUID / sysctl / `/proc/cpuinfo`), core counts (incl. P/E split), L1d/L2/L3 per-instance **and aggregate** (`l3TotalBytes`, from `index3/shared_cpu_list` on Linux / summed on Windows) sizes (sysfs / `GetLogicalProcessorInformationEx` / CPUID; on Apple, `hw.perflevel0.*` for the P-core L1/L2 with a fallback to the generic `hw.*` keys), and ISA flags from compiler feature macros (host-accurate under `-march`/`-mcpu=native`) |
| `thread_pool.cpp` | `CpuThreadPool`: persistent workers parked on a CV, `run(n, body)` barrier dispatch, per-core pinning (`pthread_setaffinity_np` / `SetThreadAffinityMask`; advisory no-op on macOS) |
| `cpu_simd.h` | Per-ISA `f32v`/`f64v`/`i32v` wrappers (AVX-512 / AVX2+FMA / SSE2 / NEON / scalar), selected by the *compile flags of the TU it is built in*, with `set`/`load`/`fma`/`add`/`hsum` + a per-ISA accumulator count (`*_NACC`) + `CPU_UNROLL_*` |
| **`cpu_kernels.h`** | Dispatch API: `CpuFeatures`, `CpuKernelTable` (fn-ptr + opsPerIter per kernel), `cpuFeatures()`, `isaName()`, `kernels()` (best variants for this host) |
| **`cpu_kernels_impl.h`** | **All** chain bodies (fp32/fp64/int32/fp16/bf16/mp/int8dp/matrix/readsum), `#if`-gated by compile features, + the per-TU `tuTable()` builder. Included once per feature TU |
| **`cpu_kernels_tu.cpp`** | Thin TU: `#include cpu_kernels_impl.h` + exports `clpeak_table_<tag>()`. CMake compiles it once per ISA (`generic`, `sse42`, `avx2`, `avx512[vnni\|bf16\|fp16]`, `amx`; `fp16`, `fp16fml`, `dotprod`, `bf16`, `i8mm`; or `native`) with that ISA's flags |
| **`cpu_dispatch.cpp`** | Runtime feature probe (x86 CPUID+XGETBV / ARM `getauxval` HWCAP / Apple `sysctlbyname`) and `kernels()` assembly — merges supported TUs (`generic` ungated floor, then higher ISA gated on full feature set) picking the widest variant per kernel |
| `compute_common.h` | `emitCompute()` — runs a chain single-threaded (`ST`) and across all cores (`MT`), emits both metrics |
| `compute_float.cpp` | `runComputeSP/DP/HP/BF16/MP` methods — look up `kernels().fpXX` and emit (or `Unsupported`). Kernel bodies live in `cpu_kernels_impl.h` |
| `compute_int.cpp` | `runComputeInt32`/`runComputeInt8DP` (via `kernels()`) |
| `cpu_matrix.cpp` | `runCpuMatrix` — emits `kernels().mat_int8` / `mat_fp` (AMX / SMMLA / BFMMLA); `Benchmark::Amx`, run in both fp and int phases |
| `bandwidth.cpp` | `runDramBandwidth` (STREAM read/copy/triad), `runCacheBandwidth` (per-level L1/L2/L3, ST+MT, shared-cache MT sets split across threads). The read kernel is `kernels().readsum`; DRAM arrays sized off the **aggregate** L3 (`pickStreamFloats`) + parallel first-touch for NUMA-local placement. No `TransferBW`/memcpy test — on a CPU it just re-measures the STREAM copy path |
| `latency.cpp` | `runMemoryLatency` (random pointer-chase per cache level, ns) |

## Build

- Built by default (`CLPEAK_ENABLE_CPU=ON`); the one backend with no external
  dependency, so it is always ENABLED.
- Optimization flags are scoped to `peak_cpu` only (see `CMakeLists.txt`).
- `peak_common` is compiled with `ENABLE_CPU` too, because `options.cpp` /
  help text gate the CPU flags (`--cpu`, `--amx`, `--cache-bandwidth`,
  `--memory-latency`) on that macro.

## ISA strategy — per-TU build + runtime dispatch

Two build modes, selected by `CLPEAK_CPU_NATIVE_ARCH` (default OFF):

- **Portable (default, OFF)** — produces ONE binary that is safe on any CPU and
  still uses the best ISA the *running* CPU has. The compute/read kernels
  (`cpu_kernels_tu.cpp`) are compiled once per **feature** TU, each with its own
  flags; `cpu_dispatch.cpp` probes the CPU at runtime and assembles `kernels()`
  by picking, per kernel, the widest variant whose *full* feature set is present.
  The non-kernel code (methods, dispatcher) is compiled at the safe baseline, so
  the binary never SIGILLs on an older CPU.
  - x86 TUs: `generic` (SSE2, ungated floor), `sse42`, `avx2`, `avx512`
    (F/BW/VL/DQ), then the fragmented sub-features as their own TUs —
    `avx512vnni`, `avx512bf16`, `avx512fp16`, `amx` (Linux). A TU is only
    *entered* when every feature it was compiled with is present (AVX-512 ≠ one
    feature: Skylake-X has F but not VNNI/BF16/FP16).
  - ARM TUs: `generic` (NEON floor; pinned to `apple-m1` on macOS so the ungated
    floor never bakes in M4-only features), plus Linux/Android `fp16`,
    `fp16fml`, `dotprod`, `bf16`, `i8mm` TUs.
  - Windows: only **real MSVC** (cl.exe, `CMAKE_CXX_COMPILER_ID == "MSVC"`) is
    restricted. clang-cl reports `MSVC=TRUE` in CMake but is classified as
    clang (`_clpeak_real_msvc=OFF`) and takes the GNU-flag path — every `-m` /
    `-march` ISA flag is routed through clang-cl's `/clang:` passthrough
    (`_clpeak_gnuflag`), giving Windows full dtype parity with Linux. The root
    `CMakeLists.txt` auto-prefers clang-cl when no compiler/toolset is pinned
    (`-T ClangCL` for the VS generator, gated on a vswhere probe for the VS
    Clang component; clang-cl from PATH for Ninja/Makefiles).
  - cl.exe x86: core tiers only (`/arch:AVX2`, `/arch:AVX512`) with
    `CLPEAK_CORE_ONLY`, since it can't isolate AVX-512 sub-features.
  - cl.exe ARM64 (`_M_ARM64`): the `generic` TU only, which is already the
    NEON floor (no `-march` flags exist, and MSVC defines no `__ARM_FEATURE_*`
    macros, so the advanced-dtype TUs can't be built). CPU name comes from the
    registry (`ProcessorNameString`) since there is no CPUID.
- **Native (ON)** — a single `native` TU built `-march=native` / `-mcpu=native`,
  merged unconditionally (trusting build==run host). Fastest for a local build,
  **not portable**.

Runtime detection lives in `cpu_dispatch.cpp` (`cpuFeatures()`): x86 CPUID +
XGETBV (checks OS AVX/AVX-512 enablement); ARM `getauxval(AT_HWCAP/2)` on
Linux/Android, `sysctlbyname("hw.optional.arm.FEAT_*")` on Apple, and on
Windows ARM64 the kernel-exported AArch64 ID registers in the registry
(`CentralProcessor\0` values `CP 4020`/`CP 4030`/`CP 4031` =
`ID_AA64PFR0/ISAR0/ISAR1_EL1`; there is no `IsProcessorFeaturePresent()`
constant for most of these features). `cpu_device.cpp`
fills `info.has*` / `isaName` from this same probe, and methods gate on
`kernels().<x>.fn != nullptr` (so the `Unsupported` rows reflect the run host).

Adding a TU: add a `clpeak_add_isa_tu(<tag> <flags>)` call in `CMakeLists.txt`
(guarded by `check_cxx_compiler_flag`) and a matching `#if CLPEAK_TU_<tag>` merge
(with the right feature predicate) in `cpu_dispatch.cpp`. The kernel bodies are
shared — `cpu_kernels_impl.h` `#if`-gates them on the compile-feature macros the
TU's flags define.

## Gotchas

- **Compute kernels must carry a real loop-carried dependency** or `-O3
  -ffast-math` deletes the work and reports a fabricated peak. The FMA chains
  use `acc = acc*b + c` with `b<1` (converges to a finite fixed point, no
  inf/denormal) and a *runtime* trip count.
- **fp16/bf16 constants must survive narrowing.** `b=0.999999` rounds to exactly
  `1.0` in fp16, making `acc=acc*1+0` invariant → the loop is deleted and fp16
  reports hundreds of TFLOPS. Use values distinct from `1.0`/`0.0` after fp16
  rounding (e.g. `0.9995`, `0.001`).
- **The `mp` (FMLAL) chain must be made NONLINEAR — only the accumulator-as-
  multiplicand trick works.** It showed the impossible `mp > fp16` (FMLAL does
  half the flops/instr of fp16 FMA, so `mp <= ~0.5x fp16`); AppleClang kept the
  loop, the aarch64 server clang fabricated it. Root cause: FMLAL only *adds* to
  the fp32 accumulator (operands are fp16), so `acc += m*b` is a LINEAR function
  of the iteration count, and `-ffast-math` scalar-evolution rewrites it to
  `acc = acc0 + N*m*b` outside the loop (every FMLAL deleted). Three fixes FAILED
  because they left the work *linear*: a live-operand recurrence, distinct
  per-chain coefficients, AND even an `asm volatile("+w")` barrier (the server
  clang still closed the form). What works is what the fp16 chain does — make the
  accumulator a *multiplicand* so the recurrence is genuinely nonlinear (no closed
  form). FMLAL's operands are fp16, so narrow `acc -> fp16` and feed it back:
  `acc += narrow(acc)*(-decay) + 1*refill` (fixed point refill/decay keeps it
  bounded; the `vcvt` is the nonlinearity). Costs ~1 `vcvt` per 2 FMLAL, so `mp`
  lands ~0.25x fp16 rather than the ideal ~0.5x — accept it; correctness beats a
  collapsible chain. Diagnostic: a *collapsed* compute kernel reports a NOISY,
  run-to-run-varying number (tiny timing); a real one is rock-steady. `otool -tv`:
  the body is `NACC*2*UNROLL_K` `fmlal` (16*2*4 = 128) interleaved with `vcvt`.
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
- **AMX (Intel) specifics.** The tile intrinsics (`_tile_dpbssd` / `_tile_dpbf16ps`)
  are *opaque* to the optimizer — it doesn't model tile contents as SSA values —
  so the accumulate loop can't be scalar-evolved/collapsed the way the FMLAL `mp`
  chain was (no `asm` barrier needed). 4 accumulator tiles (0–3) + 2 operand tiles
  (4,5) is the canonical config to hide the ~52-cycle TMUL latency; all 4 are
  stored so none is DCE'd. The TILECFG struct is exactly 64 B (palette=1, then
  `colsb`/`rows`), and `_tile_loadconfig` is per-thread (gated by a `thread_local`
  flag). Input tiles are zero (uninitialized) on purpose — AMX throughput is
  data-independent and zeros avoid int32/fp32 accumulator overflow. **Output tiles
  must be `thread_local`** (every MT worker stores to them; shared `static` is a
  data race + false sharing). `arch_prctl(ARCH_REQ_XCOMP_PERM)` is process-wide
  and requested once at `kernels()` init before any worker issues a tile op.
  Untested on real silicon — verify with `objdump` that the hot loop keeps
  `NACC*INNER` `tdpbssd`/`tdpbf16ps`, and that numbers land near spec.
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
- **MSVC does not define the GCC/Clang ISA macros.** `cpu_simd.h` gates on
  `__FMA__` / `__SSE2__` / `__SSE4_1__`, none of which MSVC ever defines (it only
  defines `__AVX2__` / `__AVX512F__` under `/arch:` and `_M_X64`). The symptom is
  the giveaway: fp32 == fp64 and both ~10× low (scalar) while `int` is fine
  (its branch only needs `__AVX2__`). The FP/SSE branches therefore also accept
  `_MSC_VER` / `_M_X64`. If you add a new `cpu_simd.h` ISA branch, give it an
  MSVC alias or it silently degrades to scalar on Windows. **The alias must be
  an architecture macro, not bare `_MSC_VER`**: MSVC ARM64 defines `_M_ARM64`
  (never `__aarch64__`, `_M_X64`, or any `__SSE*`/`__AVX*`), so a bare
  `_MSC_VER` gate selects x86 vector types on Windows ARM64 where no x86 header
  exists — that broke the first Windows ARM64 build. Every NEON branch
  (`cpu_simd.h`, the `readBufferChecksum` NEON path, `cpu_dispatch.cpp`) accepts
  `__aarch64__ || _M_ARM64`, and the int32 SSE4.1 branch gates MSVC on
  `_M_X64 || _M_IX86` **and `!__clang__`** — clang-cl defines `_MSC_VER`/`_M_X64`
  too but, unlike cl.exe, enforces target features (always_inline error when an
  SSE4.1 intrinsic lands in an SSE2-baseline TU), so it must enter intrinsic
  branches via the GNU feature macros (`__SSE4_1__` etc.) only.
- **The NEON kernels are AArch64-only.** The fused FMA (`vfmaq_f32`), horizontal
  reduce (`vaddvq_*`) and fp16 store (`vst1q_f16`) intrinsics don't exist in
  32-bit ARMv7 NEON, so `cpu_simd.h` gates NEON on `__aarch64__` and armeabi-v7a
  uses the scalar `generic` TU. `CMakeLists.txt` builds the armv8.x feature TUs
  only when `_clpeak_arm64` (`CMAKE_SIZEOF_VOID_P EQUAL 8`); the `-march=armv8.x`
  flags are invalid in AArch32 mode regardless.

## Metrics

Compute / cache-bandwidth tests emit an `ST` (single-thread,
one pinned core) and an `MT` (all logical cores) variant — `ST`/`MT` rather than
literal thread counts so results are comparable across machines with different
core counts. Memory-latency is `ST` only (pointer-chase); DRAM bandwidth emits
`read`/`copy`/`triad`.

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
- **Build the CPU backend with clang, not GCC.** NACC=24 assumes the compiler can
  schedule 24 independent FMA accumulator chains; clang (and GCC≥15) do, but
  GCC≤14 serialises them into one register and skips the k-loop unroll, roughly
  halving fp32/fp64 (a Neoverse CI run sat at ~28% of peak with GCC-13 — objdump
  showed every `fmla` targeting one reg + in-loop `str`). So the root
  `CMakeLists.txt` prefers `clang`/`clang++` over GCC on Linux when the user
  hasn't pinned a compiler, and the CI installs clang. Don't reintroduce
  per-compiler NACC constants — fix the toolchain instead. The same policy
  holds on Windows: the root `CMakeLists.txt` prefers clang-cl over cl.exe
  (see the ISA strategy section) — both for codegen and because cl.exe can't
  build the advanced-dtype TUs at all.

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate file, the `runAll()`
  dispatch + the `CpuPeak` interface (`include/cpu/cpu_peak.h`), `CMakeLists.txt`,
  and this file. New CPU-specific tests also need a `Benchmark` enum value +
  `categoryOf()` entry in `include/common/benchmark_enums.h` and a flag in
  `src/common/options.cpp`.
- If you add a new ISA capability gate → set it in `cpu_device.cpp::detectIsa()`
  and document it under `cpu_device_info_t`.
