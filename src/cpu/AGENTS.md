# src/cpu — Native CPU Backend Implementation

`CpuPeak` class: a plain-C++ / `std::thread` backend that benchmarks the host
CPU. No external dependencies (only a threading library). Built as the
`peak_cpu` static library and compiled with aggressive flags (`-O3 -ffast-math`;
`/O2 /fp:fast` on MSVC) so the kernels reach CPU peak. The compute/read kernels
are compiled once **per feature TU** (each with its own `-m`/`-arch` flags) and
selected at runtime — see the ISA strategy section below.

The CPU is modelled as a single device (index 0). The GPU mental model maps
across: SIMD lane ↔ work-item, thread/core ↔ work-group, cache hierarchy ↔
local memory, DRAM ↔ global memory.

## Quick Lookups

- Main class / orchestrator / `runAll()` / `runWorkload()`? → `cpu_peak.cpp`
- CPU detection (name, cores, cache sizes, ISA flags)? → `cpu_device.cpp`
- Pinned barrier thread pool? → `thread_pool.cpp`
- SIMD abstraction (per-ISA vector wrappers)? → `cpu_simd.h`
- Run-all-ISA-variants list / per-ISA labels? → `cpu_dispatch.cpp` (`kernelMenu()`)
- Shared 1T/NT compute runner + per-ISA test emit (`emitVariants`)? → `compute_common.h`
- FP compute (fp32/fp64/fp16/bf16/mixed/fp8 dot)? → `compute_float.cpp`
- INT compute (int32, int8 dot, int16 dot)? → `compute_int.cpp`
- CPU matrix engine (AMX / SMMLA / BFMMLA / SME)? → `cpu_matrix.cpp`
- DRAM / cache bandwidth? → `bandwidth.cpp`
- Memory (pointer-chase) latency? → `latency.cpp`
- **Kernel bodies** (fp32/fp64/int32/read; fp16/bf16/mp/int8/int16/fp8/bf16fma;
  AMX/NEON matrix; SVE; SME)? → the `kernels/` sub-headers (see below)
- **The list of feature TUs** (single source of truth)? → `cpu_tu_registry.h`

## Key Files

| File | Purpose |
|------|---------|
| `cpu_peak.cpp` | `CpuPeak`: ctor, `applyOptions`, `runAll` (category-ordered dispatch), `runWorkload` (warmup + probe + `pickIters` timed batch via the pool), `enumerate`, `printInventory` |
| `cpu_device.cpp` | `detectCpuInfo()` — brand/vendor (CPUID / sysctl / `/proc/cpuinfo`), core counts (incl. P/E split), L1d/L2/L3 per-instance **and aggregate** (`l3TotalBytes`, from `index3/shared_cpu_list` on Linux / summed on Windows) sizes (sysfs / `GetLogicalProcessorInformationEx` / CPUID; on Apple, `hw.perflevel0.*` for the P-core L1/L2 with a fallback to the generic `hw.*` keys), and ISA flags from the `cpu_dispatch.cpp` runtime probe (CPUID / HWCAP / sysctl) |
| `thread_pool.cpp` | `CpuThreadPool`: persistent workers parked on a CV, `run(n, body)` barrier dispatch, per-core pinning (`pthread_setaffinity_np` / `SetThreadAffinityMask`; advisory no-op on macOS) |
| `cpu_simd.h` | Per-ISA `f32v`/`f64v`/`i32v` wrappers (AVX-512 / AVX2+FMA / SSE2 / NEON / scalar), selected by the *compile flags of the TU it is built in*, with `set`/`load`/`fma`/`add`/`hsum` + a per-ISA accumulator count (`*_NACC`) + `CPU_UNROLL_*` |
| **`cpu_kernels.h`** | Dispatch API: `CpuFeatures`, `CpuKernelTable` (fn-ptr + opsPerIter per kernel), `cpuFeatures()`, `isaName()`, `kernels()` (best variants — bandwidth only), and `kernelMenu()` returning `CpuKernelMenu` (per-slot `std::vector<IsaVariant>` = **every** supported ISA variant + its canonical label, baseline-first) for the compute tests |
| **`cpu_kernels_impl.h`** | Per-TU **aggregator**: `#include`s the `kernels/` sub-headers + emits this TU's `tuTable()` from whatever kernels its build flags enabled. Included once per feature TU |
| **`kernels/base_compute.h`** | fp32 / fp64 / int32 FMA-chains + the streaming XOR read. Present in every TU (goes through `cpu_simd.h`) |
| **`kernels/lowp_compute.h`** | Low/mixed-precision compute: fp16 FMA, bf16 dot, mixed-precision FMLAL, int8 dot, int16 dot (x86 VPDPWSSD/WSUD), NEON fp8 dot (FEAT_FP8DOT4), AVX10.2 bf16 vector FMA. `#if`-gated per feature; whole file excluded under `CLPEAK_CORE_ONLY` |
| **`kernels/matrix_compute.h`** | CPU matrix engines: x86 AMX (int8/bf16/fp16/tf32/fp8, sharing `amxConfig16x64()`) + ARM NEON SMMLA/BFMMLA. `CLPEAK_CORE_ONLY`-excluded |
| **`kernels/sve_compute.h`** | ARM SVE (vector-length-agnostic) compute + SVE bf16/i8mm matrix + SVE2 fp8 dot. Gated on `__ARM_FEATURE_SVE && !__ARM_FEATURE_SME` (an SME TU must never pick up non-streaming SVE — Apple has none), `CLPEAK_CORE_ONLY`-excluded; owns the one `#include <arm_sve.h>` |
| **`kernels/sme_compute.h`** | ARM SME (streaming matrix engine; Apple M4+, Oryon Gen 3): ZA outer products (fp32/fp64/bf16/fp16-widening/int8 — FMOPA/BFMOPA/SMOPA, 4 tiles like AMX; fp64 uses all 8 za.d tiles) + streaming-SVE fp32/fp64 vector chains. Gated on `__ARM_FEATURE_SME`, `CLPEAK_CORE_ONLY`-excluded; owns the one `#include <arm_sme.h>` |
| **`cpu_tu_registry.h`** | `CLPEAK_TU_REGISTRY(X)` X-macro: the single list of every feature-TU tag. Drives the unconditional `clpeak_table_<tag>()` forward declarations in `cpu_dispatch.cpp` |
| **`cpu_kernels_tu.cpp`** | Thin TU: `#include cpu_kernels_impl.h` + exports `clpeak_table_<tag>()`. CMake compiles it once per ISA (`generic`, `sse42`, `avx2`, `avxvnni`, `avxvnniint8`, `avxvnniint16`, `avx512[vnni\|bf16\|fp16]`, `avx10bf16`, `amx`, `amxfp16`, `amxtf32`, `amxfp8`; `fp16`, `fp16fml`, `dotprod`, `bf16`, `i8mm`, `fp8dot`, `sve`, `svebf16`, `svei8mm`, `svefp8dot`, `sme`, `smef64`) with that ISA's flags |
| **`cpu_dispatch.cpp`** | Runtime feature probe (x86 CPUID+XGETBV / ARM `getauxval` HWCAP / Apple `sysctlbyname`); TU accessor decls (from `cpu_tu_registry.h`); `kernels()` assembly (merges supported TUs, widest variant per kernel — bandwidth only); and `kernelMenu()` which collects **every** supported variant per kernel with its canonical ISA label (explicit per-slot pushes — encodes the "collapse identical SSE float" rule by pushing only int32 from the `sse42` TU, and labels base vs feature kernels per-slot) |
| `compute_common.h` | `emitCompute()` — runs a chain single-threaded (`ST`) and across all cores (`MT`), emits both metrics; `emitVariants()` — runs **every** ISA variant from a `kernelMenu()` slot as its own test (ISA appended to the display name, `isaSlug()` into the tag for unique result keys), or one untagged `Unsupported` test if none |
| `compute_float.cpp` | `runComputeSP/DP/HP/BF16/MP/FP8DP` — `emitVariants(..., kernelMenu().fpXX, ...)` (one test per supported ISA). The fp8-dot row is arm64-only (`Benchmark::ComputeFP8DP`). Kernel bodies live in the `kernels/` sub-headers |
| `compute_int.cpp` | `runComputeInt32`/`runComputeInt8DP`/`runComputeInt16DP` (via `emitVariants` + `kernelMenu()`; the int16-dot row is x86-only, `Benchmark::ComputeInt16DP`) |
| `cpu_matrix.cpp` | `runCpuMatrix` — `emitVariants(..., kernelMenu().mat_* ...)` (AMX / SMMLA / BFMMLA / SME); `Benchmark::Amx`, run in both fp and int phases. fp16 row on x86+arm64 (AMX-FP16 / SME); tf32/fp8 rows x86-only (AMX); fp32/fp64 rows arm64-only (SME) |
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

**Compute tests run EVERY supported ISA variant, not just the best.** The compute
methods iterate `kernelMenu()` and emit a separate test per ISA — header decorated
with the canonical label, e.g. `Single-precision compute (SSE2)`,
`… (AVX2+FMA)`, `… (AVX-512)` — so users can compare instruction sets. Each ISA's
tag is slugged (`single_precision_compute_avx_512`) so dump/baseline rows stay
unique. fp32/fp64 SSE2 and SSE4.2 codegen is identical, so the `sse42` TU
contributes only its (genuinely different) int32 variant — no duplicate SSE float
row. Bandwidth still uses the single best kernel via `kernels().readsum`, and the
device-header `ISA:` property is still the widest active ISA (`isaName()`).

One **portable** build mode (there is no native/`-march=native` mode — every ISA
is covered by a feature TU + runtime dispatch instead): produces ONE binary that
is safe on any CPU and still uses the best ISA the *running* CPU has. The
compute/read kernels (`cpu_kernels_tu.cpp`) are compiled once per **feature** TU,
each with its own flags; `cpu_dispatch.cpp` probes the CPU at runtime and
assembles `kernels()` by picking, per kernel, the widest variant whose *full*
feature set is present. The non-kernel code (methods, dispatcher) is compiled at
the safe baseline, so the binary never SIGILLs on an older CPU.

  - x86 TUs: `generic` (SSE2, ungated floor), `sse42`, `avx2`, `avxvnni`
    (256-bit VEX AVX-VNNI int8 dot — the client-x86 int8 path for Alder Lake→
    Arrow Lake, Sierra Forest, Zen 5, none of which have AVX-512), `avxvnniint8`
    (256-bit signed×signed int8 dot; Zen 6, Lunar/Arrow Lake), `avxvnniint16`
    (256-bit mixed-sign int16 dot VPDPWSUD; Diamond Rapids, Nova Lake — the
    classic signed VPDPWSSD int16 dot rides in the `avxvnni`/`avx512vnni` TUs,
    having shipped with VNNI all along, so the int16 test gets up to three ISA
    rows), `avx512`
    (F/BW/VL/DQ), then the fragmented sub-features as their own TUs —
    `avx512vnni`, `avx512bf16`, `avx512fp16`, `avx10bf16` (AVX10.2-512 native
    full-rate bf16 vector FMA — a *different* peak from the bf16 dot: real bf16
    multiply-add, Diamond Rapids), `amx` (int8+bf16), and the newer AMX-dtype TUs
    `amxfp16` (Granite Rapids), `amxtf32`, `amxfp8` (Diamond Rapids) —
    clang/clang-cl, x86-64; Linux + Windows. A TU is only
    *entered* when every feature it was compiled with is present (AVX-512 ≠ one
    feature: Skylake-X has F but not VNNI/BF16/FP16). The `avxvnni`/`avxvnniint8`
    TUs are merged **before** `avx512` in `kernels()` so a wider AVX-512 base
    kernel still wins per-slot; their real contribution is the 256-bit int8dp.
    The new matrix dtypes emit their own tests via extra `mat_fp16`/`mat_tf32`/
    `mat_fp8` menu slots (all under the same `Benchmark::Amx` gate, so no new enum),
    and the bf16 FMA via a `bf16fma` slot in the BF16 phase.
  - ARM TUs: `generic` (NEON floor; pinned to `apple-m1` on macOS so the ungated
    floor never bakes in M4-only features), plus Linux/Android `fp16`,
    `fp16fml`, `dotprod`, `bf16`, `i8mm` TUs, a `fp8dot` TU (NEON fp8 4-way
    dot FDOT, FEAT_FP8DOT4 — first shipped by NVIDIA Vera; also built on Apple,
    runtime-gated for future cores), and the
    **SVE** family — `sve` (vector-length-agnostic base compute + int8 SDOT),
    `svebf16` (BFDOT + BFMMLA), `svei8mm` (SMMLA), `svefp8dot` (SVE2 FDOT).
    The SVE TUs are `NOT APPLE`
    (Apple Silicon has no non-streaming SVE — that's SME's streaming mode, see
    the SME TUs below) **and disabled on Windows** (clang's MSVC C++ ABI
    can't mangle the SVE sizeless types, so `#include <arm_sve.h>` fails to compile
    under clang-cl 19 — the NEON feature TUs still build there because NEON types
    mangle fine; Windows SVE detection is disabled to match, so it never claims
    SVE2 it can't run). The Windows exclusion is one toggle — the
    `CLPEAK_CPU_ENABLE_WIN_SVE` option gates the `_clpeak_sve_ok` variable in
    `CMakeLists.txt`; flip it (and add the Windows-ARM64 SVE runtime probe in
    `cpu_dispatch.cpp`) once the clang-cl ABI gap closes. One `sve` binary runs any
    VL (128-bit Oryon/Vera/Graviton4, 256-bit Graviton3); `svebf16`/`svei8mm` are
    on base SVE too (present on Neoverse V1 without SVE2), runtime-gated by
    `HWCAP2_SVEBF16`/`HWCAP2_SVEI8MM`.
  - **ARM SME TUs** (`sme`, `smef64`): built on Apple **and** Linux/Android
    (Apple M4+; Snapdragon X2 Elite / 8 Elite Gen 5 clusters), disabled on
    Windows behind the same `_clpeak_sve_ok` toggle (same sizeless-type ABI
    gap, plus unproven Windows ZA-state support). Compiled with
    `-march=armv8.6-a+sme[-f64f64]`, which deliberately does NOT imply `+sve`,
    and gated by a `check_cxx_source_compiles` probe (the flag alone doesn't
    prove `<arm_sme.h>` + the `__arm_locally_streaming`/`__arm_new("za")`
    keyword attributes work — clang ≥ ~18). The `sme` TU provides the ZA
    outer-product kernels (menu slots `mat_fp32`, and SME variants of
    `mat_fp`/`mat_fp16`/`mat_int8`) plus streaming-SVE fp32/fp64 vector chains
    (pushed into the base fp32/fp64 menus as an "SSVE" row); `smef64` provides
    the fp64 outer product (`mat_fp64`, FEAT_SME_F64F64 — runtime-gated;
    Apple M4+ has it). **The SME TUs are never merged into `kernels()`** —
    menu-push only — so nothing from them is reachable via a non-SME gate.
  - **Apple Silicon** gets the `apple-m1` `generic` floor
    (NEON+fp16+dotprod+fp16fml, shown as `NEON` / `NEON FP16` / `NEON DotProd` /
    `NEON FP16FML`) **plus** an Apple-only `bf16` + `i8mm` TU pair (the
    `elseif(_clpeak_arm64 AND APPLE)` branch), which the M1 floor lacks. So on M2+
    the bf16 (`NEON BF16`), BFMMLA, and SMMLA rows populate via the portable path
    (runtime-gated by `sysctlbyname(FEAT_BF16/I8MM)`, so they stay Unsupported on an
    M1). fp16/dotprod/fp16fml are **not** re-built as Apple feature TUs — they
    already come from the floor, and the `-march=armv8.6-a+bf16`/`+i8mm` flags don't
    enable FullFP16, so the pair doesn't duplicate the floor's fp16 kernel. SVE is
    still `NOT APPLE` (no non-streaming SVE); the scalable-vector story on Apple
    is the SME TU pair above (M4+ via streaming mode, runtime-gated by
    `sysctlbyname(FEAT_SME/SME2/SME_F64F64)` — M1–M3 show Unsupported rows).
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

Runtime detection lives in `cpu_dispatch.cpp` (`cpuFeatures()`): x86 CPUID +
XGETBV (checks OS AVX/AVX-512 enablement); ARM `getauxval(AT_HWCAP/2)` on
Linux/Android, `sysctlbyname("hw.optional.arm.FEAT_*")` on Apple, and on
Windows ARM64 the kernel-exported AArch64 ID registers in the registry
(`CentralProcessor\0` values `CP 4020`/`CP 4030`/`CP 4031` =
`ID_AA64PFR0/ISAR0/ISAR1_EL1`; there is no `IsProcessorFeaturePresent()`
constant for most of these features). `cpu_device.cpp`
fills `info.has*` / `isaName` from this same probe, and the compute tests emit an
`Unsupported` row when a `kernelMenu()` slot is empty (so the skips reflect the run
host).

Adding a TU (four edits, one per concern):
1. **Kernel body** — put it in the matching `kernels/` sub-header (`base_compute.h`
   for fp32/fp64/int32, `lowp_compute.h` for narrow-scalar/vector dtypes,
   `matrix_compute.h` for tile/matmul engines, `sve_compute.h` for SVE), `#if`-gated
   on the compile-feature macro the TU's flags define, plus a `CPU_HAS_<X>` /
   `CPU_MAT_<X>` define, and a table slot in `cpu_kernels_impl.h`'s `tuTable()`.
2. **Registry** — add `CLPEAK_TU(<tag>)` to `cpu_tu_registry.h` (this alone gets the
   accessor forward-declared; the declaration is unconditional so no `#if` needed).
3. **CMake** — add a `clpeak_add_isa_tu(<tag> <flags>)` call in `CMakeLists.txt`,
   guarded by `check_cxx_compiler_flag`.
4. **Dispatch wiring** — a `#if CLPEAK_TU_<tag>` merge in `kernels()` (best-variant,
   bandwidth) **and** a `#if CLPEAK_TU_<tag>` push in `kernelMenu()` (with the right
   feature predicate + canonical ISA label) so the new ISA shows up as its own
   compute test.

## Gotchas

- **Compute kernels must carry a real loop-carried dependency** or `-O3
  -ffast-math` deletes the work and reports a fabricated peak. The FMA chains
  use `acc = acc*b + c` with `b<1` (converges to a finite fixed point, no
  inf/denormal) and a *runtime* trip count.
- **fp32/fp64 affine coefficients (`b`,`c`) must be `volatile`-seeded** (the
  generic non-NEON chains). On
  non-FMA targets (the SSE2 `generic` TU, scalar fallback) `f32_fma`/`f64_fma`
  are a *transparent* `mul`+`add`, not an opaque hardware FMA. With `b`,`c` as
  compile-time constants, `-ffast-math` closes the `CPU_UNROLL_K`-unrolled chain
  — `N` steps of `acc=b*acc+c` fold to `acc*b^N + const` (verified: 4 steps →
  one `mulps`+`addps`) — deleting most of the work while `opsPerIter` still
  counts it, so SSE2 reported an impossible peak *above* AVX2. Seeding `b`,`c`
  through `volatile` (read once into a register before the loop, exactly like
  the int32 chain's multiplier) blocks the fold and is a no-op on FMA targets
  (AVX2/AVX-512 keep their real FMAs). This only surfaced once the run-all
  `kernelMenu()` started exercising the SSE2 fp variant — the old best-only
  dispatch never ran it.
- **The NEON fp32/fp64/fp16 chains are SELF-QUADRATIC (`acc = acc + acc*acc`),
  not affine — FMLA is destructive on the ADDEND.** AArch64 NEON has no
  FMAD-form instruction: `vfmaq(c, acc, b)` (= `acc*b + c`) forces the compiler
  to re-materialise the loop-invariant addend `c` with one `mov.16b` per FMLA.
  Apple cores hide the mov (zero-cycle move elimination) but Neoverse/Oryon
  don't — NEON fp32 measured **~2.4x below the SVE rows on Neoverse N2** (mov +
  fmla = 2 vector slots per FMA). `acc = acc + acc*acc` maps to a single
  `fmla v,v,v` (destructive operand IS the accumulator, no coefficient
  registers at all), and being genuinely nonlinear it has no closed form for
  fast-math factoring or scalar evolution to collapse (same argument as the
  FMLAL `mp` fix). Dynamics: from `acc0` in (-1,0) the value decays
  *harmonically* (`acc_n ~ -1/n`) toward 0 and freezes once `acc^2 < ulp/2`
  (~-1e-10 fp32, ~-5e-4 fp16) — always normal, never denormal/inf. Verified:
  hot loops are 96/96/64 back-to-back `fmla` with ZERO vector movs on both
  darwin and linux targets, and even M1 Pro (where the mov was "free") gained
  ~18%: fp32 MT 673→796, fp64 320→388, fp16 1521→1570. Expect ~2x on
  Neoverse/Oryon NEON rows. The x86/scalar chains keep the affine+volatile
  form (FMA3 `vfmadd213` is accumulator-destructive already).
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
  data race + false sharing). The tile XSTATE (component 18) must be granted by
  the OS on first use; `amxPermOk()` in `cpu_dispatch.cpp` requests it once,
  process-wide, before any worker issues a tile op — `arch_prctl(ARCH_REQ_XCOMP_PERM)`
  on Linux, `EnableProcessOptionalXStateFeatures(XSTATE_MASK_AMX_TILE_DATA)` on
  Windows 11/Server 2022 (resolved via `GetProcAddress` so the binary still loads
  on Win10, where it returns false → the Unsupported row). The AMX TU is built on
  both Linux and Windows but clang/clang-cl only (real cl.exe is core-only and
  can't build it). Untested on real silicon — verify with `objdump`/`dumpbin` that
  the hot loop keeps `NACC*INNER` `tdpbssd`/`tdpbf16ps`, and that numbers land near spec.
- **The newer x86 AI dtypes (AMX-FP16/TF32/FP8, AVX-VNNI-INT8, AVX10.2 bf16 FMA)
  are enumerated in scattered CPUID bits — getting one wrong risks a SIGILL, so
  they're triple-gated.** Bit positions (verified against Intel's ISA ref /
  `klauspost/cpuid`): leaf 7 **sub-leaf 1** EAX[21]=AMX-FP16, EDX[4]=AVX-VNNI-INT8,
  EDX[7]=AMX-TF32, EDX[19]=AVX10; leaf 7 **sub-leaf 0** ECX[3]=AMX-FP8 (note the
  odd sub-leaf — it's *not* with the other AMX dtypes); AVX10.2-512 needs leaf
  `0x24` EBX low-byte version ≥ 2 **and** EBX[18] (512-bit) **and** the AVX-512 OS
  XSTATE grant. Every AMX dtype additionally requires `amx_tile` + the XTILEDATA
  permission (`amxPermOk()`), so even a mis-read feature bit can't issue a tile op
  on a non-AMX CPU. The three new AMX kernels share `amxConfig16x64()` (the same
  palette-1 / 16-row / 64-colsb TILECFG as int8/bf16); only the element width sets
  K-per-tile (fp16 K=32, tf32 K=16, fp8 K=64) and the DP intrinsic differ. The
  AVX10.2 `bf16fma` is the one *non-dot* bf16 path — `_mm512_fmadd_pbh` full-rate,
  reduced via a raw bf16→fp32 widening (`memcpy` + `<<16`) because no cast
  intrinsic takes a whole `__m512bh`; its `b` coefficient must not round to 1.0 in
  bf16 (0.98828 is exact). **None of these run on hardware yet** — all
  codegen-verified only (`llvm-objdump --mattr=+amx-fp16,+amx-tf32,+amx-fp8,`
  `+avxvnniint8,+avx10.2-512`: hot loops emit `tdpfp16ps`/`tmmultf32ps`/`tdphf8ps`/
  `vpdpbssd`/`vfmadd*bf16`). Validate on Granite Rapids (AMX-FP16) / Diamond Rapids
  (the rest) when reachable; re-check the tf32/fp8 K-dim ops-per-instr against spec.
- **SVE is vector-length-agnostic, so its kernels break two assumptions the
  fixed-width path bakes in.** (1) *Sizeless types can't be array/struct
  members* — `svfloat32_t acc[NACC]` is a compile error, so the NACC independent
  accumulator chains are declared as individual named registers via the
  `SVE_REP16`/`SVE_REP24` X-macros in `kernels/sve_compute.h` (each `X(i)` expands
  to one `a##i`). (2) *`opsPerIter` is VL-dependent* — it's computed from
  `svcntw()`/`svcntd()`/`svcntb()` at table-build time (in `tuTable()`, only ever
  reached when SVE is the running host's ISA), not a compile-time constant; the VL
  in bytes rides in `CpuKernelTable::sveVLBytes` (propagated by `merge()`) and is
  reported as `ISA: SVE2 (VL=256b)` in the device header. The base compute uses
  `svmad` (`acc = acc*b + c`, one MAD, no reload — the SVE analogue of the NEON
  FMLA chain); int8 is `svdot` (SDOT), bf16 `svbfdot`/`svbfmmla`, int8 matrix
  `svmmla`. NACC=24 for fp/int32 (32 Z-regs give headroom), 16 for the dot/matrix
  kernels — **re-sweep on real Neoverse V2 / Grace silicon** (the NEON NACC=24 was
  tuned on Firestorm; SVE latency/pipe counts differ). Codegen-verified with
  `llvm-objdump --mattr=+sve,+bf16,+i8mm` (Apple's `otool` can't decode SMMLA — it
  prints `.long 0x45189b2X`): the hot loop must be exactly `NACC*CPU_UNROLL_K`
  back-to-back `fmad`/`sdot`/`bfdot`/`bfmmla`/`smmla` on `z` registers with no
  loads/stores. Not yet run on SVE hardware — validate on Graviton3/4, Grace, or a
  GitHub arm64 runner (Cobalt/Neoverse N2 has SVE2).
- **SME kernels are streaming functions with ZA state — five specifics.**
  (1) Each kernel is `__arm_locally_streaming` (+ `__arm_new("za")` when it
  touches ZA): streaming mode is entered/exited per call, so the fn-ptr
  dispatch and thread pool need no changes. (2) The FMOPA/SMOPA intrinsics are
  opaque ZA-state updates (like AMX tiles) — no scalar-evolution collapse —
  and inputs are zero on purpose (data-independent throughput, no overflow).
  All 4 za32 tiles (8 za64 tiles for fp64) are issued and read back so none is
  dead. (3) `opsPerIter` comes from `svcntsw()`/`svcntsd()`/`svcntsb()` — the
  *streaming* VL forms, callable from non-streaming code — in `tuTable()`,
  reached only under the runtime `f.sme` gate; the SVL is reported as
  `+ SME2 (SVL=512b)` in the device header. (4) **The SME unit is shared per
  cluster** on every current implementation (Apple: 1/cluster; X2 Elite: 3;
  8 Elite Gen 5: 2), so MT ≈ #clusters × unit peak, NOT #cores × ST — that's
  hardware behaviour, not a bug. (5) AppleClang 21 enables the `+sme-f64f64`
  *feature* without defining the ACLE `__ARM_FEATURE_SME_F64F64` macro, so the
  fp64 kernel is gated on the macro OR the CMake-passed `CLPEAK_SME_F64F64`
  define (the `clpeak_check_sme_f64` probe proves the intrinsic compiles).
  Codegen-verified (back-to-back `fmopa za0-3.s`/`bfmopa`/`smopa`/`fmopa za.d`,
  streaming `fmad z` chains, `smstart`/`zero {za}` in the prologue); **not yet
  run on SME silicon** — validate on an M4/M5 Mac and a Snapdragon X2 device,
  and expect community M4 baselines (~2 TFLOPS-class fp32 FMOPA per unit,
  ~2× bf16/fp16, ~4× int8).
- **The int16 dot chains must be NONLINEAR — clang models VPDPWSSD.** Unlike
  the int8 VPDPBUSD (opaque), instcombine knows the int16 dot's semantics:
  with constant operands `acc += dot(a,b)` is linear in the trip count, and
  the 4-unrolled 512-bit chain was strength-reduced to 1 VPDPWSSD + 3 VPADDD
  (observed; same collapse class as the FMLAL `mp` chain). Fix: feed the
  accumulator back as the multiplicand — `acc = dp(acc, acc, b)` — with a
  volatile-seeded initial value (int wraparound is well-defined for the
  intrinsic, keeps values bounded, throughput data-independent). Verified:
  16×4 back-to-back `vpdpwssd` / 12×4 `vpdpwsud` after the fix.
- **The ARM fp8 dot needs FPMR set once, not per instruction.** The `_fpm`
  intrinsics take an `fpm_t` operand that programs the FPMR format register;
  the compiler hoists the `msr FPMR` out of the hot loop (verified — exactly
  one `msr FPMR, xzr` before back-to-back `fdot`). The chain is linear
  (`acc += dot(const,const)`) but FDOT is currently an opaque target intrinsic
  so it doesn't collapse — **re-verify with objdump on new clang majors**, and
  if it ever collapses apply the int16 accumulator-feedback fix (fp8 has FCVT
  narrowing under FEAT_FP8 for the feedback). Not yet run on FP8 silicon
  (needs NVIDIA Vera / Olympus).
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
~6 cycles, so NACC=16 only reached ~62% of peak — NACC=24 plus the
self-quadratic NEON chain shape (see the FMLA-destructive-addend gotcha) lifts
**fp32 to ~796 GFLOPS MT (~90% of the ~880 GFLOPS theoretical)** and fp64 to
~388. fp16 saturates at NACC=16 (wider lanes / lower effective latency),
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
