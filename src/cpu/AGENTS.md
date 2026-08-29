# src/cpu — Native CPU Backend Implementation

`CpuPeak`: a plain-C++ / `std::thread` backend that benchmarks the host CPU.
No external dependencies (only a threading library). Built as the `peak_cpu`
static library with aggressive flags (`-O3 -ffast-math`; `/O2 /fp:fast` on
MSVC) so the kernels reach CPU peak. The compute/read kernels are compiled once
**per feature TU** (each with its own `-m`/`-arch` flags) and selected at
runtime — see the ISA strategy below.

The CPU is modelled as a single device (index 0). The GPU mental model maps
across: SIMD lane ↔ work-item, thread/core ↔ work-group, cache hierarchy ↔
local memory, DRAM ↔ global memory.

## Quick Lookups

- Main class / orchestrator / `runAll()` / `runWorkload()`? → `cpu_peak.cpp`
- CPU detection (name, cores, cache sizes, ISA flags)? → `cpu_device.cpp`
- Pinned barrier thread pool? → `thread_pool.cpp`
- SIMD abstraction (per-ISA vector wrappers, `NACC`, unroll macros)? → `cpu_simd.h`
- Run-all-ISA-variants list / per-ISA labels? → `cpu_dispatch.cpp` (`kernelMenu()`)
- Shared 1T/NT compute runner + per-ISA test emit (`emitVariants`)? → `compute_common.h`
- FP compute (fp32/fp64/fp16/bf16/mixed/fp8 dot, divide/sqrt)? → `compute_float.cpp`
- INT compute (int32, int8 dot, int16 dot, u64 divide)? → `compute_int.cpp`
- CPU matrix engine (AMX / SMMLA / BFMMLA / SME)? → `cpu_matrix.cpp`
- Apple Accelerate GEMM / BNNS matmul (AMX/SME via library)? → `apple_blas.cpp`
- Crypto throughput (AES / SHA-256 / SHA-512 / CRC32-C)? → `crypto.cpp`
- String throughput (memchr-style scan / UTF-8 validate)? → `string.cpp`
- DRAM / cache bandwidth? → `bandwidth.cpp`
- Memory (pointer-chase) latency / MLP / TLB page-walk? → `latency.cpp`
- Atomics / branch-mispredict / store-to-load forwarding / SMT scaling? → `microarch.cpp`
- **Kernel bodies**? → the `kernels/` sub-headers (see Key Files)
- **The list of feature TUs** (single source of truth)? → `cpu_tu_registry.h`

## Key Files

| File | Purpose |
|------|---------|
| `cpu_peak.cpp` | `CpuPeak`: ctor, `applyOptions`, `runAll` (category-ordered dispatch), `runWorkload` (warmup → MT clock settle → probe → `pickIters` timed batch), `enumerate`, `printInventory` |
| `cpu_device.cpp` | `detectCpuInfo()` — brand/vendor (CPUID / sysctl / `/proc/cpuinfo`, MIDR_EL1 decode on brand-string-less ARM hosts), core counts incl. P/E split, per-instance **and aggregate** cache sizes, ISA flags from the `cpu_dispatch.cpp` probe |
| `thread_pool.cpp` | `CpuThreadPool`: persistent workers parked on a CV, `run(n, body)` barrier dispatch, per-core pinning |
| `cpu_simd.h` | Per-ISA `f32v`/`f64v`/`i32v` wrappers (AVX-512 / AVX2+FMA / SSE2 / NEON / scalar), selected by the *build flags of the TU they compile in*, plus the per-ISA accumulator counts (`*_NACC`) and `CPU_UNROLL_*` |
| `cpu_kernels.h` | Dispatch API: `CpuFeatures`, `CpuKernelTable`, `cpuFeatures()`, `isaName()`, `kernels()` (widest variant per kernel — bandwidth only) and `kernelMenu()` (**every** supported ISA variant + its canonical label — the compute tests) |
| `cpu_kernels_impl.h` | Per-TU aggregator: includes the `kernels/` sub-headers and emits this TU's `tuTable()` from whatever its build flags enabled |
| `kernels/base_compute.h` | fp32 / fp64 / int32 FMA chains, fp divide/sqrt, the scalar u64 integer divide, and the streaming read + vector write/copy kernels. Present in every TU |
| `kernels/crypto_compute.h` | AES-128, SHA-256, SHA-512, CRC32-C. `opsPerIter` counts BYTES so `emitCompute()` lands in GB/s. SHA-512 is ARM-only: x86 SHA512 is *detected* but has no kernel, so that row is Unsupported there |
| `kernels/lowp_compute.h` | fp16 FMA, bf16 dot, mixed-precision FMLAL, int8 dot, int16 dot, NEON fp8 dot, AVX10.2 bf16 vector FMA |
| `kernels/matrix_compute.h` | x86 AMX (int8/bf16/fp16/fp8, sharing `amxConfig16x64()`) + ARM NEON SMMLA/BFMMLA |
| `kernels/string_compute.h` | memchr-style byte scan (incl. the historical SSE4.2 PCMPISTRI row) + Keiser-Lemire UTF-8 validation, over L1-resident buffers. `opsPerIter` counts BYTES |
| `kernels/sve_compute.h` | SVE/SVE2 compute, bf16/i8mm matrix, fp8 dot, and the SVE `strscan`. Gated on `__ARM_FEATURE_SVE && !__ARM_FEATURE_SME`; owns the one `#include <arm_sve.h>` |
| `kernels/sme_compute.h` | SME ZA outer products (fp32/fp64/bf16/fp16/int8) + streaming-SVE vector chains. Gated on `__ARM_FEATURE_SME`; owns the one `#include <arm_sme.h>` |
| `cpu_tu_registry.h` | `CLPEAK_TU_REGISTRY(X)` X-macro: the single list of feature-TU tags, driving the accessor declarations in `cpu_dispatch.cpp` |
| `cpu_kernels_tu.cpp` | Thin TU (`#include cpu_kernels_impl.h` + export `clpeak_table_<tag>()`), compiled once per ISA by `CMakeLists.txt` |
| `cpu_dispatch.cpp` | Runtime feature probe (x86 CPUID+XGETBV / ARM HWCAP / Apple sysctl / Windows-ARM64 registry), the `kernels()` merge, and `kernelMenu()` with its canonical per-slot ISA labels |
| `compute_common.h` | `emitCompute()` — runs a chain `ST` and `MT`, emits both; `emitVariants()` — runs every ISA variant of a `kernelMenu()` slot as its own test (ISA slugged into the tag), or one `Unsupported` test. The `ST`/`MT` reading notes are authored ONCE here — never repeat them at a call site |
| `compute_float.cpp` | `runComputeSP/DP/HP/BF16/MP/FP8DP/DivSqrt` (fp8 dot is arm64-only) |
| `compute_int.cpp` | `runComputeInt32`/`Int8DP`/`Int16DP` (int16 is x86-only) + `runComputeIntDiv` (scalar u64, single un-suffixed test) |
| `crypto.cpp` | `runCryptoAes/Sha256/Sha512/Crc32c` — `Category::Crypto` in GB/s, own `--crypto` flag |
| `string.cpp` | `runStringScan/Utf8Validate` — `Category::String` in GB/s, own `--string` flag |
| `cpu_matrix.cpp` | `runCpuMatrix` — AMX / SMMLA / BFMMLA / SME under `Benchmark::Amx`, run in both the fp and int phases |
| `apple_blas.cpp` | `runAppleBlas` (`--accelerate`, Apple-only) — Accelerate `cblas_?gemm` over a size sweep + `BNNSMatMul` fp16/bf16. **Library calls, not feature TUs**, and the only sanctioned route to Apple's AMX on M1–M3 (where the `matrix_*` ISA rows are correctly Unsupported). Single rows, no ST/MT split: Accelerate threads internally |
| `bandwidth.cpp` | `runDramBandwidth` (STREAM read/copy/triad) + `runCacheBandwidth` (per-level read, ST+MT, plus the L1 write/copy rows that expose the store-port width) |
| `latency.cpp` | `runMemoryLatency` — random pointer-chase per cache level, plus the `DRAM linear`, MLP (`DRAM x8`/`x32`) and TLB-miss rows |
| `microarch.cpp` | `runAtomics`, `runBranchPenalty`, `runStoreForward` (ns-per-op cost probes) and `runSmtScaling` (gflops at 1 thread/core vs all logical threads) |

## Build

- Built by default (`CLPEAK_ENABLE_CPU=ON`); the one backend with no external
  dependency, so it is always enabled.
- Optimization flags are scoped to `peak_cpu` only (see `CMakeLists.txt`).
- `peak_common` is compiled with `ENABLE_CPU` too, because `options.cpp` and the
  help text gate the CPU flags on that macro.
- **The ISA feature TUs never take part in LTO** (`clpeak_add_isa_tu` appends
  `-fno-lto` / `/GL-`). A TU's `-m`/`-march` flags don't survive GCC's LTRANS
  re-compile, so the assembler stops accepting its instructions: gcc 16.2 +
  binutils 2.47 + Arch's `-flto=auto` failed an AMX TU at link time (#193) while
  that same TU had just assembled cleanly at compile time. LTO could also inline ISA code into the
  baseline dispatcher (SIGILL on older CPUs). The GCC CI job builds with
  `-flto=auto` to keep this covered.
- **Build with clang, not GCC.** `NACC=24` assumes the compiler can schedule 24
  independent FMA chains; GCC ≤ 14 serialises them into one register and skips
  the k-loop unroll, roughly halving fp32/fp64. The root `CMakeLists.txt`
  therefore prefers clang (and clang-cl over cl.exe, which additionally cannot
  build the advanced-dtype TUs). Don't reintroduce per-compiler `NACC`
  constants — fix the toolchain instead.

## ISA strategy — per-TU build + runtime dispatch

One **portable** build mode, no `-march=native`: every ISA is covered by a
feature TU plus runtime dispatch, so one binary is safe on any CPU and still
uses the best ISA the *running* CPU has. `cpu_kernels_tu.cpp` is compiled once
per feature TU with that TU's flags; the non-kernel code stays at the safe
baseline. `cpu_dispatch.cpp` probes the CPU and enters a TU only when *every*
feature it was compiled with is present. The rationale for each TU's flags and
platform gating lives in `CMakeLists.txt`.

**Compute tests run EVERY supported ISA variant, not just the best.** The
compute methods iterate `kernelMenu()` and emit one test per ISA — decorated
with the canonical label, e.g. `Single-precision compute (AVX-512)` — so users
can compare instruction sets, with the ISA slugged into the tag so dump/baseline
rows stay unique. Bandwidth still uses the single best kernel via `kernels()`,
and the device-header `ISA:` property is the widest active ISA (`isaName()`).

TU tags (`cpu_tu_registry.h`):

- **x86**: `generic` (SSE2 floor), `sse42`, `avx2`, `avxvnni`, `avxvnniint8`,
  `avxvnniint16`, `avx512`, `avx512vnni`, `avx512bf16`, `avx512fp16`,
  `avx10bf16`, `amx`, `amxfp16`, `amxfp8`; crypto `aes`, `vaes`,
  `sha` — CRC32-C has no TU of its own here, it rides in `sse42` (the CRC32
  instruction is part of SSE4.2).
- **ARM**: `generic` (NEON floor, pinned to `apple-m1` on macOS), `fp16`,
  `fp16fml`, `dotprod`, `bf16`, `i8mm`, `fp8dot`; crypto `aes`, `sha`,
  `sha512`, `crc`; SVE `sve`, `svebf16`, `svei8mm`, `svefp8dot`; SME `sme`,
  `smef64`.
- The `aes`/`sha` tags are **shared** between the x86 and ARM branches — only
  one arch ever builds each.
- The **SVE** TUs are not built on Apple — Apple Silicon has no non-streaming
  SVE. The **SME** TUs *are* built on Apple (M4+; streaming mode is Apple's
  only scalable-vector path, runtime-gated so M1–M3 show Unsupported) as well
  as on Linux/Android.
- Both families are **disabled on Windows** behind the single
  `CLPEAK_CPU_ENABLE_WIN_SVE` toggle (clang-cl cannot yet mangle the sizeless
  types). Windows SVE *detection* is disabled to match, so it never claims an
  ISA it can't run.
- Real cl.exe gets core tiers only (`CLPEAK_CORE_ONLY`) on x86 and the `generic`
  TU alone on ARM64; clang-cl takes the GNU-flag path via `/clang:` and has full
  dtype parity with Linux.

**Adding a TU — four edits, one per concern:**

1. **Kernel body** — into the matching `kernels/` sub-header, `#if`-gated on the
   TU's compile-feature macro, plus a `CPU_HAS_<X>` / `CPU_MAT_<X>` define and a
   table slot in `cpu_kernels_impl.h`'s `tuTable()`.
2. **Registry** — add `CLPEAK_TU(<tag>)` to `cpu_tu_registry.h`.
3. **CMake** — add a `clpeak_add_isa_tu(<tag> <flags>)` call, guarded by
   `check_cxx_compiler_flag`.
4. **Dispatch** — a `#if CLPEAK_TU_<tag>` merge in `kernels()` (bandwidth) *and*
   a push in `kernelMenu()` (with the feature predicate + canonical ISA label)
   so the ISA shows up as its own compute test.

## Gotchas

The per-kernel rationale lives in the kernel headers themselves — this section
carries only the rules that span files. Read the target file's comments before
changing a kernel.

- **Every compute kernel must carry a real loop-carried dependency**, or `-O3
  -ffast-math` deletes the work and reports a fabricated peak while `opsPerIter`
  still counts it. Each dtype dodges collapse differently, and the reasoning for
  each is at the kernel: affine chains need `volatile`-seeded coefficients and
  NEON needs the self-quadratic shape (`base_compute.h`); fp16/bf16 constants
  must survive narrowing, and the FMLAL `mp` and int16-dot chains need
  accumulator feedback to be nonlinear (`lowp_compute.h`); divide/sqrt must dodge
  reciprocal hoisting *and* estimate substitution, which is why the operations
  live inside a `float_control(precise)` region in `cpu_simd.h`; crypto messages
  must be state-derived or LICM hoists the schedule (`crypto_compute.h`); string
  passes need a per-pass memory barrier, since a pure function of a read-only
  buffer is otherwise hoisted wholesale (`string_compute.h`). AMX/SME/FDOT
  intrinsics are opaque and need no barrier — verify that stays true on new
  compiler majors.
  **Diagnostic**: a collapsed kernel reports a NOISY, run-to-run-varying number;
  a real one is rock-steady.
- **Reduce EVERY accumulator, not just `acc[0]`.** Otherwise `-O3` dead-codes the
  other `NACC-1` chains, leaving one latency-bound chain while the op count still
  assumes all of them ran. This lands *near* plausible peak (when
  `NACC ≈ pipes × latency`), so it does not look like a bug.
- **Codegen verification is the acceptance test for a new kernel.** `otool -tv` /
  `llvm-objdump --mattr=…`: the hot loop must be `NACC × CPU_UNROLL_K`
  back-to-back FMA/dot/tile ops with no loads, stores or vector movs. This is how
  every collapse above was caught, and the only check available for an ISA with
  no silicon to hand.
- **Bandwidth kernels reached through a function pointer must be timed with a
  runtime size, and must not share an induction variable with their tail.**
  Compute `nblk` once and walk a single pointer — all three (`readBufferChecksum`,
  `writeBufferFill`, `copyBufferVec`) use that form; keep it when adding another.
  A fixed-size microbenchmark never reproduces the problem, which is why it went
  unnoticed. Detail in `base_compute.h`.
- **Never use libc memset/memcpy for cache-resident bandwidth.** Apple's and
  glibc's switch to non-temporal stores above a size threshold, bypassing the
  cache under test — the first "L1 write" landed exactly at DRAM bandwidth. The
  DRAM STREAM copy keeps libc memcpy on purpose.
- **Size DRAM working sets off the SUM of every cache level, and first-touch
  them in parallel.** Not "the LLC": that name fails twice over. On
  multi-CCX/CCD AMD `cpu0`'s L3 is one slice, and on a chip whose last level
  *is* the L2 (Apple Silicon, Snapdragon X, most phone SoCs) there is no L3 to
  size off at all — a Snapdragon X Elite has 36 MB of aggregate L2 and reports
  none. `l1dTotalBytes + l2TotalBytes + l3Total` at the classic STREAM 4x
  margin covers both; `pickStreamFloats` (`bandwidth.cpp`) and the DRAM
  pointer-chase (`latency.cpp`) both use it. Single-threaded init puts every
  page on one NUMA node and collapses the number.
- **No L3 is a real answer — never fabricate one.** `detectCpuInfo` leaves
  `l3CacheBytes` at 0 when the OS reports no L3, and the L3 cache-bandwidth and
  latency rows skip as Unsupported rather than measure a made-up working set
  that still fits in L2. An invented 8 MB there put an "L3" in the device
  header of every Apple part, reported an L3 *faster* than the L2 next to it,
  and undersized the STREAM arrays. Apple additionally needs its L2 total
  summed over `hw.perflevel*.cpusperl2` — one sysctl reports one cluster.
- **macOS has no hard thread affinity**, so `ST` numbers vary run-to-run as the
  kernel lands on a P- or E-core; `MT` is stable. Pinning is real on
  Linux/Windows. This is also why SMT scaling reports Unsupported on macOS —
  it needs one-thread-per-physical-core placement.
- **MSVC does not define the GCC/Clang ISA macros**, and its architecture macros
  are its own (`_M_X64`, `_M_ARM64` — never `__aarch64__` or `__SSE*`). A new
  `cpu_simd.h` branch without an MSVC alias silently degrades to scalar; a bare
  `_MSC_VER` alias instead selects x86 types on Windows ARM64. Gate on the
  architecture macro, and note that clang-cl defines `_MSC_VER` too but enforces
  target features, so it must enter intrinsic branches via the GNU macros only.
- **The NEON kernels are AArch64-only** (fused FMA, horizontal reduce and fp16
  store have no ARMv7 equivalent), so armeabi-v7a uses the scalar `generic` TU.

## Reference points

Numbers to sanity-check a change against; anything far off is a bug, not a win.

- **M1 Pro**: fp32 ~796 GFLOPS MT (~90% of theoretical), fp64 ~388, L1 read
  47.6 B/cycle (99% of the 3×16B load-port ceiling), L1 write ST ~95 GB/s
  (~93% of the 2×16B store bound), AES ~14.6 GB/s ST, SHA-256 ~3.1, SHA-512
  ~2.1, CRC32-C ~25.5 (= exactly 1 `crc32cx`/cycle), string scan 85 GB/s ST,
  UTF-8 validate 12, Accelerate sgemm 1.8–2.5 TFLOPS (wide turbo variance —
  don't chase a single number), dgemm ~0.6, BNNS fp16 ~2.0.
- **Store-forward** splits into three groups, which is what makes the row worth
  keeping: AMD/Apple ~1 cycle, Neoverse ~6, Intel ~7.5. See `microarch.cpp` for
  why ~1 cycle is *not* proof of memory renaming.
- **`NACC` must hide the FMA latency** (`min(pipes, NACC/latency)`), so it needs
  to be ≥ `pipes × latency`. NACC=16 left M1 Pro fp32 at ~62% of peak. Re-sweep
  when validating on a new x86 host or on real SVE silicon.
- Expect write MT to scale far worse than read MT on Apple (~2.5× vs ~7.7×):
  the store path has a shared ceiling, reproduced outside clpeak.
- **DRAM `read` above the memory's rated peak is always a sizing bug**, never a
  win — the arrays are partly cache-resident. `--verbose` prints the array size
  and the total cache it has to clear. `copy`/`triad` landing well *below*
  `read` is not the mirror-image bug: they count only the bytes asked for while
  most CPUs also fetch each line before overwriting it, so ~1.5x of copy's
  traffic and ~1.33x of triad's goes uncounted. Apple is the outlier that makes
  `copy` come out *above* `read` (M1 Pro: read ~119, copy ~158, triad ~132).
- The STREAM triad row **is** vectorised on both arm64 and x86 — do not
  re-investigate that. It reads below copy everywhere, but collapses
  specifically on Windows (triad/copy 0.32–0.38 vs 0.60–0.92
  elsewhere), which tracks page-size/TLB pressure over three concurrent 256 MB
  streams, not codegen.

## Metrics

Compute and cache-bandwidth tests emit an `ST` (one pinned core) and an `MT`
(all logical cores) variant — labelled `ST`/`MT` rather than literal thread
counts, so results compare across machines with different core counts. DRAM
bandwidth emits `read`/`copy`/`triad`; memory latency is `ST` only.

Crypto and string tests report GB/s with unit `gbps` **and the category passed
explicitly** in the `TestSpec` — `categoryFromUnit("gbps")` would otherwise file
them under Bandwidth. The divide/sqrt rows are GFLOPS counting one op per lane
(far below the FMA rows by design). Atomics, branch-mispredict and
store-forward report ns per op — cost probes, not throughput peaks.

Two tests were considered and deliberately dropped: **core-to-core latency**
(macOS has no hard pinning, so pair attribution is unreliable; the contended
atomics row covers fabric serialization instead) and **taken-branch throughput**
(needs a JIT'd jump chain; portable C++ only measures loop overhead). A
carry-less-multiply (PMULL/PCLMUL) crypto row was removed because its "GB/s of
multiplied operands" was a made-up convention that dwarfed the real bandwidth
rows — if it returns, frame it as GHASH GB/s or Gops.

## When You Change This Directory

- If you add a new benchmark → add it to the appropriate file, the `runAll()`
  dispatch, the `CpuPeak` interface (`include/cpu/cpu_peak.h`), `CMakeLists.txt`,
  and this file. New CPU-specific tests also need a `Benchmark` enum value +
  `categoryOf()` entry in `include/common/benchmark_enums.h` and a flag in
  `src/common/options.cpp`.
- If you add a new ISA capability gate → set it in `cpu_device.cpp::detectIsa()`
  and document it under `cpu_device_info_t`.
- If you validate a codegen-only path on real silicon → move it out of the
  Validation status table.
