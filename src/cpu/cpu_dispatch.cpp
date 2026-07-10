#ifdef ENABLE_CPU

#include "cpu_kernels.h"
#include "cpu_tu_registry.h"

#include <cstdint>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CLPEAK_X86 1
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#elif defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64)
#define CLPEAK_ARM 1
#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#endif

#if defined(__linux__) && defined(CLPEAK_X86)
#include <unistd.h>
#include <sys/syscall.h>
#elif defined(_WIN32) && defined(CLPEAK_X86)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace clpeak_cpu {
namespace {

// ---- x86 CPUID / XGETBV ----------------------------------------------------
#if defined(CLPEAK_X86)
static inline void cpuidex(uint32_t leaf, uint32_t sub, uint32_t r[4])
{
#if defined(_MSC_VER)
  int t[4]; __cpuidex(t, (int)leaf, (int)sub);
  r[0] = t[0]; r[1] = t[1]; r[2] = t[2]; r[3] = t[3];
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}
static inline uint64_t xgetbv0()
{
#if defined(_MSC_VER)
  return _xgetbv(0);
#else
  uint32_t a, d;
  __asm__ volatile("xgetbv" : "=a"(a), "=d"(d) : "c"(0));
  return ((uint64_t)d << 32) | a;
#endif
}
#endif

static CpuFeatures detect()
{
  CpuFeatures f{};
#if defined(CLPEAK_X86)
  uint32_t r[4];
  cpuidex(0, 0, r);
  uint32_t maxLeaf = r[0];
  cpuidex(1, 0, r);
  bool osxsave = (r[2] >> 27) & 1;
  bool avxcpu  = (r[2] >> 28) & 1;
  f.sse42 = (r[2] >> 20) & 1;
  bool fma = (r[2] >> 12) & 1;
  // Crypto (SSE-class, no OS XSTATE dependency): leaf 1 ECX bit 25 = AES-NI.
  // The CRC32 instruction is part of SSE4.2.
  f.aes   = (r[2] >> 25) & 1;
  f.crc32 = f.sse42;

  // OS must have enabled SSE+AVX (XCR0 bits 1,2) and AVX-512 (bits 5,6,7).
  bool osAvx = false, osAvx512 = false;
  if (osxsave)
  {
    uint64_t xcr0 = xgetbv0();
    osAvx    = (xcr0 & 0x6) == 0x6;
    osAvx512 = osAvx && ((xcr0 & 0xE0) == 0xE0);
  }

  if (maxLeaf >= 7)
  {
    cpuidex(7, 0, r);
    uint32_t ebx = r[1], ecx = r[2], edx = r[3];
    bool avx2 = (ebx >> 5) & 1;
    f.avx2 = avx2 && osAvx && avxcpu;
    f.fma  = fma && osAvx && avxcpu;
    // Crypto: leaf 7 sub-leaf 0 EBX[29] = SHA-NI; ECX[9] = VAES (the 512-bit
    // form additionally needs the AVX-512 OS grant -- checked via f.avx512f
    // at the menu push, so gate it on osAvx512 here).
    f.sha256 = (ebx >> 29) & 1;
    f.vaes   = ((ecx >> 9) & 1) && osAvx512;
    // AMX-FP8 is enumerated in leaf 7 sub-leaf 0 ECX bit 3 (unlike the other new
    // AMX dtypes, which are in sub-leaf 1); execution is additionally gated by
    // amx_tile + the XTILEDATA XSTATE grant, so a stray bit can't SIGILL.
    f.amx_fp8 = (ecx >> 3) & 1;
    // Leaf 7 sub-leaf 1 (bit positions per Intel ISA ref / klauspost-cpuid):
    //   EAX[4]=AVX-VNNI, EAX[5]=AVX512-BF16, EAX[21]=AMX-FP16;
    //   EDX[4]=AVX-VNNI-INT8, EDX[7]=AMX-TF32, EDX[19]=AVX10.
    // AVX-VNNI / AVX-VNNI-INT8 are 256-bit VEX (OS AVX state only, no AVX-512).
    {
      uint32_t r1[4];
      cpuidex(7, 1, r1);
      uint32_t eax1 = r1[0], edx1 = r1[3];
      f.avxvnni      = ((eax1 >> 4) & 1) && osAvx && avxcpu;
      f.avxvnniint8  = ((edx1 >> 4) & 1) && osAvx && avxcpu;
      f.avxvnniint16 = ((edx1 >> 10) & 1) && osAvx && avxcpu;  // EDX[10] (Intel ISA ref / klauspost-cpuid)
      f.amx_fp16    = (eax1 >> 21) & 1;
      f.amx_tf32    = (edx1 >> 7) & 1;
      f.avx10       = (edx1 >> 19) & 1;
      // Leaf 7 sub-leaf 1 EAX[0] = SHA512 (EVEX; Arrow/Lunar Lake).  Detection
      // only today: no x86 SHA-512 kernel yet, so nothing consumes it -- wired
      // so the future kernel only needs the TU + menu push.
      f.sha512      = ((eax1 >> 0) & 1) && osAvx && avxcpu;
      if (osAvx512) f.avx512bf16 = (eax1 >> 5) & 1;
      // AVX10.2 512-bit: leaf 0x24 EBX low byte = version (>=2), bit 18 = 512-bit
      // support.  512-bit AVX10 uses ZMM state, so require the AVX-512 OS grant.
      if (f.avx10 && maxLeaf >= 0x24 && osAvx512)
      {
        uint32_t r24[4];
        cpuidex(0x24, 0, r24);
        uint32_t ver = r24[1] & 0xFF;
        f.avx10_2_512 = (ver >= 2) && ((r24[1] >> 18) & 1);
      }
    }
    if (osAvx512)
    {
      f.avx512f  = (ebx >> 16) & 1;
      f.avx512dq = (ebx >> 17) & 1;
      f.avx512bw = (ebx >> 30) & 1;
      f.avx512vl = (ebx >> 31) & 1;
      f.avx512vnni = (ecx >> 11) & 1;
      f.avx512fp16 = (edx >> 23) & 1;
      f.amx_bf16 = (edx >> 22) & 1;
      f.amx_tile = (edx >> 24) & 1;
      f.amx_int8 = (edx >> 25) & 1;
    }
  }
#elif defined(CLPEAK_ARM)
  // Only AArch64 has the NEON kernels; 32-bit ARMv7 uses the scalar generic TU
  // (and its HWCAP bit layout differs from the AArch64 bits below anyway), so
  // leave every feature false there and let isaName() report "scalar".
#if defined(__aarch64__) || defined(_M_ARM64)
  f.neon = true;   // mandatory on aarch64
#if defined(__linux__) || defined(__ANDROID__)
  unsigned long hw = getauxval(AT_HWCAP);
  unsigned long hw2 = getauxval(AT_HWCAP2);
  // bit positions from <asm/hwcap.h>
  f.fp16    = hw & (1UL << 10);  // ASIMDHP
  f.dotprod = hw & (1UL << 20);  // ASIMDDP
  f.fp16fml = hw & (1UL << 23);  // ASIMDFHM
  f.bf16    = hw2 & (1UL << 14); // BF16
  f.i8mm    = hw2 & (1UL << 13); // I8MM
  f.sve     = hw & (1UL << 22);  // HWCAP_SVE
  f.sve2    = hw2 & (1UL << 1);  // HWCAP2_SVE2
  f.svei8mm = hw2 & (1UL << 9);  // HWCAP2_SVEI8MM
  f.svebf16 = hw2 & (1UL << 12); // HWCAP2_SVEBF16
  // SME: the kernel only sets these once it supports the ZA/streaming state,
  // so a set bit implies both the hardware and the OS context handling.
  f.sme       = hw2 & (1UL << 23); // HWCAP2_SME
  f.smeF64F64 = hw2 & (1UL << 25); // HWCAP2_SME_F64F64
  f.sme2      = hw2 & (1UL << 37); // HWCAP2_SME2
  f.fp8dot4   = hw2 & (1UL << 53); // HWCAP2_F8DP4
  // Crypto extensions (all in the base HWCAP word)
  f.aes    = hw & (1UL << 3);    // HWCAP_AES
  f.sha256 = hw & (1UL << 6);    // HWCAP_SHA2
  f.crc32  = hw & (1UL << 7);    // HWCAP_CRC32
  f.sha512 = hw & (1UL << 21);   // HWCAP_SHA512
#elif defined(__APPLE__)
  auto sc = [](const char *n) {
    int v = 0; size_t s = sizeof(v);
    return sysctlbyname(n, &v, &s, nullptr, 0) == 0 && v != 0;
  };
  f.fp16    = sc("hw.optional.arm.FEAT_FP16");
  f.dotprod = sc("hw.optional.arm.FEAT_DotProd");
  f.fp16fml = sc("hw.optional.arm.FEAT_FHM");
  f.bf16    = sc("hw.optional.arm.FEAT_BF16");
  f.i8mm    = sc("hw.optional.arm.FEAT_I8MM");
  // SME (M4+; replaced Apple's private AMX).  No non-streaming SVE exists on
  // Apple Silicon, so f.sve/sve2 stay false and only the SME paths light up.
  f.sme       = sc("hw.optional.arm.FEAT_SME");
  f.sme2      = sc("hw.optional.arm.FEAT_SME2");
  f.smeF64F64 = sc("hw.optional.arm.FEAT_SME_F64F64");
  f.fp8dot4   = sc("hw.optional.arm.FEAT_FP8DOT4");  // none ship it yet; future-proof
  // Crypto extensions (present on every Apple Silicon part; keys verified on M1)
  f.aes    = sc("hw.optional.arm.FEAT_AES");
  f.sha256 = sc("hw.optional.arm.FEAT_SHA256");
  f.sha512 = sc("hw.optional.arm.FEAT_SHA512");
  f.crc32  = sc("hw.optional.arm.FEAT_CRC32");
#elif defined(_WIN32)
  // Windows has no IsProcessorFeaturePresent() constant for most of these,
  // but the kernel exports the (sanitised) AArch64 ID registers as REG_QWORD
  // values "CP <enc>" under CentralProcessor\0, enc = 0x4000 | CRm<<3 | op2
  // (op0=3,op1=0,CRn=0):  CP 4000 = MIDR_EL1, CP 4020 = ID_AA64PFR0_EL1,
  // CP 4030 = ID_AA64ISAR0_EL1, CP 4031 = ID_AA64ISAR1_EL1.
  auto idreg = [](const char *value) -> uint64_t {
    uint64_t v = 0;
    DWORD sz = sizeof(v);
    if (RegGetValueA(HKEY_LOCAL_MACHINE,
                     "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     value, RRF_RT_REG_QWORD, nullptr, &v, &sz) != ERROR_SUCCESS)
      return 0;
    return v;
  };
  const uint64_t pfr0  = idreg("CP 4020");
  const uint64_t isar0 = idreg("CP 4030");
  const uint64_t isar1 = idreg("CP 4031");
  f.fp16    = ((pfr0 >> 16) & 0xF) == 1;   // PFR0.FP: 1 = FP + FEAT_FP16 (0xF = no FP)
  f.dotprod = ((isar0 >> 44) & 0xF) >= 1;  // ISAR0.DP   (FEAT_DotProd)
  f.fp16fml = ((isar0 >> 48) & 0xF) >= 1;  // ISAR0.FHM  (FEAT_FHM)
  f.bf16    = ((isar1 >> 44) & 0xF) >= 1;  // ISAR1.BF16 (FEAT_BF16)
  f.i8mm    = ((isar1 >> 52) & 0xF) >= 1;  // ISAR1.I8MM (FEAT_I8MM)
  // Crypto: ISAR0.AES (>=1), ISAR0.SHA2 (1 = SHA-256, 2 = +SHA-512),
  // ISAR0.CRC32.
  f.aes     = ((isar0 >> 4) & 0xF) >= 1;
  f.sha256  = ((isar0 >> 12) & 0xF) >= 1;
  f.sha512  = ((isar0 >> 12) & 0xF) >= 2;
  f.crc32   = ((isar0 >> 16) & 0xF) >= 1;
  // No SVE/SME on Windows: clang's MSVC C++ ABI can't mangle the SVE/SME
  // sizeless types, so <arm_sve.h>/<arm_sme.h> won't compile and no SVE/SME TU
  // is built (see CMakeLists.txt).  Detecting them here would only make the
  // device header claim features with no test rows, so leave f.sve/sve2/sme
  // false until that toolchain gap closes (then read ID_AA64ZFR0/SMFR0 from
  // the registry, "CP 4024"/"CP 4045", to match).
#endif
#endif // __aarch64__ || _M_ARM64
#endif
  return f;
}

// Only referenced under CLPEAK_TU_amx (maybe_unused: avoids -Wunused-function
// on the platforms where that TU isn't built).  AMX tile XSTATE (component 18)
// is disabled by default and must be granted by the OS on first use; the grant
// is process-wide, so request it once and cache the result.
#if defined(__linux__) && defined(CLPEAK_X86)
[[maybe_unused]] static bool amxPermOk()
{
  // ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18.
  static bool ok = (syscall(SYS_arch_prctl, 0x1023, 18) == 0);
  return ok;
}
#elif defined(_WIN32) && defined(CLPEAK_X86)
[[maybe_unused]] static bool amxPermOk()
{
  // Windows 11 / Server 2022 equivalent of the Linux arch_prctl grant:
  // EnableProcessOptionalXStateFeatures(XSTATE_MASK_AMX_TILE_DATA).  Resolve it
  // dynamically from kernel32 so the binary still loads on Win10 (where the
  // export is absent) and on a non-AMX host -> amxPermOk() is simply false and
  // the matrix tests emit the Unsupported row.
  static bool ok = [] {
    // XSTATE component 18 == XFEATURE_XTILEDATA; mask bit (winnt.h may not
    // define XSTATE_MASK_AMX_TILE_DATA on older SDKs, so spell it out).
    const DWORD64 amxTileData = 1ULL << 18;
    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    if (!k32) return false;
    using EnableFn = BOOL(WINAPI *)(DWORD64);
    auto enableFn = reinterpret_cast<EnableFn>(
        GetProcAddress(k32, "EnableProcessOptionalXStateFeatures"));
    if (!enableFn) return false;
    return enableFn(amxTileData) != 0;
  }();
  return ok;
}
#else
[[maybe_unused]] static bool amxPermOk() { return false; }
#endif

static void merge(CpuKernelTable &d, const CpuKernelTable *s)
{
  if (!s) return;
  if (s->fp32.fn)     d.fp32 = s->fp32;
  if (s->fp64.fn)     d.fp64 = s->fp64;
  if (s->int32.fn)    d.int32 = s->int32;
  if (s->fp16.fn)     d.fp16 = s->fp16;
  if (s->bf16.fn)     d.bf16 = s->bf16;
  if (s->mp.fn)       d.mp = s->mp;
  if (s->int8dp.fn)   d.int8dp = s->int8dp;
  if (s->mat_int8.fn) d.mat_int8 = s->mat_int8;
  if (s->mat_fp.fn)   d.mat_fp = s->mat_fp;
  if (s->mat_fp16.fn) d.mat_fp16 = s->mat_fp16;
  if (s->mat_tf32.fn) d.mat_tf32 = s->mat_tf32;
  if (s->mat_fp8.fn)  d.mat_fp8 = s->mat_fp8;
  if (s->bf16fma.fn)  d.bf16fma = s->bf16fma;
  if (s->int16dp.fn)  d.int16dp = s->int16dp;
  if (s->fp8dp.fn)    d.fp8dp = s->fp8dp;
  if (s->mat_fp32.fn) d.mat_fp32 = s->mat_fp32;
  if (s->mat_fp64.fn) d.mat_fp64 = s->mat_fp64;
  if (s->ssve_fp32.fn) d.ssve_fp32 = s->ssve_fp32;
  if (s->ssve_fp64.fn) d.ssve_fp64 = s->ssve_fp64;
  if (s->aes.fn)      d.aes = s->aes;
  if (s->sha256.fn)   d.sha256 = s->sha256;
  if (s->sha512.fn)   d.sha512 = s->sha512;
  if (s->crc32c.fn)   d.crc32c = s->crc32c;
  if (s->div32.fn)    d.div32 = s->div32;
  if (s->div64.fn)    d.div64 = s->div64;
  if (s->sqrt32.fn)   d.sqrt32 = s->sqrt32;
  if (s->sqrt64.fn)   d.sqrt64 = s->sqrt64;
  if (s->intdiv.fn)   d.intdiv = s->intdiv;
  if (s->readsum)     d.readsum = s->readsum;
  if (s->sveVLBytes)  d.sveVLBytes = s->sveVLBytes;
  if (s->smeSVLBytes) d.smeSVLBytes = s->smeSVLBytes;
}

} // anonymous namespace

// ---- TU accessors (defined in each compiled cpu_kernels_tu.cpp) ------------
// Declared unconditionally from the TU registry: a declaration for a TU CMake
// didn't build is harmless (the symbol is never referenced -- every call site in
// kernels()/kernelMenu() is guarded by #if CLPEAK_TU_<tag>).
#define DECL_TU(tag) extern "C" const CpuKernelTable *clpeak_table_##tag();
CLPEAK_TU_REGISTRY(DECL_TU)
#undef DECL_TU

const CpuFeatures &cpuFeatures()
{
  static CpuFeatures f = detect();
  return f;
}

const char *isaName()
{
  const CpuFeatures &f = cpuFeatures();
  if (f.avx512f) return "AVX-512";
  if (f.avx2)    return "AVX2+FMA";
  if (f.sse42)   return "SSE4.2";
  if (f.sve2)    return "SVE2";
  if (f.sve)     return "SVE";
  if (f.neon)    return "NEON";
  return "scalar";
}

const CpuKernelTable &kernels()
{
  static CpuKernelTable sel = [] {
    CpuKernelTable t{};
    const CpuFeatures &f = cpuFeatures();
    (void)f;
    // Merge supported TUs from baseline up, so the widest variant wins per kernel.
#if CLPEAK_TU_generic
    merge(t, clpeak_table_generic());     // ungated floor (SSE2 x86 / NEON arm / scalar)
#endif
#if CLPEAK_TU_sse42
    if (f.sse42) merge(t, clpeak_table_sse42());
#endif
#if CLPEAK_TU_avx2
    if (f.avx2 && f.fma) merge(t, clpeak_table_avx2());
#endif
#if CLPEAK_TU_avxvnni
    // Before avx512 so a wider AVX-512 base kernel still wins per-slot; the
    // avxvnni TU's real contribution is its 256-bit int8dp (Alder Lake+ etc.).
    if (f.avxvnni && f.avx2 && f.fma) merge(t, clpeak_table_avxvnni());
#endif
#if CLPEAK_TU_avxvnniint8
    if (f.avxvnniint8 && f.avx2 && f.fma) merge(t, clpeak_table_avxvnniint8());
#endif
#if CLPEAK_TU_avxvnniint16
    if (f.avxvnniint16 && f.avx2 && f.fma) merge(t, clpeak_table_avxvnniint16());
#endif
#if CLPEAK_TU_avx512
    if (f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq) merge(t, clpeak_table_avx512());
#endif
#if CLPEAK_TU_avx512vnni
    if (f.avx512f && f.avx512bw && f.avx512vl && f.avx512vnni) merge(t, clpeak_table_avx512vnni());
#endif
#if CLPEAK_TU_avx512bf16
    if (f.avx512f && f.avx512bw && f.avx512vl && f.avx512bf16) merge(t, clpeak_table_avx512bf16());
#endif
#if CLPEAK_TU_avx512fp16
    if (f.avx512f && f.avx512bw && f.avx512vl && f.avx512fp16) merge(t, clpeak_table_avx512fp16());
#endif
#if CLPEAK_TU_amx
    if (f.amx_tile && f.amx_int8 && f.amx_bf16 && amxPermOk()) merge(t, clpeak_table_amx());
#endif
#if CLPEAK_TU_amxfp16
    if (f.amx_tile && f.amx_fp16 && amxPermOk()) merge(t, clpeak_table_amxfp16());
#endif
#if CLPEAK_TU_amxtf32
    if (f.amx_tile && f.amx_tf32 && amxPermOk()) merge(t, clpeak_table_amxtf32());
#endif
#if CLPEAK_TU_amxfp8
    if (f.amx_tile && f.amx_fp8 && amxPermOk()) merge(t, clpeak_table_amxfp8());
#endif
#if CLPEAK_TU_avx10bf16
    if (f.avx10_2_512) merge(t, clpeak_table_avx10bf16());
#endif
#if CLPEAK_TU_fp16
    if (f.fp16) merge(t, clpeak_table_fp16());
#endif
#if CLPEAK_TU_fp16fml
    if (f.fp16fml) merge(t, clpeak_table_fp16fml());
#endif
#if CLPEAK_TU_dotprod
    if (f.dotprod) merge(t, clpeak_table_dotprod());
#endif
#if CLPEAK_TU_bf16
    if (f.bf16) merge(t, clpeak_table_bf16());
#endif
#if CLPEAK_TU_i8mm
    if (f.i8mm) merge(t, clpeak_table_i8mm());
#endif
#if CLPEAK_TU_sve
    if (f.sve) merge(t, clpeak_table_sve());
#endif
#if CLPEAK_TU_svebf16
    if (f.sve && f.svebf16) merge(t, clpeak_table_svebf16());
#endif
#if CLPEAK_TU_svei8mm
    if (f.sve && f.svei8mm) merge(t, clpeak_table_svei8mm());
#endif
    // The fp8dot/svefp8dot and SME TUs are deliberately NOT merged here.
    // kernels() only feeds the bandwidth path (readsum) and those TUs add
    // nothing there -- and their -march exceeds the platform floor, so their
    // base-kernel codegen must never be selectable via a narrower feature gate
    // (e.g. the SME TU on Apple must not contribute anything non-streaming).
    // They are menu-push-only: kernelMenu() reads their specific slots.
    t.isaName = isaName();
    return t;
  }();
  return sel;
}

// Active SVE vector length in bytes (svcntb() captured by the SVE TU's table),
// 0 when the running host has no SVE.  Used for the "SVE2 (VL=256b)" header.
int sveVLBytes()
{
  return kernels().sveVLBytes;
}

// Active SME streaming vector length in bytes (svcntsb() captured by the SME
// TU's table).  Read directly from that TU -- the SME TUs are never merged
// into kernels() (see above).  0 when the host has no SME or no SME TU built.
int smeSVLBytes()
{
#if CLPEAK_TU_sme
  if (cpuFeatures().sme)
    return clpeak_table_sme()->smeSVLBytes;
#endif
  return 0;
}

// ---- Run-all menu: every supported variant, baseline-first -----------------
// Unlike kernels() (which merges down to the single widest variant per kernel),
// this exposes ALL supported ISA variants so the compute tests can report each
// one separately.  Labels are the canonical isaName() style.  The "collapse
// identical SSE float" rule is encoded by only pushing int32 (not fp32/fp64)
// from the sse42 TU: SSE4.2 fp32/fp64 codegen is identical to the SSE2 floor.
const CpuKernelMenu &kernelMenu()
{
  static CpuKernelMenu menu = [] {
    CpuKernelMenu m{};
    const CpuFeatures &f = cpuFeatures();
    (void)f;

    auto add = [](std::vector<IsaVariant> &vec, const ChainVariant &v, const char *isa) {
      if (v.fn) vec.push_back({v, isa});
    };
    // Push the base dtypes (fp32/fp64/int32) a tier TU provides, all one label.
    auto addBase = [&](const CpuKernelTable *t, const char *isa) {
      add(m.fp32, t->fp32, isa);
      add(m.fp64, t->fp64, isa);
      add(m.int32, t->int32, isa);
    };
    // Divide/sqrt ride with the same tier TUs (they go through cpu_simd.h /
    // the SVE kernels).  Not part of addBase: the sse42 TU pushes int32 only,
    // and its div/sqrt codegen is identical to the SSE2 floor's.
    auto addDivSqrt = [&](const CpuKernelTable *t, const char *isa) {
      add(m.div32, t->div32, isa);
      add(m.div64, t->div64, isa);
      add(m.sqrt32, t->sqrt32, isa);
      add(m.sqrt64, t->sqrt64, isa);
    };

    // ---- x86 tier TUs (base dtypes) ----
#if CLPEAK_TU_generic
    // The ungated floor: SSE2 on x86, NEON on aarch64, scalar elsewhere.
#if defined(CLPEAK_X86)
    const char *genIsa = "SSE2";
#elif defined(CLPEAK_ARM) && (defined(__aarch64__) || defined(_M_ARM64))
    const char *genIsa = "NEON";
#else
    const char *genIsa = "scalar";
#endif
    {
      const CpuKernelTable *t = clpeak_table_generic();
      addBase(t, genIsa);
      addDivSqrt(t, genIsa);
      // On Apple the generic (apple-m1) floor also carries the advanced NEON
      // dtypes; push whatever it actually provides, labeled by feature.
      if (f.fp16)    add(m.fp16, t->fp16, "NEON FP16");
      if (f.dotprod) add(m.int8dp, t->int8dp, "NEON DotProd");
      if (f.fp16fml) add(m.mp, t->mp, "NEON FP16FML");
    }
#endif
#if CLPEAK_TU_sse42
    if (f.sse42)
    {
      add(m.int32, clpeak_table_sse42()->int32, "SSE4.2");  // fp32/fp64 == SSE2
      // The CRC32 instruction is part of SSE4.2, so the CRC32-C kernel rides
      // in this TU rather than a dedicated one (the ARM analogue is the crc TU).
      if (f.crc32) add(m.crc32c, clpeak_table_sse42()->crc32c, "SSE4.2");
    }
#endif
#if CLPEAK_TU_avx2
    if (f.avx2 && f.fma)
    {
      addBase(clpeak_table_avx2(), "AVX2+FMA");
      addDivSqrt(clpeak_table_avx2(), "AVX2");   // no FMA in a div/sqrt chain
    }
#endif
#if CLPEAK_TU_avx512
    if (f.avx512f && f.avx512bw && f.avx512vl && f.avx512dq)
    {
      addBase(clpeak_table_avx512(), "AVX-512");
      addDivSqrt(clpeak_table_avx512(), "AVX-512");
    }
#endif
    // ---- x86 feature TUs (advanced dtypes only) ----
#if CLPEAK_TU_avxvnni
    // 256-bit AVX-VNNI int8 dot -- the client-x86 int8 path (Alder Lake+, Zen 5,
    // Sierra Forest have no AVX-512).  Baseline-first: pushed before the 512-bit
    // VNNI variant below.  The VNNI TUs also carry the int16 dot (VPDPWSSD has
    // shipped with VNNI from the start).
    if (f.avxvnni && f.avx2 && f.fma)
    {
      const CpuKernelTable *t = clpeak_table_avxvnni();
      add(m.int8dp, t->int8dp, "AVX-VNNI");
      add(m.int16dp, t->int16dp, "AVX-VNNI");
    }
#endif
#if CLPEAK_TU_avxvnniint8
    if (f.avxvnniint8 && f.avx2 && f.fma)
      add(m.int8dp, clpeak_table_avxvnniint8()->int8dp, "AVX-VNNI-INT8");
#endif
#if CLPEAK_TU_avxvnniint16
    // AVX-VNNI-INT16 (Diamond Rapids, Nova Lake): the mixed-sign int16 dot
    // (VPDPWSUD) as its own ISA row next to the classic VNNI VPDPWSSD.
    if (f.avxvnniint16 && f.avx2 && f.fma)
      add(m.int16dp, clpeak_table_avxvnniint16()->int16dp, "AVX-VNNI-INT16");
#endif
#if CLPEAK_TU_avx512vnni
    if (f.avx512f && f.avx512bw && f.avx512vl && f.avx512vnni)
    {
      const CpuKernelTable *t = clpeak_table_avx512vnni();
      add(m.int8dp, t->int8dp, "AVX-512 VNNI");
      add(m.int16dp, t->int16dp, "AVX-512 VNNI");
    }
#endif
#if CLPEAK_TU_avx512bf16
    if (f.avx512f && f.avx512bw && f.avx512vl && f.avx512bf16)
      add(m.bf16, clpeak_table_avx512bf16()->bf16, "AVX-512 BF16");
#endif
#if CLPEAK_TU_avx512fp16
    if (f.avx512f && f.avx512bw && f.avx512vl && f.avx512fp16)
      add(m.fp16, clpeak_table_avx512fp16()->fp16, "AVX-512 FP16");
#endif
#if CLPEAK_TU_amx
    if (f.amx_tile && f.amx_int8 && f.amx_bf16 && amxPermOk())
    {
      const CpuKernelTable *t = clpeak_table_amx();
      add(m.mat_int8, t->mat_int8, "AMX");
      add(m.mat_fp, t->mat_fp, "AMX");
    }
#endif
#if CLPEAK_TU_amxfp16
    if (f.amx_tile && f.amx_fp16 && amxPermOk())
      add(m.mat_fp16, clpeak_table_amxfp16()->mat_fp16, "AMX FP16");
#endif
#if CLPEAK_TU_amxtf32
    if (f.amx_tile && f.amx_tf32 && amxPermOk())
      add(m.mat_tf32, clpeak_table_amxtf32()->mat_tf32, "AMX TF32");
#endif
#if CLPEAK_TU_amxfp8
    if (f.amx_tile && f.amx_fp8 && amxPermOk())
      add(m.mat_fp8, clpeak_table_amxfp8()->mat_fp8, "AMX FP8");
#endif
#if CLPEAK_TU_avx10bf16
    if (f.avx10_2_512)
      add(m.bf16fma, clpeak_table_avx10bf16()->bf16fma, "AVX10.2");
#endif
    // ---- Crypto TUs (x86: AES-NI, VAES-512, SHA-NI;
    //      ARM: FEAT_AES, FEAT_SHA256, FEAT_SHA512, FEAT_CRC32) ----
#if CLPEAK_TU_aes
#if defined(CLPEAK_X86)
    if (f.aes) add(m.aes, clpeak_table_aes()->aes, "AES-NI");
#else
    if (f.aes) add(m.aes, clpeak_table_aes()->aes, "NEON AES");
#endif
#endif
#if CLPEAK_TU_vaes
    // The 512-bit EVEX form needs the AVX-512 OS grant (f.avx512f includes it).
    if (f.vaes && f.avx512f)
      add(m.aes, clpeak_table_vaes()->aes, "VAES-512");
#endif
#if CLPEAK_TU_sha
#if defined(CLPEAK_X86)
    if (f.sha256) add(m.sha256, clpeak_table_sha()->sha256, "SHA-NI");
#else
    if (f.sha256) add(m.sha256, clpeak_table_sha()->sha256, "NEON SHA2");
#endif
#endif
#if CLPEAK_TU_sha512
    if (f.sha512) add(m.sha512, clpeak_table_sha512()->sha512, "NEON SHA512");
#endif
#if CLPEAK_TU_crc
    if (f.crc32) add(m.crc32c, clpeak_table_crc()->crc32c, "ARMv8 CRC32");
#endif
    // ---- ARM feature TUs (advanced dtypes only; base == generic NEON) ----
#if CLPEAK_TU_fp16
    if (f.fp16) add(m.fp16, clpeak_table_fp16()->fp16, "NEON FP16");
#endif
#if CLPEAK_TU_fp16fml
    if (f.fp16fml) add(m.mp, clpeak_table_fp16fml()->mp, "NEON FP16FML");
#endif
#if CLPEAK_TU_dotprod
    if (f.dotprod) add(m.int8dp, clpeak_table_dotprod()->int8dp, "NEON DotProd");
#endif
#if CLPEAK_TU_bf16
    if (f.bf16)
    {
      const CpuKernelTable *t = clpeak_table_bf16();
      add(m.bf16, t->bf16, "NEON BF16");
      // Matrix engine: tag with the matrix instruction itself (BFMMLA, part of
      // FEAT_BF16) rather than the feature name, paralleling x86 "AMX".
      add(m.mat_fp, t->mat_fp, "BFMMLA");
    }
#endif
#if CLPEAK_TU_i8mm
    if (f.i8mm) add(m.mat_int8, clpeak_table_i8mm()->mat_int8, "SMMLA");  // FEAT_I8MM matrix instr
#endif
#if CLPEAK_TU_fp8dot
    // Native fp8 4-way dot -> fp32 (FEAT_FP8DOT4; NVIDIA Vera first).
    if (f.fp8dot4) add(m.fp8dp, clpeak_table_fp8dot()->fp8dp, "NEON FP8");
#endif
    // ---- ARM SVE (vector-length-agnostic; own base compute + advanced dtypes) ----
    // Base compute labeled SVE2 vs SVE by the runtime feature; the matrix/dot
    // feature TUs are tagged by their instruction (parallel to NEON BFMMLA/SMMLA)
    // with an "SVE " prefix so their rows stay distinct from the NEON variants.
#if CLPEAK_TU_sve
    if (f.sve)
    {
      const char *sveLbl = f.sve2 ? "SVE2" : "SVE";
      const CpuKernelTable *t = clpeak_table_sve();
      add(m.fp32, t->fp32, sveLbl);
      add(m.fp64, t->fp64, sveLbl);
      add(m.int32, t->int32, sveLbl);
      add(m.int8dp, t->int8dp, sveLbl);
      addDivSqrt(t, sveLbl);
    }
#endif
#if CLPEAK_TU_svebf16
    if (f.sve && f.svebf16)
    {
      const CpuKernelTable *t = clpeak_table_svebf16();
      add(m.bf16, t->bf16, "SVE BF16");
      add(m.mat_fp, t->mat_fp, "SVE BFMMLA");
    }
#endif
#if CLPEAK_TU_svei8mm
    if (f.sve && f.svei8mm)
      add(m.mat_int8, clpeak_table_svei8mm()->mat_int8, "SVE SMMLA");
#endif
#if CLPEAK_TU_svefp8dot
    if (f.sve && f.sve2 && f.fp8dot4)
      add(m.fp8dp, clpeak_table_svefp8dot()->fp8dp, "SVE2 FP8");
#endif
    // ---- ARM SME (streaming; Apple M4+, Snapdragon X2 / 8 Elite Gen 5) ----
    // ZA outer products go to the matrix slots (SME is the ARM analogue of AMX,
    // labeled by the instruction like BFMMLA/SMMLA are), and the streaming-SVE
    // vector chains ride the base fp32/fp64 menus as an "SSVE" ISA variant.
    // Note the SME unit is shared per cluster on all current cores, so the MT
    // rows scale with cluster count, not core count.
#if CLPEAK_TU_sme
    if (f.sme)
    {
      const CpuKernelTable *t = clpeak_table_sme();
      add(m.fp32, t->ssve_fp32, "SSVE");
      add(m.fp64, t->ssve_fp64, "SSVE");
      add(m.mat_fp32, t->mat_fp32, "SME FMOPA");
      add(m.mat_fp, t->mat_fp, "SME BFMOPA");
      add(m.mat_fp16, t->mat_fp16, "SME FMOPA-FP16");
      add(m.mat_int8, t->mat_int8, "SME SMOPA");
    }
#endif
#if CLPEAK_TU_smef64
    if (f.sme && f.smeF64F64)
      add(m.mat_fp64, clpeak_table_smef64()->mat_fp64, "SME FMOPA-FP64");
#endif
    return m;
  }();
  return menu;
}

} // namespace clpeak_cpu

#endif // ENABLE_CPU
