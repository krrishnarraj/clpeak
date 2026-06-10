#ifdef ENABLE_CPU

#include "cpu_kernels.h"

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
      cpuidex(7, 1, r);
      f.avx512bf16 = (r[0] >> 5) & 1;
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
#endif
#endif // __aarch64__ || _M_ARM64
#endif
  return f;
}

#if defined(__linux__) && defined(CLPEAK_X86)
static bool amxPermOk()
{
  // ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18. Process-wide.
  static bool ok = (syscall(SYS_arch_prctl, 0x1023, 18) == 0);
  return ok;
}
#else
static bool amxPermOk() { return false; }
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
  if (s->readsum)     d.readsum = s->readsum;
}

} // anonymous namespace

// ---- TU accessors (defined in each compiled cpu_kernels_tu.cpp) ------------
#define DECL_TU(tag) extern "C" const CpuKernelTable *clpeak_table_##tag();
#if CLPEAK_NATIVE_BUILD
DECL_TU(native)
#else
#if CLPEAK_TU_generic
DECL_TU(generic)
#endif
#if CLPEAK_TU_sse42
DECL_TU(sse42)
#endif
#if CLPEAK_TU_avx2
DECL_TU(avx2)
#endif
#if CLPEAK_TU_avx512
DECL_TU(avx512)
#endif
#if CLPEAK_TU_avx512vnni
DECL_TU(avx512vnni)
#endif
#if CLPEAK_TU_avx512bf16
DECL_TU(avx512bf16)
#endif
#if CLPEAK_TU_avx512fp16
DECL_TU(avx512fp16)
#endif
#if CLPEAK_TU_amx
DECL_TU(amx)
#endif
#if CLPEAK_TU_neon
DECL_TU(neon)
#endif
#if CLPEAK_TU_fp16
DECL_TU(fp16)
#endif
#if CLPEAK_TU_fp16fml
DECL_TU(fp16fml)
#endif
#if CLPEAK_TU_dotprod
DECL_TU(dotprod)
#endif
#if CLPEAK_TU_bf16
DECL_TU(bf16)
#endif
#if CLPEAK_TU_i8mm
DECL_TU(i8mm)
#endif
#endif // CLPEAK_NATIVE_BUILD

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
  if (f.neon)    return "NEON";
  return "scalar";
}

const CpuKernelTable &kernels()
{
  static CpuKernelTable sel = [] {
    CpuKernelTable t{};
    const CpuFeatures &f = cpuFeatures();
    (void)f;
#if CLPEAK_NATIVE_BUILD
    merge(t, clpeak_table_native());   // native build: trust build==run host
#else
    // Merge supported TUs from baseline up, so the widest variant wins per kernel.
#if CLPEAK_TU_generic
    merge(t, clpeak_table_generic());     // ungated floor (SSE2 x86 / NEON arm / scalar)
#endif
#if CLPEAK_TU_sse42
    if (f.sse42) merge(t, clpeak_table_sse42());
#endif
#if CLPEAK_TU_neon
    if (f.neon) merge(t, clpeak_table_neon());
#endif
#if CLPEAK_TU_avx2
    if (f.avx2 && f.fma) merge(t, clpeak_table_avx2());
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
#endif // CLPEAK_NATIVE_BUILD
    t.isaName = isaName();
    return t;
  }();
  return sel;
}

} // namespace clpeak_cpu

#endif // ENABLE_CPU
