#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>

#ifdef CLPEAK_ONEAPI_HAS_JOINT_MATRIX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#include <sycl/ext/oneapi/bfloat16.hpp>
#define CLPEAK_ONEAPI_JM_HAS_BF16 1
#endif
#endif

// XMX matrix engine peak — Intel's analog of rocWMMA / cuda WMMA / Vulkan
// coopMatrix / Metal simdgroup_matrix.  Tile shapes map to Intel XMX:
//   bf16/fp16 -> 8x16x16 (A,B 16-bit, fp32 accumulate)
//   tf32      -> 8x16x8  (A,B tf32,   fp32 accumulate)
//   int8      -> 8x16x32 (A,B int8,   int32 accumulate)
//
// Each sub-group runs Iters back-to-back MMA ops on its own accumulator;
// per-sub-group ops = M*N*K*2*Iters (multiply-add counted as 2).  One
// sub-group per work-group, so numBlocks == number of sub-groups.

#ifdef CLPEAK_ONEAPI_HAS_JOINT_MATRIX

namespace {

namespace syclex = sycl::ext::oneapi::experimental::matrix;

constexpr uint32_t JM_M = 8;
constexpr uint32_t JM_N = 16;
constexpr uint32_t JM_ITERS = 256;
constexpr int      JM_SG = 16;   // XMX sub-group (lane) width on Intel Xe

// Per-variant kernel-name tags (SYCL needs a unique type per parallel_for).
struct JmBf16Tag; struct JmFp16Tag; struct JmTf32Tag; struct JmInt8Tag;

static const char *mtName(syclex::matrix_type t)
{
  using mt = syclex::matrix_type;
  switch (t)
  {
    case mt::bf16:   return "bf16";
    case mt::fp16:   return "fp16";
    case mt::tf32:   return "tf32";
    case mt::fp32:   return "fp32";
    case mt::fp64:   return "fp64";
    case mt::sint8:  return "sint8";
    case mt::sint16: return "sint16";
    case mt::sint32: return "sint32";
    case mt::sint64: return "sint64";
    case mt::uint8:  return "uint8";
    case mt::uint16: return "uint16";
    case mt::uint32: return "uint32";
    case mt::uint64: return "uint64";
    default:         return "?";
  }
}

// Pull the device's joint_matrix combination table.  Sets `threw` so callers
// can distinguish "queried OK, none" from "couldn't query at all".
static std::vector<syclex::combination> queryCombos(const sycl::device &d, bool &threw)
{
  threw = false;
  try {
    return d.get_info<
      sycl::ext::oneapi::experimental::info::device::matrix_combinations>();
  } catch (const std::exception &e) {
    CLPEAK_VLOG("joint_matrix: matrix_combinations query threw: %s\n", e.what());
    threw = true;
    return {};
  }
}

// Verbose-only: print every (a/b/c/d type, M/N/K, max M/N/K) the device accepts.
// This is the ground truth for picking tile shapes on a new device.
static void dumpMatrixCombinations(const sycl::device &d)
{
  if (!::clpeak::verboseEnabled()) return;
  bool threw = false;
  auto combos = queryCombos(d, threw);
  if (threw) return;
  CLPEAK_VLOG("joint_matrix: device reports %zu matrix combination(s):\n",
              combos.size());
  for (const auto &c : combos)
    CLPEAK_VLOG("  a=%-5s b=%-5s c=%-5s d=%-5s  M=%zu N=%zu K=%zu  "
                "(max M=%zu N=%zu K=%zu)\n",
                mtName(c.atype), mtName(c.btype), mtName(c.ctype), mtName(c.dtype),
                c.msize, c.nsize, c.ksize,
                c.max_msize, c.max_nsize, c.max_ksize);
}

// Ask whether (atype,btype,ctype) at shape MxNxK is in the device table.
// Returns 1 supported, 0 unsupported (queried OK but absent, incl. empty table),
// -1 when the query itself threw (caller should attempt and let runKernel report).
static int jmComboSupport(const sycl::device &d,
                          syclex::matrix_type at, syclex::matrix_type bt,
                          syclex::matrix_type ct,
                          size_t M, size_t N, size_t K)
{
  bool threw = false;
  auto combos = queryCombos(d, threw);
  if (threw) return -1;
  for (const auto &c : combos) {
    if (c.atype != at || c.btype != bt || c.ctype != ct) continue;
    // Xe reports a flexible dim as 0 with a max_* bound; AMX-style backends
    // report a single fixed size.  Accept either encoding.
    const bool mok = (c.msize == 0) ? (M <= c.max_msize) : (M == c.msize);
    const bool nok = (c.nsize == 0) ? (N <= c.max_nsize) : (N == c.nsize);
    const bool kok = (c.ksize == 0) ? (K <= c.max_ksize) : (K == c.ksize);
    if (mok && nok && kok) return 1;
  }
  return 0;  // queried fine; shape/type absent (empty table => nothing supported)
}

// Run one matrix-engine variant.  ABt is the joint_matrix element type
// (e.g. bfloat16, sycl::half, precision::tf32, int8_t); FillT is the type
// used to fill A/B (float for tf32, otherwise == ABt); ACCt is the
// accumulator/output element type.
template <typename KernelName, typename ABt, typename FillT, typename ACCt, int K>
static float runJmVariant(OneapiPeak &peak, OneapiDevice &dev,
                          ACCt *outBuf, uint32_t numBlocks, uint32_t blockSize,
                          FillT abFill, ACCt cFill,
                          unsigned int targetTimeUs, unsigned int forced)
{
  const uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  auto submit = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<KernelName>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(JM_SG)]] {
          auto sg = it.get_sub_group();
          syclex::joint_matrix<sycl::sub_group, ABt,  syclex::use::a, JM_M, K, syclex::layout::row_major>       a;
          // Intel XMX requires the B operand in VNNI/packed layout; a row_major
          // B is rejected at launch on Xe-HPG (Arc/DG2).
          syclex::joint_matrix<sycl::sub_group, ABt,  syclex::use::b, K, JM_N, syclex::layout::ext_intel_packed> b;
          syclex::joint_matrix<sycl::sub_group, ACCt, syclex::use::accumulator, JM_M, JM_N> c;
          syclex::joint_matrix_fill(sg, a, abFill);
          syclex::joint_matrix_fill(sg, b, abFill);
          syclex::joint_matrix_fill(sg, c, cFill);
          #pragma unroll 1
          for (int i = 0; i < (int)JM_ITERS; i++)
            syclex::joint_matrix_mad(sg, c, a, b, c);

          ACCt *blockOut = outBuf + (size_t)it.get_group(0) * JM_M * JM_N;
          syclex::joint_matrix_store(sg, c,
            sycl::address_space_cast<sycl::access::address_space::global_space,
                                     sycl::access::decorated::no>(blockOut),
            JM_N, syclex::layout::row_major);
        });
    });
  };
  return peak.runKernel(dev, submit, targetTimeUs, forced);
}

static void emitJm(logger::TestScope &test, const char *metric, float us,
                   uint32_t numBlocks, uint32_t K)
{
  if (us <= 0.0f)
  {
    test.skip(metric, ResultStatus::Error, "kernel launch failed");
    return;
  }
  const double ops = (double)numBlocks * (double)JM_M * (double)JM_N *
                     (double)K * 2.0 * (double)JM_ITERS;
  test.emit(metric, (float)(ops * 1.0e6 / us / 1.0e12));
}

} // namespace

#endif // CLPEAK_ONEAPI_HAS_JOINT_MATRIX

int OneapiPeak::runJointMatrix(OneapiDevice &dev, benchmark_config_t &cfg, Category category)
{
  const bool isInt = (category == Category::IntCompute);
  auto test = currentDeviceScope->beginTest(
    {isInt ? "joint-matrix-int" : "joint-matrix-fp",
     isInt ? "joint_matrix int8xint8+int32 8x16x32"
           : "joint_matrix (bf16/fp16/tf32)x(bf16/fp16/tf32)+fp32 8x16x{16,16,8}",
     isInt ? "tops" : "tflops"});

#ifndef CLPEAK_ONEAPI_HAS_JOINT_MATRIX
  if (isInt)
    test.skip("joint_matrix_int8", ResultStatus::Unsupported,
              "joint_matrix header not available in this oneAPI toolchain");
  else
    test.skipAll({"joint_matrix_bf16", "joint_matrix_fp16", "joint_matrix_tf32"},
                 ResultStatus::Unsupported,
                 "joint_matrix header not available in this oneAPI toolchain");
  return 0;
#else
  namespace syclex = sycl::ext::oneapi::experimental::matrix;

  // Ground-truth diagnostic (verbose only): what shapes/types does this device
  // actually accept?  Dump once (FP pass), BEFORE the xmxSupported gate, so even
  // a device we mis-classify still reveals its real table under --verbose.
  if (!isInt)
    dumpMatrixCombinations(dev.dev);

  if (!dev.info.xmxSupported)
  {
    if (isInt)
      test.skip("joint_matrix_int8", ResultStatus::Unsupported,
                "XMX matrix engine not available on this device");
    else
      test.skipAll({"joint_matrix_bf16", "joint_matrix_fp16", "joint_matrix_tf32"},
                   ResultStatus::Unsupported,
                   "XMX matrix engine not available on this device");
    return 0;
  }

  // One sub-group per work-group: joint_matrix executes per sub-group and the
  // ops accounting below counts one matrix chain per block, so the work-group
  // must be exactly one XMX sub-group (JM_SG lanes), matching reqd_sub_group_size.
  const uint32_t blockSize = (uint32_t)JM_SG;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);

  uint64_t wantBlocks = globalThreads / blockSize;
  uint64_t bytesPerBlock = (uint64_t)JM_M * JM_N * sizeof(float);
  uint64_t maxBlocks = dev.info.totalGlobalMem / 4 / bytesPerBlock;
  uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
  if (pickBlocks == 0) pickBlocks = 1;
  uint32_t numBlocks = (uint32_t)pickBlocks;
  const uint64_t outElems = (uint64_t)numBlocks * JM_M * JM_N;

  const unsigned int forced = forceIters ? specifiedIters : 0;

  if (isInt)
  {
    int32_t *out = sycl::malloc_device<int32_t>(outElems, dev.stream);
    if (!out)
    {
      test.skip("joint_matrix_int8", ResultStatus::Error, "Failed to allocate output buffer");
      return -1;
    }
    if (jmComboSupport(dev.dev, syclex::matrix_type::sint8, syclex::matrix_type::sint8,
                       syclex::matrix_type::sint32, JM_M, JM_N, 32) == 0)
    {
      test.skip("joint_matrix_int8", ResultStatus::Unsupported,
                "int8 8x16x32 not in this device's matrix-engine combinations");
    }
    else
    {
      float us = runJmVariant<JmInt8Tag, int8_t, int8_t, int32_t, 32>(
          *this, dev, out, numBlocks, blockSize, (int8_t)1, (int32_t)0,
          cfg.targetTimeUs, forced);
      emitJm(test, "joint_matrix_int8", us, numBlocks, /*K=*/32);
    }
    sycl::free(out, dev.stream);
    return 0;
  }

  // FP category: bf16, fp16, tf32 — all fp32 accumulate, one shared buffer.
  float *out = sycl::malloc_device<float>(outElems, dev.stream);
  if (!out)
  {
    test.skipAll({"joint_matrix_bf16", "joint_matrix_fp16", "joint_matrix_tf32"},
                 ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

#ifdef CLPEAK_ONEAPI_JM_HAS_BF16
  if (jmComboSupport(dev.dev, syclex::matrix_type::bf16, syclex::matrix_type::bf16,
                     syclex::matrix_type::fp32, JM_M, JM_N, 16) == 0)
  {
    test.skip("joint_matrix_bf16", ResultStatus::Unsupported,
              "bf16 8x16x16 not in this device's matrix-engine combinations");
  }
  else
  {
    using bfloat16 = sycl::ext::oneapi::bfloat16;
    float us = runJmVariant<JmBf16Tag, bfloat16, bfloat16, float, 16>(
        *this, dev, out, numBlocks, blockSize, bfloat16(1.0f), 0.0f,
        cfg.targetTimeUs, forced);
    emitJm(test, "joint_matrix_bf16", us, numBlocks, /*K=*/16);
  }
#else
  test.skip("joint_matrix_bf16", ResultStatus::Unsupported,
            "SYCL bfloat16 header not available in this oneAPI toolchain");
#endif

  if (jmComboSupport(dev.dev, syclex::matrix_type::fp16, syclex::matrix_type::fp16,
                     syclex::matrix_type::fp32, JM_M, JM_N, 16) == 0)
  {
    test.skip("joint_matrix_fp16", ResultStatus::Unsupported,
              "fp16 8x16x16 not in this device's matrix-engine combinations");
  }
  else
  {
    float us = runJmVariant<JmFp16Tag, sycl::half, sycl::half, float, 16>(
        *this, dev, out, numBlocks, blockSize, (sycl::half)1.0f, 0.0f,
        cfg.targetTimeUs, forced);
    emitJm(test, "joint_matrix_fp16", us, numBlocks, /*K=*/16);
  }

  if (jmComboSupport(dev.dev, syclex::matrix_type::tf32, syclex::matrix_type::tf32,
                     syclex::matrix_type::fp32, JM_M, JM_N, 8) == 0)
  {
    // tf32 XMX is a PVC/Xe-HPC feature; Xe-HPG (Arc/DG2) reports no tf32 combo.
    test.skip("joint_matrix_tf32", ResultStatus::Unsupported,
              "tf32 8x16x8 not in this device's matrix-engine combinations (PVC-class only)");
  }
  else
  {
    // tf32: matrix element type is precision::tf32, filled with float.
    float us = runJmVariant<JmTf32Tag, syclex::precision::tf32, float, float, 8>(
        *this, dev, out, numBlocks, blockSize, 1.0f, 0.0f,
        cfg.targetTimeUs, forced);
    emitJm(test, "joint_matrix_tf32", us, numBlocks, /*K=*/8);
  }

  sycl::free(out, dev.stream);
  return 0;
#endif
}

#endif // ENABLE_ONEAPI
