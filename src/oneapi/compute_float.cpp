#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>

#include <sycl/sycl.hpp>
#if defined(CLPEAK_ONEAPI_HAS_BF16) || __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#include <sycl/ext/oneapi/bfloat16.hpp>
#endif

namespace clpeak_oneapi {
uint32_t pickComputeBlocks(const oneapi_device_info_t &info,
                           uint32_t blockSize, uint32_t outElemsPerBlock,
                           uint32_t elemSize);
float    computeGflops(uint64_t totalThreads, uint32_t workPerWI, float meanUs,
                       double unitDivider);
}

// --------------------------------------------------------------------------
// MAD macro shape matches the ROCm/CUDA/OpenCL backends: 16 fused mul-adds
// per MAD_16, x = x*x + c, building a dependency chain the compiler cannot
// hoist or vectorize away.  One MAD_16 = 16 fma = 32 flops per lane.
//
// Chain shape and why: see the MAD chain block in include/common/common.h.
//
// Total ops/WI is width-invariant: for vector width W we run baseIters/W
// outer iterations, each doing 32*W flops, so total = baseIters*32 flops/WI.
// SP/HP: baseIters=128 -> 4096 (COMPUTE_FP_WORK_PER_WI).
// DP:    baseIters=16  -> 512  (COMPUTE_DP_WORK_PER_WI).
// --------------------------------------------------------------------------
#define MAD_4(x, c)  x = sycl::fma(x, x, c); x = sycl::fma(x, x, c); \
                     x = sycl::fma(x, x, c); x = sycl::fma(x, x, c);
#define MAD_16(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c) MAD_4(x, c)

// Second chain shape, raced against the one above: x_k = a*x_k + b over N
// independent accumulators, three distinct source registers per fma.  Intel
// Alchemist halves a three-source mad whose operands are not all distinct, so
// x = x*x + c reports half rate there -- which is most of what this backend
// runs on.  Full rationale, and why the integer families use a third shape
// instead: the MAD chain block in include/common/common.h.
//
// N is 4 at width 1, 2 at width 2 and 1 above, where the vector itself
// supplies the parallelism; live values per lane stay at ~5 either way.
// Both shapes issue 16 chain instructions per outer iteration, so the
// per-work-item op budget and the two readings stay comparable.
template <int W> struct AffineChains { static constexpr int N = (W == 1) ? 4 : (W == 2) ? 2 : 1; };

// Four-chain affine, for the families whose accumulator round-trips through a
// narrow type once per outer iteration (mp, bf16).  The chains earn their keep:
// with one chain, mp on an Intel CPU runtime measured 373.1 against 373.8, a
// dead wash, while the four-chain float path on the same device went 396 ->
// 1552 GFLOPS.
//
// Four chains do cost four conversions per round against the squaring build's
// one, which is why both families run a 128-instruction inner loop rather than
// MAD_16.  Amortised over 16 that was 25% uncounted instruction overhead, and
// on Alchemist it is not a shape you can decline -- the squaring build, which
// needs only one conversion, is the one the three-distinct-source rule halves
// there.  An Arc A380 read mp at 2.78 TFLOPS against 4.89 for fp32 (57%) at
// MAD_16, where an RTX 5060, free to take the squaring build, sat at 88%.
// Over 128 the same four conversions cost ~3%.
#define MAD_G_AFF4(a, b)     x0 = sycl::fma(a, x0, b); x1 = sycl::fma(a, x1, b); \
                             x2 = sycl::fma(a, x2, b); x3 = sycl::fma(a, x3, b);
#define MAD_16_AFF4(a, b)    MAD_G_AFF4(a, b) MAD_G_AFF4(a, b) \
                             MAD_G_AFF4(a, b) MAD_G_AFF4(a, b)

// Per-family kernel-name tags (SYCL needs a unique type per parallel_for).
namespace { struct SpTag; struct HpTag; struct DpTag; }
template <typename Tag, typename T, int W> class compute_fp_vec_kernel;
template <typename Tag, typename T, int W> class compute_fp_aff_kernel;

// One vector-width variant of an FP compute test.  Builds sycl::vec<T,W>
// with distinct per-lane seeds (so the compiler can't collapse the vector to
// a scalar broadcast), runs the FMA dependency chain, reduces lanes into the
// output, times via runKernel, and emits the metric.
template <typename Tag, typename T, int W>
static void runFpWidth(OneapiPeak &peak, OneapiDevice &dev,
                       logger::TestScope &test, const char *label,
                       T *out, uint64_t totalThreads, uint32_t blockSize,
                       int baseIters, double scalarA, uint32_t workPerWI,
                       unsigned int targetTimeUs, unsigned int forced)
{
  using VecT = sycl::vec<T, W>;
  int iters = baseIters / W;
  if (iters < 1) iters = 1;
  const T A = (T)scalarA;

  auto submit = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_fp_vec_kernel<Tag, T, W>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          VecT x, c;
          // Seeds in T arithmetic only: a double here would pull in the fp64
          // aspect and make the kernel fail to launch on devices without fp64
          // (e.g. Intel Arc). That only bit the vector widths, because the
          // scalar W=1 case constant-folds the k==0 double term away.
          #pragma unroll
          for (int k = 0; k < W; k++)
          {
            x[k] = A + (T)k;
            c[k] = (T)it.get_local_id(0) + (T)k;
          }
          #pragma unroll 1
          for (int i = 0; i < iters; i++) { MAD_16(x, c) }
          VecT r = x;
          T acc = (T)0;
          #pragma unroll
          for (int k = 0; k < W; k++) acc += r[k];
          out[it.get_global_id(0)] = acc;
        });
    });
  };

  // Affine twin of the same kernel: N independent x_k = a*x_k + b chains,
  // same 16 chain instructions per outer iteration.
  constexpr int NCHAIN = AffineChains<W>::N;
  auto submitAff = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_fp_aff_kernel<Tag, T, W>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          VecT xs[NCHAIN], a, b;
          // Seeds in T arithmetic only, for the same fp64-aspect reason as
          // the squaring kernel above.
          #pragma unroll
          for (int k = 0; k < W; k++)
          {
            a[k] = (T)it.get_local_id(0) + (T)k;
            b[k] = a[k] + (T)2;
          }
          // Chain n starts a whole vector width past chain n-1, not one past.
          // Two scalar chains that start on the same value stay bitwise
          // identical forever, and a compiler that scalarises the vector then
          // CSEs one away, inflating the reading by NCHAIN/(NCHAIN-1).  At
          // W=2 a +n spacing gave xs[0] = (A, A+1) and xs[1] = (A+1, A+2).
          #pragma unroll
          for (int n = 0; n < NCHAIN; n++)
          {
            #pragma unroll
            for (int k = 0; k < W; k++) xs[n][k] = A + (T)k + (T)(n * W);
          }

          #pragma unroll 1
          for (int i = 0; i < iters; i++)
          {
            #pragma unroll
            for (int m = 0; m < 16 / NCHAIN; m++)
            {
              #pragma unroll
              for (int n = 0; n < NCHAIN; n++) xs[n] = sycl::fma(a, xs[n], b);
            }
          }

          VecT r = xs[0];
          #pragma unroll
          for (int n = 1; n < NCHAIN; n++) r += xs[n];
          T acc = (T)0;
          #pragma unroll
          for (int k = 0; k < W; k++) acc += r[k];
          out[it.get_global_id(0)] = acc;
        });
    });
  };

  const char *note = oneapiWidthNote(W);
  float us = peak.runKernel(dev, submit, targetTimeUs, forced);
  if (us <= 0.0f)
  {
    test.skip(label, ResultStatus::Error, "kernel launch failed", note);
    return;
  }
  float value = clpeak_oneapi::computeGflops(totalThreads, workPerWI, us, 1e9);

  // Race the affine chain and keep the faster reading.  A failure here is not
  // an error -- the squaring chain already produced one.
  float affUs = peak.runKernel(dev, submitAff, targetTimeUs, forced);
  if (affUs > 0.0f)
  {
    float affValue = clpeak_oneapi::computeGflops(totalThreads, workPerWI, affUs, 1e9);
    CLPEAK_VLOG("%s: squaring chain %.1f, alt chain %.1f gflops\n",
                label, value, affValue);
    if (affValue > value * MAX_ALT_CHAIN_RATIO)
      CLPEAK_VLOG("%s: alt chain %.1fx faster -- rejecting it as a compiler fold\n",
                  label, affValue / value);
    else if (affValue > value)
      value = affValue;
  }

  test.emit(label, value, note);
}

// Drive the {1,2,4,8,16} sweep for one FP family.
template <typename Tag, typename T>
static void runFpSweep(OneapiPeak &peak, OneapiDevice &dev,
                       logger::TestScope &test, const char *baseLabel,
                       T *out, uint64_t totalThreads, uint32_t blockSize,
                       int baseIters, double scalarA, uint32_t workPerWI,
                       unsigned int targetTimeUs, unsigned int forced)
{
  const std::string b(baseLabel);
  runFpWidth<Tag, T, 1 >(peak, dev, test, b.c_str(),          out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced);
  runFpWidth<Tag, T, 2 >(peak, dev, test, (b + "2").c_str(),  out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced);
  runFpWidth<Tag, T, 4 >(peak, dev, test, (b + "4").c_str(),  out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced);
  runFpWidth<Tag, T, 8 >(peak, dev, test, (b + "8").c_str(),  out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced);
  runFpWidth<Tag, T, 16>(peak, dev, test, (b + "16").c_str(), out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced);
}

// --------------------------------------------------------------------------
// Single precision — float / float2 / float4 / float8 / float16
// --------------------------------------------------------------------------
int OneapiPeak::runComputeSP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"single_precision_compute", "Single-precision compute", "gflops",
     Category::Unknown,
     "Peak arithmetic speed of the device's compute units on 32-bit fractional "
     "numbers -- the ordinary float type.  Nothing touches memory, so only the "
     "arithmetic units limit the rate.",
     TestShape::Homogeneous, "vector width"});

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(float));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  float *out = sycl::malloc_device<float>(totalThreads, dev.stream);
  if (!out)
  {
    test.skipAll({"float", "float2", "float4", "float8", "float16"},
                 ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  runFpSweep<SpTag, float>(*this, dev, test, "float", out, totalThreads, blockSize,
                           /*baseIters=*/128, /*A=*/1.3, COMPUTE_FP_WORK_PER_WI,
                           cfg.targetTimeUs, forceIters ? specifiedIters : 0);

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Half precision — half / half2 / half4 / half8 / half16
// --------------------------------------------------------------------------
int OneapiPeak::runComputeHP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"half_precision_compute", "Half-precision compute", "gflops",
     Category::Unknown,
     "Peak arithmetic speed on 16-bit fractional numbers -- half the size of a "
     "normal float, and what graphics and on-device AI mostly run on.",
     TestShape::Homogeneous, "vector width"});

  if (!dev.info.fp16Supported)
  {
    test.skipAll({"half", "half2", "half4", "half8", "half16"},
                 ResultStatus::Unsupported, "fp16 not supported by this oneAPI device");
    return 0;
  }

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(sycl::half));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  sycl::half *out = sycl::malloc_device<sycl::half>(totalThreads, dev.stream);
  if (!out)
  {
    test.skipAll({"half", "half2", "half4", "half8", "half16"},
                 ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  runFpSweep<HpTag, sycl::half>(*this, dev, test, "half", out, totalThreads, blockSize,
                                /*baseIters=*/128, /*A=*/1.3, COMPUTE_FP_WORK_PER_WI,
                                cfg.targetTimeUs, forceIters ? specifiedIters : 0);

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Double precision — double / double2 / double4 / double8 / double16
// workPerWI = 512 (COMPUTE_DP_WORK_PER_WI), baseIters = 16.
// --------------------------------------------------------------------------
int OneapiPeak::runComputeDP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"double_precision_compute", "Double-precision compute", "gflops",
     Category::Unknown,
     "Peak arithmetic speed on 64-bit fractional numbers, the high-accuracy type "
     "scientific computing relies on.  Consumer graphics parts run these far "
     "slower than 32-bit; the datacenter parts do not.",
     TestShape::Homogeneous, "vector width"});

  if (!dev.info.fp64Supported)
  {
    test.skipAll({"double", "double2", "double4", "double8", "double16"},
                 ResultStatus::Unsupported, "fp64 not supported by this oneAPI device");
    return 0;
  }

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(double));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  double *out = sycl::malloc_device<double>(totalThreads, dev.stream);
  if (!out)
  {
    test.skipAll({"double", "double2", "double4", "double8", "double16"},
                 ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  runFpSweep<DpTag, double>(*this, dev, test, "double", out, totalThreads, blockSize,
                            /*baseIters=*/16, /*A=*/1.3, COMPUTE_DP_WORK_PER_WI,
                            cfg.targetTimeUs, forceIters ? specifiedIters : 0);

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Mixed precision (fp16 multiply -> fp32 accumulate).  Mirrors compute_mp.hip:
// round-trip through half to force the lower-precision multiply, accumulate
// in float.  Scalar only (the round-trip is inherently per-element).  4096 ops/WI.
// --------------------------------------------------------------------------
class compute_mp_kernel;
class compute_mp_alt_kernel;

int OneapiPeak::runComputeMP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"mixed_precision_compute", "Mixed-precision compute fp16xfp16+fp32", "gflops",
     Category::Unknown,
     "Peak speed when the device multiplies 16-bit numbers but keeps the running "
     "total in 32 bits -- the accuracy-preserving pattern AI code uses.  This is "
     "the general compute units, not the matrix engine.",
     TestShape::Homogeneous, "vector width"});

  if (!dev.info.fp16Supported)
  {
    test.skip("mp", ResultStatus::Unsupported, "fp16 not supported by this oneAPI device");
    return 0;
  }

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(float));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  float *out = sycl::malloc_device<float>(totalThreads, dev.stream);
  if (!out)
  {
    test.skip("mp", ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }
  const float A = 1.3f;

  auto submit = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_mp_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          float x = (float)(sycl::half)A;
          float c = (float)(sycl::half)(float)it.get_local_id(0);
          #pragma unroll 1
          for (int i = 0; i < 16; i++) {
            // MAD_128 = 8 * MAD_16 = 256 ops
            MAD_16(x, c) MAD_16(x, c) MAD_16(x, c) MAD_16(x, c)
            MAD_16(x, c) MAD_16(x, c) MAD_16(x, c) MAD_16(x, c)
            x = (float)(sycl::half)x;
          }
          out[it.get_global_id(0)] = x;
        });
    });
  };

  auto submitAlt = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_mp_alt_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          float a = (float)(sycl::half)(float)it.get_local_id(0);
          float b = a + 2.0f;
          float x0 = (float)(sycl::half)A;
          float x1 = x0 + 1.0f, x2 = x0 + 2.0f, x3 = x0 + 3.0f;
          #pragma unroll 1
          for (int i = 0; i < 16; i++) {
            // MAD_128 = 8 * MAD_16_AFF4 = 256 ops
            MAD_16_AFF4(a, b) MAD_16_AFF4(a, b) MAD_16_AFF4(a, b) MAD_16_AFF4(a, b)
            MAD_16_AFF4(a, b) MAD_16_AFF4(a, b) MAD_16_AFF4(a, b) MAD_16_AFF4(a, b)
            x0 = (float)(sycl::half)x0; x1 = (float)(sycl::half)x1;
            x2 = (float)(sycl::half)x2; x3 = (float)(sycl::half)x3;
          }
          out[it.get_global_id(0)] = (x0 + x1) + (x2 + x3);
        });
    });
  };

  float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
  {
    test.skip("mp", ResultStatus::Error, "kernel launch failed");
    sycl::free(out, dev.stream);
    return 0;
  }
  float value = clpeak_oneapi::computeGflops(totalThreads, COMPUTE_FP_WORK_PER_WI, us, 1e9);

  // Race the affine chain and keep the faster reading.
  float altUs = runKernel(dev, submitAlt, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (altUs > 0.0f)
  {
    float altValue = clpeak_oneapi::computeGflops(totalThreads, COMPUTE_FP_WORK_PER_WI, altUs, 1e9);
    CLPEAK_VLOG("mp: squaring chain %.1f, alt chain %.1f gflops\n", value, altValue);
    if (altValue > value * MAX_ALT_CHAIN_RATIO)
      CLPEAK_VLOG("mp: alt chain %.1fx faster -- rejecting it as a compiler fold\n",
                  altValue / value);
    else if (altValue > value)
      value = altValue;
  }
  test.emit("mp", value);

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// BF16 compute (bf16xbf16 -> fp32 accumulate).  Gated by aspect probe;
// emulated on iGPUs without native bf16, hardware on Arc/PVC/Battlemage.
// Scalar only.  16 outer iters * MAD_128 (256 ops) = 4096 ops/WI.
// --------------------------------------------------------------------------
#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
class compute_bf16_kernel;
class compute_bf16_alt_kernel;

int OneapiPeak::runComputeBF16(OneapiDevice &dev, benchmark_config_t &cfg)
{
  using bfloat16 = sycl::ext::oneapi::bfloat16;

  auto test = currentDeviceScope->beginTest(
    {"bfloat16_compute", "BF16 compute bf16xbf16+fp32", "gflops",
     Category::Unknown,
     "Peak speed on bfloat16 -- 16 bits arranged for AI work, trading digits of "
     "accuracy for the number range of a full float.  Integrated graphics "
     "without bf16 hardware emulate it, and the rate drops accordingly.",
     TestShape::Homogeneous, "vector width"});

  if (!dev.info.bf16Supported)
  {
    test.skip("bf16", ResultStatus::Unsupported, "bf16 not supported by this oneAPI device");
    return 0;
  }

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(float));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  float *out = sycl::malloc_device<float>(totalThreads, dev.stream);
  if (!out)
  {
    test.skip("bf16", ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }
  const float A = 1.3f;

  auto submit = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_bf16_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          float x = (float)bfloat16(A);
          float c = (float)bfloat16((float)it.get_local_id(0));
          #pragma unroll 1
          for (int i = 0; i < 16; i++) {
            // MAD_128 = 8 * MAD_16 = 256 ops
            MAD_16(x, c) MAD_16(x, c) MAD_16(x, c) MAD_16(x, c)
            MAD_16(x, c) MAD_16(x, c) MAD_16(x, c) MAD_16(x, c)
            x = (float)bfloat16(x);
          }
          out[it.get_global_id(0)] = x;
        });
    });
  };

  auto submitAlt = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_bf16_alt_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          float a = (float)bfloat16((float)it.get_local_id(0));
          float b = a + 2.0f;
          float x0 = (float)bfloat16(A);
          float x1 = x0 + 1.0f, x2 = x0 + 2.0f, x3 = x0 + 3.0f;
          #pragma unroll 1
          for (int i = 0; i < 16; i++) {
            MAD_16_AFF4(a, b) MAD_16_AFF4(a, b) MAD_16_AFF4(a, b) MAD_16_AFF4(a, b)
            MAD_16_AFF4(a, b) MAD_16_AFF4(a, b) MAD_16_AFF4(a, b) MAD_16_AFF4(a, b)
            x0 = (float)bfloat16(x0); x1 = (float)bfloat16(x1);
            x2 = (float)bfloat16(x2); x3 = (float)bfloat16(x3);
          }
          out[it.get_global_id(0)] = (x0 + x1) + (x2 + x3);
        });
    });
  };

  float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
  {
    test.skip("bf16", ResultStatus::Error, "kernel launch failed");
    sycl::free(out, dev.stream);
    return 0;
  }
  float value = clpeak_oneapi::computeGflops(totalThreads, COMPUTE_FP_WORK_PER_WI, us, 1e9);

  // Race the affine chain and keep the faster reading.
  float altUs = runKernel(dev, submitAlt, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (altUs > 0.0f)
  {
    float altValue = clpeak_oneapi::computeGflops(totalThreads, COMPUTE_FP_WORK_PER_WI, altUs, 1e9);
    CLPEAK_VLOG("bf16: squaring chain %.1f, alt chain %.1f gflops\n", value, altValue);
    if (altValue > value * MAX_ALT_CHAIN_RATIO)
      CLPEAK_VLOG("bf16: alt chain %.1fx faster -- rejecting it as a compiler fold\n",
                  altValue / value);
    else if (altValue > value)
      value = altValue;
  }
  test.emit("bf16", value);

  sycl::free(out, dev.stream);
  return 0;
}
#else
int OneapiPeak::runComputeBF16(OneapiDevice &, benchmark_config_t &)
{
  auto test = currentDeviceScope->beginTest(
    {"bfloat16_compute", "BF16 compute bf16xbf16+fp32", "gflops",
     Category::Unknown,
     "Peak speed on bfloat16 -- 16 bits arranged for AI work, trading digits of "
     "accuracy for the number range of a full float.  Integrated graphics "
     "without bf16 hardware emulate it, and the rate drops accordingly.",
     TestShape::Homogeneous, "vector width"});
  test.skip("bf16", ResultStatus::Unsupported,
            "SYCL bfloat16 header not available in this oneAPI toolchain");
  return 0;
}
#endif

#endif // ENABLE_ONEAPI
