#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <common/form_race.h>

#include <sycl/sycl.hpp>
#if defined(CLPEAK_ONEAPI_HAS_BF16) || __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#include <sycl/ext/oneapi/bfloat16.hpp>
#endif
#include <algorithm>

namespace clpeak_oneapi {
uint32_t pickComputeBlocks(const oneapi_device_info_t &info,
                           uint32_t blockSize, uint32_t outElemsPerBlock,
                           uint32_t elemSize);
float    computeFlops(uint64_t totalThreads, uint32_t workPerWI, float meanUs);
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
//
// A third shape races beside those two: the same affine chains with b
// uniform -- one value in every component -- and, for fp32/fp16, 128 chain
// instructions per trip.  Both are for Alchemist, which this backend mostly
// runs on.  At SIMD32 a mad whose three sources share a register bank pays for
// the conflict, and a uniform b caps that at one half of a mad; and these
// loops are pinned rolled (#pragma unroll 1), so at 16 a trip the counter is
// a tenth of the instructions issued.  The MAD chain block in
// include/common/common.h has the measurements.  It races rather than
// replacing the second shape because, in its place on Intel's CPU runtime
// (Threadripper 3955WX), it read float2/float4 at 291/701 GFLOPS against
// 783/1546 and cost double2 and half2-half8 30-54%, while lifting float16 and
// double4-16 by 38-90%: SYCL's vectors compile differently enough there that
// neither form wins everywhere.
//
// A fourth races wherever the device offers a sub-group of 16: the second
// shape's per-lane b at the third shape's depth, pinned to 16 lanes.  Vulkan
// and OpenCL time their affine chain at that width too, because at SIMD32 an
// Alchemist fp32 value spans four registers in both banks and the allocator
// lays the three sources of a mad out to collide; at SIMD16 they mostly do
// not.  On an Arc A380 (driver 8993), at the width the compiler picks, mp read
// 3.42 TFLOPS with the second shape and 3.89 with the third, and pinned to 16
// it read 4.60; float and float2 went from 4.08-4.12 to 4.85-4.88.  fp16 is
// where oneAPI parts company with the other two: their fp16 pinned to 16 runs
// at its fp32 rate, but here the pinned build read 9.50 against 9.75 at the
// compiler's width and won at half8, 9.37 against 7.79, so nothing on that
// card drops it.  The widths race as a clpeak::FormRace that drops only a
// clear loser (include/common/form_race.h), as in those two backends.
template <int W> struct AffineChains { static constexpr int N = (W == 1) ? 4 : (W == 2) ? 2 : 1; };

static bool offersSubGroup16(const oneapi_device_info_t &info)
{
  return std::find(info.subGroupSizes.begin(), info.subGroupSizes.end(), (size_t)16) !=
         info.subGroupSizes.end();
}

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
template <typename Tag, typename T, int W> class compute_fp_affu_kernel;
template <typename Tag, typename T, int W> class compute_fp_aff16_kernel;

// One vector-width variant of an FP compute test.  Builds sycl::vec<T,W>
// with distinct per-lane seeds (so the compiler can't collapse the vector to
// a scalar broadcast), runs the FMA dependency chain, reduces lanes into the
// output, times via runKernel, and emits the metric.  `widthRace` carries the
// sub-group race from one vector width to the next: form `false` is the
// compiler's width, `true` the pinned 16.
template <typename Tag, typename T, int W, int D>
static void runFpWidth(OneapiPeak &peak, OneapiDevice &dev,
                       logger::TestScope &test, const char *label,
                       T *out, uint64_t totalThreads, uint32_t blockSize,
                       int baseIters, double scalarA, uint32_t workPerWI,
                       unsigned int targetTimeUs, unsigned int forced,
                       clpeak::FormRace &widthRace)
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

  // Third shape: the affine chains with a uniform b, D MAD_16s a trip.
  int itersU = baseIters / (W * D);
  if (itersU < 1) itersU = 1;
  auto submitAffU = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_fp_affu_kernel<Tag, T, W>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          VecT xs[NCHAIN], a;
          #pragma unroll
          for (int k = 0; k < W; k++) a[k] = (T)it.get_local_id(0) + (T)k;
          const VecT b(A + (T)2);
          #pragma unroll
          for (int n = 0; n < NCHAIN; n++)
          {
            #pragma unroll
            for (int k = 0; k < W; k++) xs[n][k] = A + (T)k + (T)(n * W);
          }

          #pragma unroll 1
          for (int i = 0; i < itersU; i++)
          {
            #pragma unroll
            for (int m = 0; m < 16 * D / NCHAIN; m++)
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

  // Fourth shape: the second's per-lane b at the third's depth, pinned to a
  // sub-group of 16.  Submitted only where the device offers 16.
  auto submitAff16 = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_fp_aff16_kernel<Tag, T, W>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
          VecT xs[NCHAIN], a, b;
          #pragma unroll
          for (int k = 0; k < W; k++)
          {
            a[k] = (T)it.get_local_id(0) + (T)k;
            b[k] = a[k] + (T)2;
          }
          #pragma unroll
          for (int n = 0; n < NCHAIN; n++)
          {
            #pragma unroll
            for (int k = 0; k < W; k++) xs[n][k] = A + (T)k + (T)(n * W);
          }

          #pragma unroll 1
          for (int i = 0; i < itersU; i++)
          {
            #pragma unroll
            for (int m = 0; m < 16 * D / NCHAIN; m++)
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
  const float squaring = clpeak_oneapi::computeFlops(totalThreads, workPerWI, us);
  float value = squaring;

  // Time a second shape against the squaring reading: its rate, or 0 when it
  // failed to launch or the fold guard rejects it.  A failure here is not an
  // error -- the squaring chain already produced a reading.
  auto timeShape = [&](const OneapiPeak::KernelSubmitter &shape, const char *what) -> double {
    float shapeUs = peak.runKernel(dev, shape, targetTimeUs, forced);
    if (shapeUs <= 0.0f)
      return 0.0;
    float rate = clpeak_oneapi::computeFlops(totalThreads, workPerWI, shapeUs);
    CLPEAK_VLOG("%s: %s %.1f flops\n", label, what, rate);
    if (rate > squaring * MAX_ALT_CHAIN_RATIO)
    {
      CLPEAK_VLOG("%s: %s %.1fx faster -- rejecting it as a compiler fold\n",
                  label, what, rate / squaring);
      return 0.0;
    }
    return rate;
  };

  CLPEAK_VLOG("%s: squaring chain %.1f flops\n", label, squaring);
  const bool sg16 = offersSubGroup16(dev.info);
  double raced[2] = {0.0, 0.0};
  if (widthRace.runs(false))
  {
    raced[0] = timeShape(submitAff, "alt chain");
    raced[0] = std::max(raced[0], timeShape(submitAffU, "uniform-b alt chain"));
  }
  if (sg16 && widthRace.runs(true))
    raced[1] = timeShape(submitAff16, "alt chain at sub-group 16");
  value = std::max(value, (float)std::max(raced[0], raced[1]));

  if (sg16)
  {
    if (widthRace.runs(true) && raced[1] <= 0.0)
      widthRace.drop(true);
    if (widthRace.runs(false) && widthRace.runs(true))
    {
      widthRace.dropTrailing(raced);
      if (!widthRace.runs(false) || !widthRace.runs(true))
        CLPEAK_VLOG("%s: alt chain at the %s sub-group trails by %.1fx -- dropped from here\n",
                    label, widthRace.runs(true) ? "compiler's" : "16",
                    widthRace.runs(true) ? raced[1] / raced[0] : raced[0] / raced[1]);
    }
  }

  test.emit(label, value, note);
}

// Drive the {1,2,4,8,16} sweep for one FP family.  D is the uniform-b shape's
// MAD_16s per trip.
template <typename Tag, typename T, int D>
static void runFpSweep(OneapiPeak &peak, OneapiDevice &dev,
                       logger::TestScope &test, const char *baseLabel,
                       T *out, uint64_t totalThreads, uint32_t blockSize,
                       int baseIters, double scalarA, uint32_t workPerWI,
                       unsigned int targetTimeUs, unsigned int forced)
{
  const std::string b(baseLabel);
  clpeak::FormRace widthRace;
  runFpWidth<Tag, T, 1, D>(peak, dev, test, b.c_str(),          out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced, widthRace);
  runFpWidth<Tag, T, 2, D>(peak, dev, test, (b + "2").c_str(),  out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced, widthRace);
  runFpWidth<Tag, T, 4, D>(peak, dev, test, (b + "4").c_str(),  out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced, widthRace);
  runFpWidth<Tag, T, 8, D>(peak, dev, test, (b + "8").c_str(),  out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced, widthRace);
  // 16-wide keeps one MAD_16 a trip -- already 256 lane-FMAs -- like OpenCL's
  // float16, which as one straight-line trip Intel's CPU runtime stopped
  // vectorising across work-items.
  runFpWidth<Tag, T, 16, 1>(peak, dev, test, (b + "16").c_str(), out, totalThreads, blockSize, baseIters, scalarA, workPerWI, targetTimeUs, forced, widthRace);
}

// --------------------------------------------------------------------------
// Single precision — float / float2 / float4 / float8 / float16
// --------------------------------------------------------------------------
int OneapiPeak::runComputeSP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"single_precision_compute", "Single-precision compute", "flops",
     Category::Unknown,
     "Peak fp32 arithmetic rate of the device's ALUs, with no memory traffic.",
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

  runFpSweep<SpTag, float, 8>(*this, dev, test, "float", out, totalThreads, blockSize,
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
    {"half_precision_compute", "Half-precision compute", "flops",
     Category::Unknown,
     "Peak fp16 arithmetic rate, with fp16 inputs and accumulator.",
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

  runFpSweep<HpTag, sycl::half, 8>(*this, dev, test, "half", out, totalThreads, blockSize,
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
    {"double_precision_compute", "Double-precision compute", "flops",
     Category::Unknown,
     "Peak fp64 arithmetic rate.",
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

  runFpSweep<DpTag, double, 1>(*this, dev, test, "double", out, totalThreads, blockSize,
                            /*baseIters=*/16, /*A=*/1.3, COMPUTE_DP_WORK_PER_WI,
                            cfg.targetTimeUs, forceIters ? specifiedIters : 0);

  sycl::free(out, dev.stream);
  return 0;
}

// The scalar families' race (mp, bf16), from the squaring chain's time: the
// affine chain, its uniform-b twin and, where the device offers a sub-group
// of 16, the affine chain pinned to it.  The fastest reading the fold guard
// accepts wins.
static float raceScalarShapes(OneapiPeak &peak, OneapiDevice &dev, const char *label,
                              uint64_t totalThreads, float squaringUs,
                              const OneapiPeak::KernelSubmitter &alt,
                              const OneapiPeak::KernelSubmitter &altU,
                              const OneapiPeak::KernelSubmitter &alt16,
                              unsigned int targetTimeUs, unsigned int forced)
{
  const float squaring = clpeak_oneapi::computeFlops(totalThreads, COMPUTE_FP_WORK_PER_WI, squaringUs);
  CLPEAK_VLOG("%s: squaring chain %.1f flops\n", label, squaring);
  float value = squaring;
  auto race = [&](const OneapiPeak::KernelSubmitter &shape, const char *what) {
    float shapeUs = peak.runKernel(dev, shape, targetTimeUs, forced);
    if (shapeUs <= 0.0f)
      return;
    float rate = clpeak_oneapi::computeFlops(totalThreads, COMPUTE_FP_WORK_PER_WI, shapeUs);
    CLPEAK_VLOG("%s: %s %.1f flops\n", label, what, rate);
    if (rate > squaring * MAX_ALT_CHAIN_RATIO)
      CLPEAK_VLOG("%s: %s %.1fx faster -- rejecting it as a compiler fold\n",
                  label, what, rate / squaring);
    else if (rate > value)
      value = rate;
  };
  race(alt, "alt chain");
  race(altU, "uniform-b alt chain");
  if (offersSubGroup16(dev.info))
    race(alt16, "alt chain at sub-group 16");
  return value;
}

// --------------------------------------------------------------------------
// Mixed precision (fp16 multiply -> fp32 accumulate).  Mirrors compute_mp.hip:
// round-trip through half to force the lower-precision multiply, accumulate
// in float.  Scalar only (the round-trip is inherently per-element).  4096 ops/WI.
// --------------------------------------------------------------------------
class compute_mp_kernel;
class compute_mp_alt_kernel;
class compute_mp_altu_kernel;
class compute_mp_alt16_kernel;

int OneapiPeak::runComputeMP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"mixed_precision_compute", "Mixed-precision compute fp16xfp16+fp32", "flops",
     Category::Unknown,
     "Peak rate of fp16 multiplies accumulated in fp32, without the matrix engine.",
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

  // The affine chain, built twice: at the compiler's sub-group size and
  // pinned to 16.
  auto altChain = [=](sycl::nd_item<1> it) {
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
  };
  auto submitAlt = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_mp_alt_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize), altChain);
    });
  };
  auto submitAlt16 = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_mp_alt16_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] { altChain(it); });
    });
  };

  // The same chains with b uniform, raced beside them.
  auto submitAltU = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_mp_altu_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          float a = (float)(sycl::half)(float)it.get_local_id(0);
          float x0 = (float)(sycl::half)A;
          float b = x0 + 2.0f;   // uniform: see the third shape above
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
  test.emit("mp", raceScalarShapes(*this, dev, "mp", totalThreads, us, submitAlt, submitAltU,
                                   submitAlt16, cfg.targetTimeUs,
                                   forceIters ? specifiedIters : 0));

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
class compute_bf16_altu_kernel;
class compute_bf16_alt16_kernel;

int OneapiPeak::runComputeBF16(OneapiDevice &dev, benchmark_config_t &cfg)
{
  using bfloat16 = sycl::ext::oneapi::bfloat16;

  auto test = currentDeviceScope->beginTest(
    {"bfloat16_compute", "BF16 compute bf16xbf16+fp32", "flops",
     Category::Unknown,
     "Peak rate of bf16 multiplies accumulated in fp32, without the matrix engine.  "
     "Devices without bf16 hardware emulate it, at a lower rate.",
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

  // The affine chain, built twice: at the compiler's sub-group size and
  // pinned to 16.
  auto altChain = [=](sycl::nd_item<1> it) {
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
  };
  auto submitAlt = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_bf16_alt_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize), altChain);
    });
  };
  auto submitAlt16 = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_bf16_alt16_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] { altChain(it); });
    });
  };

  // The same chains with b uniform, raced beside them.
  auto submitAltU = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_bf16_altu_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          float a = (float)bfloat16((float)it.get_local_id(0));
          float x0 = (float)bfloat16(A);
          float b = x0 + 2.0f;   // uniform: see the third shape at the top
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
  test.emit("bf16", raceScalarShapes(*this, dev, "bf16", totalThreads, us, submitAlt, submitAltU,
                                     submitAlt16, cfg.targetTimeUs,
                                     forceIters ? specifiedIters : 0));

  sycl::free(out, dev.stream);
  return 0;
}
#else
int OneapiPeak::runComputeBF16(OneapiDevice &, benchmark_config_t &)
{
  auto test = currentDeviceScope->beginTest(
    {"bfloat16_compute", "BF16 compute bf16xbf16+fp32", "flops",
     Category::Unknown,
     "Peak rate of bf16 multiplies accumulated in fp32, without the matrix engine.  "
     "Devices without bf16 hardware emulate it, at a lower rate.",
     TestShape::Homogeneous, "vector width"});
  test.skip("bf16", ResultStatus::Unsupported,
            "SYCL bfloat16 header not available in this oneAPI toolchain");
  return 0;
}
#endif

#endif // ENABLE_ONEAPI
