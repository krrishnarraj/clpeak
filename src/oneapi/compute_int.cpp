#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>
#include <string>

namespace clpeak_oneapi {
uint32_t pickComputeBlocks(const oneapi_device_info_t &info,
                           uint32_t blockSize, uint32_t outElemsPerBlock,
                           uint32_t elemSize);
float    computeGflops(uint64_t totalThreads, uint32_t workPerWI, float meanUs,
                       double unitDivider);
}

// Integer MAD macros: shape mirrors compute_int32.hip exactly.  The alternating
// read/write builds a dependency chain so the loop can't be hoisted.
// Chain shape and why: see the MAD chain block in include/common/common.h.
// One IMAD_16 = 16 mul-adds = 32 int ops per lane.  Width-invariant total:
// width W runs baseIters/W iters * 32*W ops = baseIters*32 ops/WI.
// baseIters=128 -> 4096 (COMPUTE_FP_WORK_PER_WI, matches ROCm int32).
#define IMAD_4(x, c)  x = x * x + c; x = x * x + c; x = x * x + c; x = x * x + c;
#define IMAD_16(x, c) IMAD_4(x, c) IMAD_4(x, c) IMAD_4(x, c) IMAD_4(x, c)

// Second chain shape, raced against the one above: x_k = x_k * x_(k+1) + c
// over N accumulators.  Intel Alchemist halves a three-source mad whose source
// operands are not all distinct, and x = x*x + c reads {x, x, c}; rotating the
// multiplier through the other accumulators fixes that while staying
// quadratic, which the affine shape the FP families use is not.  That matters
// here and not there: an integer affine recurrence folds legally, because
// integer multiply and add really are associative and distributive.  Full
// rationale in the MAD chain block in include/common/common.h.
//
// N is 4 at width 1 and 2 above; rotating needs two accumulators minimum to
// stay quadratic.  Both shapes issue 16 chain instructions per outer
// iteration, so the two readings stay comparable.
template <int W> struct RotChains { static constexpr int N = (W == 1) ? 4 : 2; };

namespace { struct IntTag; }
template <typename Tag, int W> class compute_int_vec_kernel;
template <typename Tag, int W> class compute_int_rot_kernel;

template <typename Tag, int W>
static void runIntWidth(OneapiPeak &peak, OneapiDevice &dev,
                        logger::TestScope &test, const char *label,
                        int *out, uint64_t totalThreads, uint32_t blockSize,
                        int baseIters, int scalarA, uint32_t workPerWI,
                        unsigned int targetTimeUs, unsigned int forced)
{
  using VecT = sycl::vec<int, W>;
  int iters = baseIters / W;
  if (iters < 1) iters = 1;
  const int A = scalarA;

  auto submit = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_int_vec_kernel<Tag, W>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          VecT x, c;
          #pragma unroll
          for (int k = 0; k < W; k++)
          {
            x[k] = A + k;
            c[k] = (int)it.get_local_id(0) + k;
          }
          #pragma unroll 1
          for (int i = 0; i < iters; i++) { IMAD_16(x, c) }
          int acc = 0;
          #pragma unroll
          for (int k = 0; k < W; k++) acc += x[k];
          out[it.get_global_id(0)] = acc;
        });
    });
  };

  constexpr int NCHAIN = RotChains<W>::N;
  auto submitRot = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_int_rot_kernel<Tag, W>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          VecT xs[NCHAIN], c;
          #pragma unroll
          for (int k = 0; k < W; k++) c[k] = (int)it.get_local_id(0) + k;
          #pragma unroll
          for (int n = 0; n < NCHAIN; n++)
          {
            #pragma unroll
            for (int k = 0; k < W; k++) xs[n][k] = A + k + n;
          }

          #pragma unroll 1
          for (int i = 0; i < iters; i++)
          {
            #pragma unroll
            for (int m = 0; m < 16 / NCHAIN; m++)
            {
              #pragma unroll
              for (int n = 0; n < NCHAIN; n++)
                xs[n] = xs[n] * xs[(n + 1) % NCHAIN] + c;
            }
          }

          VecT r = xs[0];
          #pragma unroll
          for (int n = 1; n < NCHAIN; n++) r += xs[n];
          int acc = 0;
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

  // Race the rotating chain and keep the faster reading.
  float rotUs = peak.runKernel(dev, submitRot, targetTimeUs, forced);
  if (rotUs > 0.0f)
  {
    float rotValue = clpeak_oneapi::computeGflops(totalThreads, workPerWI, rotUs, 1e9);
    CLPEAK_VLOG("%s: squaring chain %.1f, alt chain %.1f gops\n", label, value, rotValue);
    if (rotValue > value * MAX_ALT_CHAIN_RATIO)
      CLPEAK_VLOG("%s: alt chain %.1fx faster -- rejecting it as a compiler fold\n",
                  label, rotValue / value);
    else if (rotValue > value)
      value = rotValue;
  }

  test.emit(label, value, {false, note});
}

// --------------------------------------------------------------------------
// Integer compute (32-bit IMAD) — int / int2 / int4 / int8 / int16
// --------------------------------------------------------------------------
int OneapiPeak::runComputeInt32(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"integer_compute", "Integer compute (32-bit IMAD)", "gops", Category::Unknown,
     "Peak speed on 32-bit whole numbers -- the arithmetic behind indexing, "
     "addressing and bit manipulation, which kernels do alongside their "
     "fractional maths."});

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(int));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  int *out = sycl::malloc_device<int>(totalThreads, dev.stream);
  if (!out)
  {
    test.skipAll({"int", "int2", "int4", "int8", "int16"},
                 ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  const int A = 3;
  const unsigned int forced = forceIters ? specifiedIters : 0;
  runIntWidth<IntTag, 1 >(*this, dev, test, "int",   out, totalThreads, blockSize, 128, A, COMPUTE_FP_WORK_PER_WI, cfg.targetTimeUs, forced);
  runIntWidth<IntTag, 2 >(*this, dev, test, "int2",  out, totalThreads, blockSize, 128, A, COMPUTE_FP_WORK_PER_WI, cfg.targetTimeUs, forced);
  runIntWidth<IntTag, 4 >(*this, dev, test, "int4",  out, totalThreads, blockSize, 128, A, COMPUTE_FP_WORK_PER_WI, cfg.targetTimeUs, forced);
  runIntWidth<IntTag, 8 >(*this, dev, test, "int8",  out, totalThreads, blockSize, 128, A, COMPUTE_FP_WORK_PER_WI, cfg.targetTimeUs, forced);
  runIntWidth<IntTag, 16>(*this, dev, test, "int16", out, totalThreads, blockSize, 128, A, COMPUTE_FP_WORK_PER_WI, cfg.targetTimeUs, forced);

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// INT8 dot-product compute (DP4a-style).  Mirrors compute_int8_dp.hip.
//
// dp4(xp, yp, a): unpack 4 signed int8 lanes from packed ints xp, yp and
//   accumulate sum(xi*yi) into a.  4 muls + 4 adds = 8 ops.  On Intel HW the
//   compiler may fuse this to dp4a; on CPU it is honest int MACs.
//
// The chain shape.  Three constraints have to hold at once, and each one has
// already produced a wrong reading in some backend:
//
//  - Both multiplicands may not be loop-invariant.  a = dp4(x, y, a) with x
//    and y both fixed is a + n*dot(x, y), which a compiler may and does
//    strength-reduce; the OpenCL backend shipped that shape, and in Vulkan it
//    read 74939 GOPS on an RTX 5060 whose dp4a peak, measured by CUDA's own
//    __dp4a on the same card, is 33928.
//
//  - Nothing may run between the dots.  This kernel used to keep an operand
//    moving by rewriting it from the accumulator (y ^= a), but that XOR is a
//    second dependent integer op per dot and the op budget credits none of
//    it.  On an Arc A380 that mistake cost the Vulkan int8 chain more than
//    half its rate (8832 GOPS against 19497).
//
//  - All three source operands must be distinct registers.  Intel Alchemist
//    halves a three-source op that reads the same register twice, so
//    a = dp4(x, a, a) is not the answer either.
//
// What satisfies all three: two accumulators feeding each other.  Each dot
// reads {x, the other accumulator, its own} and writes its own, so a pair is
// one dependent chain, not two, and because the dot extracts the bytes of a
// value that is itself a 32-bit accumulator the recurrence is not affine and
// has no closed form to fold to.  NCH counts pairs, so the ILP ladder is
// 1/2/4/8 independent copies of the pair.
//
// Op accounting: 1024 dp4 calls per work-item, each 8 ops = 8192
// (COMPUTE_INT8_DP_WORK_PER_WI), the same for every variant.  One STEP2 is
// two dots, so every variant issues 512 of them over 64 outer iterations:
// the chain count sets the steps per chain (8/NCH), not the trip count.
// Accumulators are seeded 4 apart -- chains that start on the same value stay
// bitwise equal forever and a compiler is free to keep only one of them.
//
// NOTE: dp4 is spelled out rather than issued as one instruction, so the
// unpack shifts are work the 8-ops-per-dot budget does not credit either.
// That is a separate, larger accounting gap from the XOR removed here, and it
// is why this row is not directly comparable to the CUDA/ROCm ones.
// --------------------------------------------------------------------------
template <int NCH> class compute_int8_dp_kernel;

// The chain count divides the 8 STEP2 an outer iteration issues, so it has to
// divide 8 exactly or the variant silently reports against the wrong budget.
template <int NCH> struct Int8DpChains
{
  static_assert(NCH >= 1 && NCH <= 8 && 8 % NCH == 0,
                "int8-dp chain count must divide 8 (STEPS = 8 / NCH)");
  static constexpr int steps = 8 / NCH;
};

template <int NCH>
static void runInt8DpVariant(OneapiPeak &peak, OneapiDevice &dev,
                             logger::TestScope &test, const char *label,
                             int *out, uint64_t totalThreads, uint32_t blockSize,
                             int outerIters, int scalarA,
                             unsigned int targetTimeUs, unsigned int forced)
{
  const int A = scalarA;
  auto submit = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_int8_dp_kernel<NCH>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          auto dp4 = [](int xp, int yp, int a) {
            // sign-extend each 8-bit lane via arithmetic shift, multiply-add.
            a += ((xp << 24) >> 24) * ((yp << 24) >> 24);
            a += ((xp << 16) >> 24) * ((yp << 16) >> 24);
            a += ((xp <<  8) >> 24) * ((yp <<  8) >> 24);
            a += ( xp        >> 24) * ( yp        >> 24);
            return a;
          };

          int lid = (int)it.get_local_id(0);
          int x = (A & 0xff) | (((A + 1) & 0xff) << 8)
                | (((A + 2) & 0xff) << 16) | (((A + 3) & 0xff) << 24);

          // Chain c is the pair (p[c], q[c]); no two accumulators start equal.
          int p[NCH], q[NCH];
          #pragma unroll
          for (int c = 0; c < NCH; c++) { p[c] = lid + 8 * c; q[c] = lid + 8 * c + 4; }

          // 8 STEP2 per outer iteration in total, however the chains divide it.
          constexpr int STEPS = Int8DpChains<NCH>::steps;

          #pragma unroll 1
          for (int i = 0; i < outerIters; i++)
          {
            #pragma unroll
            for (int s = 0; s < STEPS; s++)
            {
              #pragma unroll
              for (int c = 0; c < NCH; c++)
              {
                // STEP2: two dots, one dependent chain, nothing beside them.
                p[c] = dp4(x, q[c], p[c]);
                q[c] = dp4(x, p[c], q[c]);
              }
            }
          }
          int acc = 0;
          #pragma unroll
          for (int c = 0; c < NCH; c++) acc += p[c] + q[c];
          out[it.get_global_id(0)] = acc;
        });
    });
  };

  const char *note = oneapiChainNote(NCH);
  float us = peak.runKernel(dev, submit, targetTimeUs, forced);
  if (us <= 0.0f) test.skip(label, ResultStatus::Error, "kernel launch failed", note);
  else            test.emit(label, clpeak_oneapi::computeGflops(totalThreads, COMPUTE_INT8_DP_WORK_PER_WI, us, 1e9),
                            {false, note});
}

int OneapiPeak::runComputeInt8DP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"integer_compute_int8_dp", "INT8 dot-product compute (DP4a)", "gops",
     Category::Unknown,
     "Peak speed of the 8-bit dot product, which multiplies four pairs of small "
     "whole numbers and sums them in one step -- the general-compute path for "
     "quantized (compressed) neural networks."});

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(int));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  int *out = sycl::malloc_device<int>(totalThreads, dev.stream);
  if (!out)
  {
    test.skipAll({"int8_dp", "int8_dp2", "int8_dp4", "int8_dp8"},
                 ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  const int A = 4;
  const unsigned int forced = forceIters ? specifiedIters : 0;
  // 64 outer iters for every variant: the chain count divides the 8 STEP2 per
  // iteration between the chains, so all four total 8192 ops/WI.
  runInt8DpVariant<1>(*this, dev, test, "int8_dp",  out, totalThreads, blockSize, 64, A, cfg.targetTimeUs, forced);
  runInt8DpVariant<2>(*this, dev, test, "int8_dp2", out, totalThreads, blockSize, 64, A, cfg.targetTimeUs, forced);
  runInt8DpVariant<4>(*this, dev, test, "int8_dp4", out, totalThreads, blockSize, 64, A, cfg.targetTimeUs, forced);
  runInt8DpVariant<8>(*this, dev, test, "int8_dp8", out, totalThreads, blockSize, 64, A, cfg.targetTimeUs, forced);

  sycl::free(out, dev.stream);
  return 0;
}



#endif // ENABLE_ONEAPI
