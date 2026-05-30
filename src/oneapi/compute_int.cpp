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
// One IMAD_16 = 16 mul-adds = 32 int ops per lane.  Width-invariant total:
// width W runs baseIters/W iters * 32*W ops = baseIters*32 ops/WI.
// baseIters=128 -> 4096 (COMPUTE_FP_WORK_PER_WI, matches ROCm int32).
#define IMAD_4(x, y)  x = y * x + y; y = x * y + x; x = y * x + y; y = x * y + x;
#define IMAD_16(x, y) IMAD_4(x, y) IMAD_4(x, y) IMAD_4(x, y) IMAD_4(x, y)

namespace { struct IntTag; }
template <typename Tag, int W> class compute_int_vec_kernel;

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
          VecT x, y;
          #pragma unroll
          for (int k = 0; k < W; k++)
          {
            x[k] = A + k;
            y[k] = (int)it.get_local_id(0) + k;
          }
          #pragma unroll 1
          for (int i = 0; i < iters; i++) { IMAD_16(x, y) }
          int acc = 0;
          #pragma unroll
          for (int k = 0; k < W; k++) acc += y[k];
          out[it.get_global_id(0)] = acc;
        });
    });
  };

  float us = peak.runKernel(dev, submit, targetTimeUs, forced);
  if (us <= 0.0f) test.skip(label, ResultStatus::Error, "kernel launch failed");
  else            test.emit(label, clpeak_oneapi::computeGflops(totalThreads, workPerWI, us, 1e9));
}

// --------------------------------------------------------------------------
// Integer compute (32-bit IMAD) — int / int2 / int4 / int8 / int16
// --------------------------------------------------------------------------
int OneapiPeak::runComputeInt32(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"integer_compute", "Integer compute (32-bit IMAD)", "gops"});

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
// INT8 dot-product compute (DP4a-style).  Mirrors compute_int8_dp.hip:
// each STEP does a 4-lane int8 dot-accumulate into an int32, then feeds the
// accumulator back into an operand (y ^= a) so the chain CANNOT be hoisted
// (this is the fix for the bogus loop-invariant version).
//
// dp4(xp, yp, a): unpack 4 signed int8 lanes from packed ints xp, yp and
//   accumulate sum(xi*yi) into a.  4 muls + 4 adds = 8 ops.
// STEP_16 = 16 dp4 = 128 ops.  All variants total 8192 ops/WI
// (COMPUTE_INT8_DP_WORK_PER_WI); they differ only in ILP (chain count):
//   dp:  64 iters * 1 chain, dp2: 32 * 2, dp4: 16 * 4, dp8: 8 * 8.
// On Intel HW the compiler may fuse to dp4a; on CPU it is honest int MACs.
// --------------------------------------------------------------------------
template <int NCH> class compute_int8_dp_kernel;

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

          int y[NCH], a[NCH];
          #pragma unroll
          for (int c = 0; c < NCH; c++) { y[c] = lid + c; a[c] = lid + 7 * c; }

          #pragma unroll 1
          for (int i = 0; i < outerIters; i++)
          {
            #pragma unroll
            for (int c = 0; c < NCH; c++)
            {
              // STEP_16: 16 dot-accumulates with feedback per chain
              #pragma unroll
              for (int s = 0; s < 16; s++)
              {
                a[c] = dp4(x, y[c], a[c]);
                y[c] ^= a[c];
              }
            }
          }
          int acc = 0;
          #pragma unroll
          for (int c = 0; c < NCH; c++) acc += a[c];
          out[it.get_global_id(0)] = acc;
        });
    });
  };

  float us = peak.runKernel(dev, submit, targetTimeUs, forced);
  if (us <= 0.0f) test.skip(label, ResultStatus::Error, "kernel launch failed");
  else            test.emit(label, clpeak_oneapi::computeGflops(totalThreads, COMPUTE_INT8_DP_WORK_PER_WI, us, 1e9));
}

int OneapiPeak::runComputeInt8DP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"integer_compute_int8_dp", "INT8 dot-product compute (DP4a)", "gops"});

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
  // outerIters scaled by chain count so every variant totals 8192 ops/WI.
  runInt8DpVariant<1>(*this, dev, test, "int8_dp",  out, totalThreads, blockSize, 64, A, cfg.targetTimeUs, forced);
  runInt8DpVariant<2>(*this, dev, test, "int8_dp2", out, totalThreads, blockSize, 32, A, cfg.targetTimeUs, forced);
  runInt8DpVariant<4>(*this, dev, test, "int8_dp4", out, totalThreads, blockSize, 16, A, cfg.targetTimeUs, forced);
  runInt8DpVariant<8>(*this, dev, test, "int8_dp8", out, totalThreads, blockSize,  8, A, cfg.targetTimeUs, forced);

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Packed INT4 compute (emulated — same shape as compute_int4_packed.hip).
// Real x<->y dependency chain already; single metric (matches ROCm).
// 64 outer iters * MAD_16 = 1024 MACs; reported as 4096 ops/WI.
// --------------------------------------------------------------------------
class compute_int4_packed_kernel;

int OneapiPeak::runComputeInt4Packed(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"int4_packed_compute", "Packed INT4 compute (emulated)", "gops"});

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(int));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  int *out = sycl::malloc_device<int>(totalThreads, dev.stream);
  if (!out)
  {
    test.skip("int4_packed", ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }
  const int A = 3;

  auto submit = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_int4_packed_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          int x = A;
          int y = (int)it.get_local_id(0);
          auto MAC = [](int &dst, int src) {
            int _d = dst, _s = src;
            int _dl = (_d << 28) >> 28;
            int _dh = _d >> 4;
            int _sl = (_s << 28) >> 28;
            int _sh = _s >> 4;
            _dl = _sl * _dl + _sl;
            _dh = _sh * _dh + _sh;
            dst = ((_dl & 0x0F) | ((_dh & 0x0F) << 4));
          };
          #pragma unroll 1
          for (int i = 0; i < 64; i++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
              MAC(x, y); MAC(y, x); MAC(x, y); MAC(y, x);
            }
          }
          out[it.get_global_id(0)] = x + y;
        });
    });
  };

  float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
    test.skip("int4_packed", ResultStatus::Error, "kernel launch failed");
  else
    test.emit("int4_packed", clpeak_oneapi::computeGflops(totalThreads, COMPUTE_INT4_PACKED_WORK_PER_WI, us, 1e9));

  sycl::free(out, dev.stream);
  return 0;
}

#endif // ENABLE_ONEAPI
