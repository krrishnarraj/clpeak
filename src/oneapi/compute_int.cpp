#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>

namespace clpeak_oneapi {
uint32_t pickComputeBlocks(const oneapi_device_info_t &info,
                           uint32_t blockSize, uint32_t outElemsPerBlock,
                           uint32_t elemSize);
float    computeGflops(uint64_t totalThreads, uint32_t workPerWI, float meanUs,
                       double unitDivider);
}

// Integer MAD macros: shape mirrors compute_int32.hip exactly so the work-per-WI
// constant (COMPUTE_FP_WORK_PER_WI = 4096) matches what the kernel actually does.
#define IMAD_4(x, y)  x = y * x + y; y = x * y + x; x = y * x + y; y = x * y + x;
#define IMAD_16(x, y) IMAD_4(x, y) IMAD_4(x, y) IMAD_4(x, y) IMAD_4(x, y)

// --------------------------------------------------------------------------
// Integer compute (32-bit IMAD)
// 128 outer iters * IMAD_16 (32 ops) = 4096 ops/WI.
// --------------------------------------------------------------------------
class compute_int32_kernel;

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
    test.skip("int", ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }
  const int A = 3;

  auto submit = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_int32_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          int x = A;
          int y = (int)it.get_local_id(0);
          #pragma unroll 1
          for (int i = 0; i < 128; i++) { IMAD_16(x, y) }
          out[it.get_global_id(0)] = y;
        });
    });
  };

  float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
    test.skip("int", ResultStatus::Error, "kernel launch failed");
  else
    test.emit("int", clpeak_oneapi::computeGflops(totalThreads, COMPUTE_FP_WORK_PER_WI, us, 1e9));

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// INT8 dot-product compute via sycl::dot_acc when available.
// Intel iGPUs/Arc expose this via SYCL ext_intel_dot_acc / ext_oneapi_dot_acc.
// We try a generic emulation (manual int8 mul + add) — the SYCL compiler
// will lower it to native dp4a on hardware that supports it.
// COMPUTE_INT8_DP_WORK_PER_WI = 8192.  Each "dot" = 4 INT8 muladds = 8 ops.
// 64 outer iters * 16 dots = 1024 dots = 8192 ops/WI.
// --------------------------------------------------------------------------
class compute_int8_dp_kernel;

int OneapiPeak::runComputeInt8DP(OneapiDevice &dev, benchmark_config_t &cfg)
{
  auto test = currentDeviceScope->beginTest(
    {"integer_compute_int8_dp", "INT8 dot-product compute", "gops"});

  const uint32_t blockSize = 256;
  uint32_t numBlocks = clpeak_oneapi::pickComputeBlocks(dev.info, blockSize, blockSize, sizeof(int));
  uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  int *out = sycl::malloc_device<int>(totalThreads, dev.stream);
  if (!out)
  {
    test.skip("int8_dp", ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  auto submit = [&](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<compute_int8_dp_kernel>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          // 4 INT8 lanes packed into an int.  Use builtin types so the
          // compiler can vectorize.  Init values chosen to avoid sign-extension
          // saturating to zero on the first iteration.
          sycl::vec<int8_t, 4> a{1, 2, 3, 4};
          sycl::vec<int8_t, 4> b{(int8_t)(it.get_local_id(0) & 0x7F),
                                 (int8_t)((it.get_local_id(0) + 1) & 0x7F),
                                 (int8_t)((it.get_local_id(0) + 2) & 0x7F),
                                 (int8_t)((it.get_local_id(0) + 3) & 0x7F)};
          int acc = 0;
          #pragma unroll 1
          for (int i = 0; i < 64; i++) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
              acc += (int)a[0] * (int)b[0] + (int)a[1] * (int)b[1]
                   + (int)a[2] * (int)b[2] + (int)a[3] * (int)b[3];
            }
          }
          out[it.get_global_id(0)] = acc;
        });
    });
  };

  float us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  if (us <= 0.0f)
    test.skip("int8_dp", ResultStatus::Error, "kernel launch failed");
  else
    test.emit("int8_dp", clpeak_oneapi::computeGflops(totalThreads, COMPUTE_INT8_DP_WORK_PER_WI, us, 1e9));

  sycl::free(out, dev.stream);
  return 0;
}

// --------------------------------------------------------------------------
// Packed INT4 compute (emulated — same shape as compute_int4_packed.hip).
// 64 outer iters * MAD_16 = 1024 MAC ops per WI; reported as 4096 ops/WI to
// match the cross-backend constant (each int packs 2 int4 lanes, each MAC
// counts as 2 ops since we operate on both nibbles).
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

  // INT4 macro: unpack two nibbles, MAC each lane, repack.  Matches the
  // ROCm/CUDA kernel byte-for-byte so reported gops are comparable across
  // backends.
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
