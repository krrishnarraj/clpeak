#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>

#ifdef CLPEAK_ONEAPI_HAS_JOINT_MATRIX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#include <sycl/ext/oneapi/bfloat16.hpp>
#endif
#endif

// XMX matrix engine peak — Intel's analog of rocWMMA / cuda WMMA / Vulkan
// coopMatrix / Metal simdgroup_matrix.  We pick an MxNxK tile that maps to
// the underlying hardware:  8x16x16 BF16->FP32 on Xe-HPG/Battlemage,
// 8x16x32 INT8->INT32 on the same.
//
// Each sub-group computes (Iters) MMA operations on its own accumulator;
// the per-sub-group ops budget is M*N*K*2*Iters (multiply-add counted as 2).

class joint_matrix_fp16_kernel;
class joint_matrix_int8_kernel;

int OneapiPeak::runJointMatrix(OneapiDevice &dev, benchmark_config_t &cfg, Category category)
{
  const bool isInt = (category == Category::IntCompute);
  auto test = currentDeviceScope->beginTest(
    {isInt ? "joint-matrix-int" : "joint-matrix-fp",
     isInt ? "joint_matrix int8xint8+int32 8x16x32"
           : "joint_matrix bf16xbf16+fp32 8x16x16",
     isInt ? "tops" : "tflops"});

  const char *metric = isInt ? "joint_matrix_int8" : "joint_matrix_bf16";

#ifndef CLPEAK_ONEAPI_HAS_JOINT_MATRIX
  test.skip(metric, ResultStatus::Unsupported,
            "joint_matrix header not available in this oneAPI toolchain");
  return 0;
#else
  if (!dev.info.xmxSupported)
  {
    test.skip(metric, ResultStatus::Unsupported,
              "XMX matrix engine not available on this device");
    return 0;
  }

  namespace syclex = sycl::ext::oneapi::experimental::matrix;

  const uint32_t SG  = dev.info.preferredSubGroupSize ? dev.info.preferredSubGroupSize : 16;
  const uint32_t blockSize = SG;
  uint64_t globalThreads = targetGlobalThreads((uint32_t)dev.info.numCUs);

  constexpr uint32_t M = 8;
  constexpr uint32_t N = 16;
  constexpr uint32_t K_FP = 16;
  constexpr uint32_t K_INT = 32;
  constexpr uint32_t Iters = 256;
  const uint32_t K = isInt ? K_INT : K_FP;

  uint64_t wantBlocks = globalThreads / blockSize;
  uint64_t bytesPerBlock = (uint64_t)M * N * sizeof(float);
  uint64_t maxBlocks = dev.info.totalGlobalMem / 4 / bytesPerBlock;
  uint64_t pickBlocks = (wantBlocks < maxBlocks) ? wantBlocks : maxBlocks;
  if (pickBlocks == 0) pickBlocks = 1;
  uint32_t numBlocks = (uint32_t)pickBlocks;
  const uint64_t outElems = (uint64_t)numBlocks * M * N;
  const uint64_t outBytes = outElems * (isInt ? sizeof(int32_t) : sizeof(float));
  const uint64_t totalThreads = (uint64_t)numBlocks * blockSize;

  void *outBuf = sycl::malloc_device(outBytes, dev.stream);
  if (!outBuf)
  {
    test.skip(metric, ResultStatus::Error, "Failed to allocate output buffer");
    return -1;
  }

  float us = -1.0f;

#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
  using bfloat16 = sycl::ext::oneapi::bfloat16;
  if (!isInt)
  {
    float *out = (float *)outBuf;
    auto submit = [=](sycl::queue &q) -> sycl::event {
      return q.submit([&](sycl::handler &h) {
        h.parallel_for<joint_matrix_fp16_kernel>(
          sycl::nd_range<1>(totalThreads, blockSize),
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            syclex::joint_matrix<sycl::sub_group, bfloat16, syclex::use::a, M, K_FP, syclex::layout::row_major> a;
            syclex::joint_matrix<sycl::sub_group, bfloat16, syclex::use::b, K_FP, N, syclex::layout::row_major> b;
            syclex::joint_matrix<sycl::sub_group, float, syclex::use::accumulator, M, N> c;
            syclex::joint_matrix_fill(sg, a, bfloat16(1.0f));
            syclex::joint_matrix_fill(sg, b, bfloat16(1.0f));
            syclex::joint_matrix_fill(sg, c, 0.0f);
            #pragma unroll 1
            for (int i = 0; i < (int)Iters; i++)
              syclex::joint_matrix_mad(sg, c, a, b, c);

            float *blockOut = out + (size_t)it.get_group(0) * M * N;
            syclex::joint_matrix_store(sg, c,
              sycl::address_space_cast<sycl::access::address_space::global_space,
                                       sycl::access::decorated::no>(blockOut),
              N, syclex::layout::row_major);
          });
      });
    };
    us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  }
  else
#endif
  if (isInt)
  {
    int32_t *out = (int32_t *)outBuf;
    auto submit = [=](sycl::queue &q) -> sycl::event {
      return q.submit([&](sycl::handler &h) {
        h.parallel_for<joint_matrix_int8_kernel>(
          sycl::nd_range<1>(totalThreads, blockSize),
          [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            auto sg = it.get_sub_group();
            syclex::joint_matrix<sycl::sub_group, int8_t,  syclex::use::a, M, K_INT, syclex::layout::row_major> a;
            syclex::joint_matrix<sycl::sub_group, int8_t,  syclex::use::b, K_INT, N, syclex::layout::row_major> b;
            syclex::joint_matrix<sycl::sub_group, int32_t, syclex::use::accumulator, M, N> c;
            syclex::joint_matrix_fill(sg, a, (int8_t)1);
            syclex::joint_matrix_fill(sg, b, (int8_t)1);
            syclex::joint_matrix_fill(sg, c, 0);
            #pragma unroll 1
            for (int i = 0; i < (int)Iters; i++)
              syclex::joint_matrix_mad(sg, c, a, b, c);

            int32_t *blockOut = out + (size_t)it.get_group(0) * M * N;
            syclex::joint_matrix_store(sg, c,
              sycl::address_space_cast<sycl::access::address_space::global_space,
                                       sycl::access::decorated::no>(blockOut),
              N, syclex::layout::row_major);
          });
      });
    };
    us = runKernel(dev, submit, cfg.targetTimeUs, forceIters ? specifiedIters : 0);
  }
  else
  {
    test.skip(metric, ResultStatus::Unsupported,
              "SYCL bfloat16 header missing — bf16 joint_matrix variant unavailable");
    sycl::free(outBuf, dev.stream);
    return 0;
  }

  if (us <= 0.0f)
  {
    test.skip(metric, ResultStatus::Error, "kernel launch failed");
    sycl::free(outBuf, dev.stream);
    return 0;
  }

  // Per-sub-group ops: M*N*K*2*Iters.  numBlocks == sub-groups (1 SG/group).
  const double ops = (double)numBlocks * (double)M * (double)N *
                     (double)K * 2.0 * (double)Iters;
  float value = (float)(ops * 1.0e6 / us / 1.0e12);
  test.emit(metric, value);

  sycl::free(outBuf, dev.stream);
  return 0;
#endif
}

#endif // ENABLE_ONEAPI
