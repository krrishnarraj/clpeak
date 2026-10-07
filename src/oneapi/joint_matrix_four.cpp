#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <sycl/sycl.hpp>

#include <cstdint>
#include <type_traits>

#ifdef CLPEAK_ONEAPI_HAS_JOINT_MATRIX
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#include <sycl/ext/oneapi/bfloat16.hpp>
#define CLPEAK_ONEAPI_JM_HAS_BF16 1
#endif

// A joint_matrix tile timed with four accumulators -- the second shape
// runJointMatrix (joint_matrix.cpp) races against its accumulator chain.
//
// In a file of its own because of how DPC++ packages device code: the kernels
// of one source file that use the same matrix types and tile shape go into one
// device image, and the SYCL runtime builds an image whole, the first time one
// of its kernels launches.  Beside the chain, a driver that could not build
// this kernel would fail the chain's build with it, and lose a reading that
// never needed this one.  A kernel from another file gets an image of its own.

namespace {

namespace syclex = sycl::ext::oneapi::experimental::matrix;

// Per-instantiation kernel-name tag, as JmTag is for the chain.
template <typename At, typename Bt, typename Acc, int M, int N, int K> struct JmTag4;

// A fill a step away from `v`, so the four A tiles and the four B tiles of
// runJmVariant4 all differ and every product it issues is a distinct pair.
template <typename T>
static T jmFillStep(T v, int k)
{
  if constexpr (std::is_integral_v<T>)
    return (T)(v + k);
  else
    return (T)((float)v + 0.25f * (float)k);
}

// The chain in joint_matrix.cpp (runJmVariant) is one dependent run of
// joint_matrix_mad, and Alchemist forwards no accumulator from one DPAS to the
// next (IGC's hasDpasFwdAndDoubleSrcReadSupression is Xe3P-only), so each mad
// waits out the one before it.  What IGC does issue back to back is a run of
// DPAS that write different accumulators and read the same B tile
// (canInSameDPASMacro).  An Arc A380 (driver 8993) shows the cost: its 32x32
// bf16 tile, sixteen independent 8x8 DPAS a step, read 18.7 TFLOPS where the
// 8x8 chain read 15.5, and oneMKL's int8 GEMM read 37.9 TOPS where the chain
// read 31.0.
//
// The shape is Vulkan's four-accumulator coopmat run (CM_CHAIN_ALT in
// src/vulkan/shaders/coopmat_chain.glsl): four A tiles and four B tiles,
// sixteen distinct products a trip, each B tile meeting all four accumulators
// in a row, and a trip count passed in at run time so nothing lets the
// compiler unroll or fold the run.  joint_matrix.cpp passes JM_ITERS / 16
// trips, the same 256 mads a sub-group as its chain.  All four tiles are
// stored, so a block takes four tiles of the buffer.
template <typename At, typename Bt, typename Acc, int M, int N, int K,
          typename FillA, typename FillB>
static float runJmVariant4(OneapiPeak &peak, OneapiDevice &dev,
                           Acc *outBuf, uint32_t *sgCountBuf,
                           uint32_t numBlocks, uint32_t blockSize,
                           FillA aFill, FillB bFill, uint32_t trips,
                           unsigned int targetTimeUs, unsigned int forced)
{
  const uint64_t totalThreads = (uint64_t)numBlocks * blockSize;
  const FillA a0f = jmFillStep(aFill, 0), a1f = jmFillStep(aFill, 1),
              a2f = jmFillStep(aFill, 2), a3f = jmFillStep(aFill, 3);
  const FillB b0f = jmFillStep(bFill, 0), b1f = jmFillStep(bFill, 1),
              b2f = jmFillStep(bFill, 2), b3f = jmFillStep(bFill, 3);

  auto submit = [=](sycl::queue &q) -> sycl::event {
    return q.submit([&](sycl::handler &h) {
      h.parallel_for<JmTag4<At, Bt, Acc, M, N, K>>(
        sycl::nd_range<1>(totalThreads, blockSize),
        [=](sycl::nd_item<1> it) {
          auto sg = it.get_sub_group();
          using MatA = syclex::joint_matrix<sycl::sub_group, At, syclex::use::a, M, K,
                                            syclex::layout::row_major>;
          using MatB = syclex::joint_matrix<sycl::sub_group, Bt, syclex::use::b, K, N,
                                            syclex::layout::ext_intel_packed>;
          using MatC = syclex::joint_matrix<sycl::sub_group, Acc, syclex::use::accumulator, M, N>;
          MatA a0, a1, a2, a3;
          MatB b0, b1, b2, b3;
          MatC c0, c1, c2, c3;
          syclex::joint_matrix_fill(sg, a0, a0f);
          syclex::joint_matrix_fill(sg, a1, a1f);
          syclex::joint_matrix_fill(sg, a2, a2f);
          syclex::joint_matrix_fill(sg, a3, a3f);
          syclex::joint_matrix_fill(sg, b0, b0f);
          syclex::joint_matrix_fill(sg, b1, b1f);
          syclex::joint_matrix_fill(sg, b2, b2f);
          syclex::joint_matrix_fill(sg, b3, b3f);
          syclex::joint_matrix_fill(sg, c0, (Acc)0);
          syclex::joint_matrix_fill(sg, c1, (Acc)0);
          syclex::joint_matrix_fill(sg, c2, (Acc)0);
          syclex::joint_matrix_fill(sg, c3, (Acc)0);
          #pragma unroll 1
          for (uint32_t i = 0; i < trips; i++)
          {
            syclex::joint_matrix_mad(sg, c0, a0, b0, c0);
            syclex::joint_matrix_mad(sg, c1, a1, b0, c1);
            syclex::joint_matrix_mad(sg, c2, a2, b0, c2);
            syclex::joint_matrix_mad(sg, c3, a3, b0, c3);
            syclex::joint_matrix_mad(sg, c0, a1, b1, c0);
            syclex::joint_matrix_mad(sg, c1, a2, b1, c1);
            syclex::joint_matrix_mad(sg, c2, a3, b1, c2);
            syclex::joint_matrix_mad(sg, c3, a0, b1, c3);
            syclex::joint_matrix_mad(sg, c0, a2, b2, c0);
            syclex::joint_matrix_mad(sg, c1, a3, b2, c1);
            syclex::joint_matrix_mad(sg, c2, a0, b2, c2);
            syclex::joint_matrix_mad(sg, c3, a1, b2, c3);
            syclex::joint_matrix_mad(sg, c0, a3, b3, c0);
            syclex::joint_matrix_mad(sg, c1, a0, b3, c1);
            syclex::joint_matrix_mad(sg, c2, a1, b3, c2);
            syclex::joint_matrix_mad(sg, c3, a2, b3, c3);
          }

          // JM_SG_COUNT_NOTE in joint_matrix.cpp: the host counts the ops per
          // sub-group, from the count reported here.
          const uint32_t sgPerWG = (uint32_t)sg.get_group_range()[0];
          if (it.get_global_linear_id() == 0) *sgCountBuf = sgPerWG;

          Acc *blockOut = outBuf + (size_t)it.get_group(0) * (size_t)(4 * M * N);
          auto store = [&](MatC &c, size_t slot) {
            syclex::joint_matrix_store(sg, c,
              sycl::address_space_cast<sycl::access::address_space::global_space,
                                       sycl::access::decorated::no>(blockOut + slot * M * N),
              N, syclex::layout::row_major);
          };
          store(c0, 0);
          store(c1, 1);
          store(c2, 2);
          store(c3, 3);
        });
    });
  };
  return peak.runKernel(dev, submit, targetTimeUs, forced);
}

// The shapes built with four accumulators: the ones whose four fit
// JM_FOUR_ACC_BYTES_PER_LANE (joint_matrix.cpp) at the sub-group widths Xe
// offers.
#define CLPEAK_JM_SHAPES4(X)  X(8, 8)  X(8, 16)

template <typename At, typename Bt, typename Acc, int K, typename FillA, typename FillB>
static float runJmShape4(OneapiPeak &peak, OneapiDevice &dev,
                         void *out, uint32_t *sgCount,
                         uint32_t numBlocks, uint32_t blockSize,
                         FillA aFill, FillB bFill, uint32_t M, uint32_t N,
                         uint32_t trips, unsigned int targetTimeUs, unsigned int forced)
{
#define CLPEAK_JM_CASE4(MM, NN)                                                 \
  if (M == (MM) && N == (NN))                                                   \
    return runJmVariant4<At, Bt, Acc, MM, NN, K, FillA, FillB>(                  \
        peak, dev, (Acc *)out, sgCount, numBlocks, blockSize,                    \
        aFill, bFill, trips, targetTimeUs, forced);
  CLPEAK_JM_SHAPES4(CLPEAK_JM_CASE4)
#undef CLPEAK_JM_CASE4
  return -1.0f;
}

} // namespace

namespace clpeak_oneapi {

// One tile with four accumulators, `trips` trips of sixteen mads: the
// microseconds a launch took (OneapiPeak::runKernel), or -1 when it failed or
// the tile has no instantiation here.  The types are dispatched as runJmFp
// and runJmInt dispatch the chain's, and filled alike.
float runJmFour(OneapiPeak &peak, OneapiDevice &dev, void *out, uint32_t *sgCount,
                uint32_t numBlocks, uint32_t blockSize,
                syclex::matrix_type at, syclex::matrix_type bt, syclex::matrix_type ct,
                uint32_t M, uint32_t N, uint32_t K, uint32_t trips,
                unsigned int targetTimeUs, unsigned int forced)
{
  using mt = syclex::matrix_type;
  if (ct == mt::fp32)
  {
#ifdef CLPEAK_ONEAPI_JM_HAS_BF16
    using bfloat16 = sycl::ext::oneapi::bfloat16;
    if (at == mt::bf16 && bt == mt::bf16 && K == 16)
      return runJmShape4<bfloat16, bfloat16, float, 16>(
          peak, dev, out, sgCount, numBlocks, blockSize,
          (bfloat16)1.f, (bfloat16)1.f, M, N, trips, targetTimeUs, forced);
#endif
    if (at == mt::fp16 && bt == mt::fp16 && K == 16)
      return runJmShape4<sycl::half, sycl::half, float, 16>(
          peak, dev, out, sgCount, numBlocks, blockSize,
          (sycl::half)1.f, (sycl::half)1.f, M, N, trips, targetTimeUs, forced);
    if (at == mt::tf32 && bt == mt::tf32 && K == 8)
      return runJmShape4<syclex::precision::tf32, syclex::precision::tf32, float, 8>(
          peak, dev, out, sgCount, numBlocks, blockSize,
          1.f, 1.f, M, N, trips, targetTimeUs, forced);
    return -1.0f;
  }
  if (ct != mt::sint32 || K != 32)
    return -1.0f;
  if (at == mt::sint8 && bt == mt::sint8)
    return runJmShape4<int8_t, int8_t, int32_t, 32>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        (int8_t)1, (int8_t)1, M, N, trips, targetTimeUs, forced);
  if (at == mt::uint8 && bt == mt::uint8)
    return runJmShape4<uint8_t, uint8_t, int32_t, 32>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        (uint8_t)1, (uint8_t)1, M, N, trips, targetTimeUs, forced);
  if (at == mt::uint8 && bt == mt::sint8)
    return runJmShape4<uint8_t, int8_t, int32_t, 32>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        (uint8_t)1, (int8_t)1, M, N, trips, targetTimeUs, forced);
  if (at == mt::sint8 && bt == mt::uint8)
    return runJmShape4<int8_t, uint8_t, int32_t, 32>(
        peak, dev, out, sgCount, numBlocks, blockSize,
        (int8_t)1, (uint8_t)1, M, N, trips, targetTimeUs, forced);
  return -1.0f;
}

} // namespace clpeak_oneapi

#endif // CLPEAK_ONEAPI_HAS_JOINT_MATRIX
#endif // ENABLE_ONEAPI
