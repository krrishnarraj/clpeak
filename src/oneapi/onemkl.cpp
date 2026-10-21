#ifdef ENABLE_ONEAPI

#include <oneapi/oneapi_peak.h>
#include <common/common.h>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <functional>
#include <string>
#include <vector>

#if __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
#include <sycl/ext/oneapi/bfloat16.hpp>
#define CLPEAK_ONEMKL_HAS_BF16 1
#endif

#ifdef CLPEAK_ONEAPI_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

// oneMKL GEMM peak — analog of rocBLAS / cuBLAS / MPSGraph.  FP category
// reports flops for FP32, FP64, FP16, BF16 (each gated by device aspect);
// INT category reports ops for INT8 (32-bit totals, through gemm_bias or
// gemm's int8 overload, whichever runs faster).  This mirrors the datatype
// coverage of the joint_matrix microbenchmark so every SDK-supported GEMM
// dtype is measured by both the raw and the library path.
//
// Sizing matches the ROCm rocBLAS benchmark: D = round_up_256(2048 + 128*CUs),
// clamped to [2048, 16384], halved while 3*D*D*8 > totalGlobalMem/4.

static uint32_t pickOnemklGemmDim(const oneapi_device_info_t &info)
{
  uint32_t cus = (uint32_t)(info.numCUs > 0 ? info.numCUs : 16);
  uint64_t D = 2048 + (uint64_t)cus * 128;
  D = (D + 255) & ~uint64_t(255);
  if (D < 2048)  D = 2048;
  if (D > 16384) D = 16384;
  uint64_t budget = info.totalGlobalMem ? info.totalGlobalMem / 4 : ((uint64_t)4 << 30);
  while (D > 1024 && 3ULL * D * D * 8 > budget)
    D /= 2;
  return (uint32_t)D;
}

int OneapiPeak::runOnemkl(OneapiDevice &dev, benchmark_config_t &)
{

  // One note per dtype row, shared by every emit and skip path below.
  const char *fp32Note = "fp32 inputs and accumulator, without the matrix "
                         "engine.";
  const char *fp64Note = "fp64 inputs and accumulator, on a smaller matrix than "
                         "the other rows.";
  const char *fp16Note = "fp16 inputs, fp16 or fp32 output.";
  const char *bf16Note = "bf16 inputs, fp32 accumulator.";
  const char *int8Note = "int8 inputs, int32 accumulator.";

  auto test = currentDeviceScope->beginTest(
        {"onemkl_gemm", "oneMKL GEMM peak", "flops", Category::Unknown,
         "Peak GEMM rate through Intel's oneMKL library, on a large square problem.",
         TestShape::Heterogeneous, "data type"});

  auto mklOpts = [&](const char *note) {
    logger::EmitOptions o;
    if (note) o.description = note;
    return o;
  };
  auto intOpts = [&](const char *note) {
    logger::EmitOptions o;
    if (note) o.description = note;
    o.unit = "ops";
    return o;
  };

#ifndef CLPEAK_ONEAPI_HAS_ONEMKL
  test.skip("fp32", ResultStatus::Unsupported, "oneMKL not found at configure time", mklOpts(fp32Note));
  test.skip("fp64", ResultStatus::Unsupported, "oneMKL not found at configure time", mklOpts(fp64Note));
  test.skip("fp16", ResultStatus::Unsupported, "oneMKL not found at configure time", mklOpts(fp16Note));
  test.skip("bf16", ResultStatus::Unsupported, "oneMKL not found at configure time", mklOpts(bf16Note));
  test.skip("int8", ResultStatus::Unsupported, "oneMKL not found at configure time", intOpts(int8Note));
  return 0;
#else
  namespace mkl = oneapi::mkl;
  using mkl::transpose;
  const std::int64_t D = pickOnemklGemmDim(dev.info);

  // fp64 throughput on most GPUs is a small fraction of fp32 (often 1/16..1/64+,
  // measured ~1/16 on Arc-class parts).  A full-size fp64 GEMM can therefore run
  // for many seconds in a SINGLE call and trip the GPU watchdog, surfacing as
  // CL_OUT_OF_RESOURCES.  Shrink the fp64 tile so one call stays short;
  // pickIters() then runs proportionally more iterations to still fill the 5 s
  // budget, so the measured peak is unaffected (a 3584^3 GEMM still saturates).
  const std::int64_t fp64Dim = std::max<std::int64_t>(1024, D / 4);

  // One way to put a dtype's GEMM to oneMKL: an overload and a layout.
  // issue(q, dA, dB, dC, dCo, dim) issues one square dim*dim*dim GEMM (dCo is
  // the int8 bias buffer, unused by the other forms).  Buffers are sized at
  // fp64 width and reinterpreted per form.
  using GemmIssue = std::function<void(sycl::queue &, void *, void *, void *, void *, std::int64_t)>;
  struct GemmForm
  {
    std::string name;
    GemmIssue issue;
  };

  // Run ONE GEMM dtype in full isolation: its own context + queue + buffers,
  // all torn down before the next dtype.  Two reasons:
  //  1. A faulting GEMM (e.g. fp64 returning a *sticky* CL_OUT_OF_RESOURCES on
  //     some drivers) corrupts only this disposable context, so the shared
  //     dev.stream stays healthy for every later benchmark.
  //  2. Per-dtype isolation means each dtype reports its own pass/fail instead
  //     of one bad dtype poisoning all the rest — a precise signal for the
  //     driver team about exactly which GEMM dtype faults.
  //
  // A dtype with several forms races them, as cuBLASLt's and hipBLASLt's
  // candidate algorithms are raced (cuda_blas.cpp): each form runs its warm-up
  // -- the first call builds its kernel, so it is never timed -- and then a
  // short probe, and the fastest is timed over the 5 s budget.  oneMKL picks
  // the kernel, and nothing in its interface says which overload and layout
  // reach its fastest one on a given GPU.
  auto measure = [&](const char *label, std::int64_t dim, const std::vector<GemmForm> &forms,
                     logger::EmitOptions opts) {
    const size_t cells = (size_t)dim * (size_t)dim;
    const double flops = 2.0 * (double)dim * (double)dim * (double)dim;
    const char *unit = opts.unit.empty() ? "flops" : opts.unit.c_str();
    sycl::queue q = [&]() -> sycl::queue {
      try
      {
        return sycl::queue(sycl::context(dev.dev), dev.dev,
                           sycl::property::queue::in_order{});
      }
      catch (const std::exception &e)
      {
        CLPEAK_VLOG("oneMKL %s: private context create failed (%s); shared queue\n",
                    label, e.what());
        return dev.stream;
      }
    }();

    void *dA  = sycl::malloc_device(cells * sizeof(double), q);
    void *dB  = sycl::malloc_device(cells * sizeof(double), q);
    void *dC  = sycl::malloc_device(cells * sizeof(double), q);
    void *dCo = sycl::malloc_device(sizeof(std::int32_t), q);  // int8 bias
    auto freeAll = [&]() {
      if (dA)  { try { sycl::free(dA,  q); } catch (...) {} }
      if (dB)  { try { sycl::free(dB,  q); } catch (...) {} }
      if (dC)  { try { sycl::free(dC,  q); } catch (...) {} }
      if (dCo) { try { sycl::free(dCo, q); } catch (...) {} }
    };
    if (!dA || !dB || !dC || !dCo)
    {
      test.skip(label, ResultStatus::Error, "Failed to allocate GEMM buffers", opts);
      freeAll();
      return;
    }
    try { q.memset(dA,  0x3f, cells * sizeof(double)).wait(); } catch (...) {}
    try { q.memset(dB,  0x3f, cells * sizeof(double)).wait(); } catch (...) {}
    try { q.memset(dC,  0,    cells * sizeof(double)).wait(); } catch (...) {}
    try { q.memset(dCo, 0,    sizeof(std::int32_t)).wait();   } catch (...) {}

    // Mean microseconds per call over n calls of one form, or -1 when it throws.
    auto runBatch = [&](const GemmForm &form, unsigned int n) -> double {
      try
      {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (unsigned int i = 0; i < n; i++)
          form.issue(q, dA, dB, dC, dCo, dim);
        q.wait_and_throw();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        return (double)ns / 1000.0 / (double)n;
      }
      catch (const std::exception &e)
      {
        // Contained to this dtype's private context (see above).
        CLPEAK_VLOG("oneMKL %s as %s failed: %s\n", label, form.name.c_str(), e.what());
        return -1.0;
      }
    };

    // The probe: one timed call, then up to three more while they fit in
    // 200 ms -- enough to rank forms on a GPU, where a call takes a few
    // milliseconds, without spending seconds a form on the CPU fallback.
    const unsigned int warm = warmupCount > 0 ? warmupCount : 2;
    const unsigned int probeMaxIters = 4;
    const double probeBudgetUs = 200000.0;
    size_t best = forms.size();
    double bestUs = 0.0;
    for (size_t f = 0; f < forms.size(); f++)
    {
      if (runBatch(forms[f], warm) <= 0.0)
        continue;
      double us = runBatch(forms[f], 1);
      if (us <= 0.0)
        continue;
      unsigned int probeIters = 1;
      const unsigned int more =
          (unsigned int)std::min<double>(probeMaxIters - 1, probeBudgetUs / us);
      if (more > 0)
      {
        const double moreUs = runBatch(forms[f], more);
        if (moreUs <= 0.0)
          continue;
        us = (us + moreUs * more) / (more + 1);
        probeIters += more;
      }
      if (forms.size() > 1)
        CLPEAK_VLOG("oneMKL %s as %s: %.1f %s over a %u-call probe\n", label,
                    forms[f].name.c_str(), flops * 1.0e6 / us, unit, probeIters);
      if (best == forms.size() || us < bestUs)
      {
        best = f;
        bestUs = us;
      }
    }
    if (best == forms.size())
      test.skip(label, ResultStatus::Error, "timing probe failed", opts);
    else
    {
      unsigned int iters = pickIters(bestUs, 5000000u, forceIters ? specifiedIters : 0);
      double meanUs = runBatch(forms[best], iters);
      if (meanUs <= 0.0)
        test.skip(label, ResultStatus::Error, "oneMKL GEMM failed", opts);
      else
      {
        if (forms.size() > 1)
          opts.description += "  Fastest form: " + forms[best].name + ".";
        test.emit(label, (float)(flops * 1.0e6 / meanUs), opts);
      }
    }
    freeAll();
    // q and its private context are destroyed here.
  };

  // All four layouts of one overload, for a dtype on the matrix engine: NN,
  // NT, TN -- A stored transposed so both inputs keep K contiguous, the
  // layout cuBLASLt's tensor-core kernels are tuned around (cuda_blas.cpp) --
  // and TT.  How oneMKL picks its kernel is not public, and the layout alone
  // can move it by several times: an Arc A380 (driver 8993) read int8 at 11.6
  // TOPS NN and 37.6 TN through the same call, where oneDNN's catalog for
  // DG2 (src/gpu/intel/gemm/jit/selector/db/kernel.db there) has the two 13%
  // apart.
  using LayoutGemm = std::function<void(sycl::queue &, transpose, transpose, void *, void *,
                                        void *, void *, std::int64_t)>;
  auto layouts = [](std::vector<GemmForm> &forms, const std::string &what, LayoutGemm gemm) {
    for (transpose ta : {transpose::nontrans, transpose::trans})
      for (transpose tb : {transpose::nontrans, transpose::trans})
        forms.push_back({what + ", " + (ta == transpose::trans ? "T" : "N") +
                             (tb == transpose::trans ? "T" : "N"),
                         [=](sycl::queue &q, void *dA, void *dB, void *dC, void *dCo,
                             std::int64_t n) { gemm(q, ta, tb, dA, dB, dC, dCo, n); }});
  };

  measure("fp32", D,
          {{"NN", [](sycl::queue &q, void *dA, void *dB, void *dC, void *, std::int64_t n) {
             mkl::blas::column_major::gemm(
               q, transpose::nontrans, transpose::nontrans, n, n, n, 1.0f,
               (const float *)dA, n, (const float *)dB, n, 0.0f, (float *)dC, n);
           }}},
          mklOpts(fp32Note));

  if (dev.info.fp64Supported)
    measure("fp64", fp64Dim,
            {{"NN", [](sycl::queue &q, void *dA, void *dB, void *dC, void *, std::int64_t n) {
               mkl::blas::column_major::gemm(
                 q, transpose::nontrans, transpose::nontrans, n, n, n, 1.0,
                 (const double *)dA, n, (const double *)dB, n, 0.0, (double *)dC, n);
             }}},
            mklOpts(fp64Note));
  else
    test.skip("fp64", ResultStatus::Unsupported, "fp64 not supported by this oneAPI device", mklOpts(fp64Note));

  // fp16 writes its result either as fp16 or as fp32.  That catalog lists no
  // matrix-engine kernel for DG2 that takes fp16 in to fp16 out -- its four
  // for that run on the vector units -- and an Arc A380 (driver 8993) read
  // this row at 8.77 TFLOPS with fp16 out: 90% of its vector fp16 rate,
  // where its matrix engine reads 15.5 through joint_matrix.
  if (dev.info.fp16Supported)
  {
    std::vector<GemmForm> forms;
    layouts(forms, "fp16 out", [](sycl::queue &q, transpose ta, transpose tb,
                                  void *dA, void *dB, void *dC, void *, std::int64_t n) {
      mkl::blas::column_major::gemm(
        q, ta, tb, n, n, n, sycl::half(1.0f),
        (const sycl::half *)dA, n, (const sycl::half *)dB, n,
        sycl::half(0.0f), (sycl::half *)dC, n);
    });
    layouts(forms, "fp32 out", [](sycl::queue &q, transpose ta, transpose tb,
                                  void *dA, void *dB, void *dC, void *, std::int64_t n) {
      mkl::blas::column_major::gemm(
        q, ta, tb, n, n, n, 1.0f,
        (const sycl::half *)dA, n, (const sycl::half *)dB, n, 0.0f, (float *)dC, n);
    });
    measure("fp16", D, forms, mklOpts(fp16Note));
  }
  else
    test.skip("fp16", ResultStatus::Unsupported, "fp16 not supported by this oneAPI device", mklOpts(fp16Note));

  // BF16: bf16 inputs, fp32 output + accumulate (HPA) -- the dtype combo the
  // Intel XMX bf16 GEMM peak is quoted against, matching joint_matrix.cpp.
#if defined(CLPEAK_ONEMKL_HAS_BF16)
  if (dev.info.bf16Supported)
  {
    using bfloat16 = sycl::ext::oneapi::bfloat16;
    std::vector<GemmForm> forms;
    layouts(forms, "fp32 out", [](sycl::queue &q, transpose ta, transpose tb,
                                  void *dA, void *dB, void *dC, void *, std::int64_t n) {
      mkl::blas::column_major::gemm(
        q, ta, tb, n, n, n, 1.0f,
        (const bfloat16 *)dA, n, (const bfloat16 *)dB, n, 0.0f, (float *)dC, n);
    });
    measure("bf16", D, forms, mklOpts(bf16Note));
  }
  else
    test.skip("bf16", ResultStatus::Unsupported, "bf16 not supported by this oneAPI device", mklOpts(bf16Note));
#else
  test.skip("bf16", ResultStatus::Unsupported,
            "SYCL bfloat16 header not available in this oneAPI toolchain", mklOpts(bf16Note));
#endif

  // INT8 through both of oneMKL's int8 entry points: gemm_bias (s8 x u8, its
  // A and B offsets zero) and gemm's int8 overload (s8 x s8, which takes no
  // offsets), since a nonzero offset needs row and column sums and the
  // catalog's fastest DG2 int8 kernels compute none.  On an Arc A380 (driver
  // 8993) the two read alike and the layout decided it: 11.6-11.8 TOPS NN,
  // 37.4-37.6 TN.
  if (!dev.info.xmxSupported)
    test.skip("int8", ResultStatus::Unsupported,
              "int8 GEMM requires Intel XMX (Arc/PVC/Battlemage)", intOpts(int8Note));
  else
  {
    std::vector<GemmForm> forms;
    layouts(forms, "gemm_bias s8 x u8", [](sycl::queue &q, transpose ta, transpose tb,
                                           void *dA, void *dB, void *dC, void *dCo,
                                           std::int64_t n) {
      mkl::blas::column_major::gemm_bias(
        q, ta, tb, mkl::offset::fix, n, n, n, 1.0f,
        (const std::int8_t *)dA, n, (std::int8_t)0,
        (const std::uint8_t *)dB, n, (std::uint8_t)0,
        0.0f, (std::int32_t *)dC, n, (const std::int32_t *)dCo);
    });
    layouts(forms, "gemm s8 x s8", [](sycl::queue &q, transpose ta, transpose tb,
                                      void *dA, void *dB, void *dC, void *, std::int64_t n) {
      mkl::blas::column_major::gemm(
        q, ta, tb, n, n, n, 1.0f,
        (const std::int8_t *)dA, n, (const std::int8_t *)dB, n, 0.0f, (std::int32_t *)dC, n);
    });
    measure("int8", D, forms, intOpts(int8Note));
  }

  return 0;
#endif
}

#endif // ENABLE_ONEAPI