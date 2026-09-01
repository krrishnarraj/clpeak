#if defined(ENABLE_CPU) && defined(__APPLE__)

// Apple Accelerate-framework GEMM tests: the sanctioned route to the matrix
// coprocessor Apple does not expose as an ISA (AMX on M1-M3, SME units on
// M4+; AVX on Intel Macs).  The per-ISA microbench rows in cpu_matrix.cpp
// probe only *architectural* engines (BFMMLA/SMMLA/SME), so on M1-M3 they
// report Unsupported while ~2/3 of the chip's CPU matrix throughput lives
// behind Accelerate -- these rows close that gap.
//
//   * Accelerate GEMM: cblas_sgemm / cblas_dgemm over a size sweep.  The
//     coprocessor has a real size cliff (fp64 peaks near N=512, fp32 near
//     N=2048, both sag at larger N), so the headline row is the sweep peak
//     and an "8k" row reports the large-N sustained rate.  Per-size results
//     go to stderr under --verbose.
//   * BNNS matmul: fp16 / bf16 via BNNSMatMul -- Accelerate's only public
//     reduced-precision GEMM.  BNNSMatMul is deprecated in favour of the
//     BNNSGraph API, but BNNSGraph consumes compiled model artifacts, which
//     a self-contained microbench can't ship; the direct call still routes
//     to the same kernels.  int8 has NO public matmul in BNNS (quantized
//     inference layers only) and is emitted as an Unsupported row.
//
// Accelerate threads internally, so these are single rows (no ST/MT split):
// the number is the library/coprocessor peak, not a per-core figure.
// bf16 note: on cores without hardware bf16 (M1) the library emulates and
// the row lands *below* the NEON fp32 rows -- honest, and it shows exactly
// what the API delivers on that part.

#include <cpu/cpu_peak.h>
#include <common/common.h>

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

double nowUs()
{
  using clock = std::chrono::high_resolution_clock;
  return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
             clock::now().time_since_epoch()).count() / 1000.0;
}

// Best FLOPS over a few repetitions of one gemm() call (2*N^3 flops).
// GEMM calls are chunky and low-variance: 1 warmup, then rep until ~250 ms
// of timed work or 6 reps, whichever comes first (always >= 2).
template <class Fn>
double bestFlops(uint64_t n, Fn gemm)
{
  const double flops = 2.0 * (double)n * (double)n * (double)n;
  gemm();  // warmup (first call pays internal setup / thread spin-up)

  double best = 0.0, spentUs = 0.0;
  for (int r = 0; r < 6; r++)
  {
    double t0 = nowUs();
    gemm();
    double us = nowUs() - t0;
    spentUs += us;
    if (us > 0.0)
      best = std::max(best, flops / (us * 1e-6));
    if (r >= 1 && spentUs >= 250e3)
      break;
  }
  return best;
}

// Skip sizes whose three square buffers would not comfortably fit in RAM.
bool fitsInMem(uint64_t n, size_t elemBytes, uint64_t totalMemBytes)
{
  uint64_t need = 3ull * n * n * elemBytes;
  uint64_t cap  = totalMemBytes ? totalMemBytes / 4 : (2ull << 30);
  return need <= std::min(cap, 2ull << 30);
}

struct SweepResult {
  double peakFlops    = 0.0;   // best over the sweep
  uint64_t peakN       = 0;
  double largestFlops = 0.0;   // rate at the largest size that ran
  uint64_t largestN    = 0;
};

template <class RunSize>
SweepResult sweep(const std::vector<uint64_t> &sizes, size_t elemBytes,
                  uint64_t totalMemBytes, const char *label, RunSize runSize)
{
  SweepResult res;
  for (uint64_t n : sizes)
  {
    if (!fitsInMem(n, elemBytes, totalMemBytes))
      break;
    double g = runSize(n);
    if (g <= 0.0)
      continue;
    CLPEAK_VLOG("  [accelerate] %s N=%llu: %.1f GFLOPS\n", label,
                (unsigned long long)n, g);
    if (g > res.peakFlops) { res.peakFlops = g; res.peakN = n; }
    res.largestFlops = g;
    res.largestN      = n;
  }
  return res;
}

} // namespace

int CpuPeak::runAppleBlas(benchmark_config_t &cfg)
{
  (void)cfg;
#if defined(__aarch64__)
  const char *engine = "AMX/SME";
#else
  const char *engine = "";
#endif

  // ---- Accelerate BLAS GEMM: fp32 + fp64 ----
  {
    auto test = currentDeviceScope->beginTest(
        {"accelerate_gemm", "Accelerate GEMM", "flops",
         Category::Unknown,
         "Matrix-multiply speed through Apple's Accelerate library, the only way "
         "to reach the matrix coprocessor Apple ships but does not expose as "
         "instructions.  The library uses every core itself, so there is no "
         "single-thread row.",
         // Precision and problem size both vary, so no one noun heads the
         // column -- but none of the three readings stands for the others.
         TestShape::Heterogeneous, "", Direction::FromUnit, engine});

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    SweepResult sp = sweep({256, 512, 1024, 2048, 4096, 8192}, sizeof(float),
                           info.totalMemBytes, "sgemm", [](uint64_t n) {
      std::vector<float> a(n * n, 1.0f), b(n * n, 0.5f), c(n * n, 0.0f);
      return bestFlops(n, [&] {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)n, (int)n, (int)n,
                    1.0f, a.data(), (int)n, b.data(), (int)n,
                    0.0f, c.data(), (int)n);
      });
    });

    // fp64 sweep stops at 4096: the fp64 peak sits at small N anyway and the
    // large sizes only add multi-second runtime at a known-sagging rate.
    SweepResult dp = sweep({256, 512, 1024, 2048, 4096}, sizeof(double),
                           info.totalMemBytes, "dgemm", [](uint64_t n) {
      std::vector<double> a(n * n, 1.0), b(n * n, 0.5), c(n * n, 0.0);
      return bestFlops(n, [&] {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    (int)n, (int)n, (int)n,
                    1.0, a.data(), (int)n, b.data(), (int)n,
                    0.0, c.data(), (int)n);
      });
    });
#pragma clang diagnostic pop

    // Which N peaked is the AMX/SME size-cliff signature -- worth recording
    // when validating a new Apple part.
    CLPEAK_VLOG("  [accelerate] peak sizes: sgemm N=%llu, dgemm N=%llu\n",
                (unsigned long long)sp.peakN, (unsigned long long)dp.peakN);

    const char *spNote = "Best 32-bit result over a sweep of matrix sizes; the "
                         "coprocessor is fastest at one particular size.";
    const char *dpNote = "Best 64-bit result over the same size sweep.";
    if (sp.peakFlops > 0.0) test.emit("sgemm", (float)sp.peakFlops, spNote);
    else                     test.skip("sgemm", ResultStatus::Error, "sgemm sweep failed", spNote);
    // Large-N sustained row only when the 8k size actually ran (RAM-gated).
    if (sp.largestN >= 8192)
      test.emit("sgemm 8k", (float)sp.largestFlops,
                "The 32-bit rate on 8192x8192 matrices -- what large, realistic "
                "problems actually sustain.");
    if (dp.peakFlops > 0.0) test.emit("dgemm", (float)dp.peakFlops, dpNote);
    else                     test.skip("dgemm", ResultStatus::Error, "dgemm sweep failed", dpNote);
  }

  // ---- BNNS matmul: fp16 / bf16 (+ int8 unsupported row) ----
  {
    auto test = currentDeviceScope->beginTest(
        {"bnns_matmul", "BNNS matmul", "flops",
         Category::Unknown,
         "Matrix-multiply speed through Apple's BNNS library, the only public "
         "route to reduced-precision (16-bit) matrix maths on Apple Silicon.",
         TestShape::Heterogeneous, "data type", Direction::FromUnit, engine});

    const char *fp16Note = "16-bit float inputs.";
    const char *bf16Note = "bfloat16 inputs, the AI-oriented 16-bit format.  On "
                           "chips with no bf16 hardware the library emulates it, "
                           "and the rate falls accordingly.";
    const char *int8Note = "8-bit whole-number inputs.";

    if (__builtin_available(macOS 13.0, iOS 16.0, *))
    {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
      auto runBnns = [&](BNNSDataType dt, size_t elemBytes, const char *label)
                         -> double {
        SweepResult r = sweep({512, 1024, 2048, 4096}, elemBytes,
                              info.totalMemBytes, label, [&](uint64_t n) {
          std::vector<char> a(n * n * elemBytes, 0), b(n * n * elemBytes, 0),
                            c(n * n * elemBytes, 0);
          auto desc = [&](void *data) {
            BNNSNDArrayDescriptor d;
            std::memset(&d, 0, sizeof(d));
            d.layout    = BNNSDataLayoutRowMajorMatrix;
            d.size[0]   = n;
            d.size[1]   = n;
            d.data      = data;
            d.data_type = dt;
            return d;
          };
          BNNSNDArrayDescriptor da = desc(a.data()), db = desc(b.data()),
                                dc = desc(c.data());
          ssize_t ws = BNNSMatMulWorkspaceSize(false, false, 1.0f,
                                               &da, &db, &dc, nullptr);
          std::vector<char> wk(ws > 0 ? (size_t)ws : 0);
          void *wkp = wk.empty() ? nullptr : wk.data();
          if (BNNSMatMul(false, false, 1.0f, &da, &db, &dc, wkp, nullptr) != 0)
            return 0.0;
          return bestFlops(n, [&] {
            BNNSMatMul(false, false, 1.0f, &da, &db, &dc, wkp, nullptr);
          });
        });
        return r.peakFlops;
      };

      double fp16 = runBnns(BNNSDataTypeFloat16, 2, "bnns fp16");
      if (fp16 > 0.0) test.emit("fp16", (float)fp16, fp16Note);
      else            test.skip("fp16", ResultStatus::Unsupported,
                                "BNNSMatMul rejected fp16 operands", fp16Note);

      double bf16 = runBnns(BNNSDataTypeBFloat16, 2, "bnns bf16");
      if (bf16 > 0.0) test.emit("bf16", (float)bf16, bf16Note);
      else            test.skip("bf16", ResultStatus::Unsupported,
                                "BNNSMatMul rejected bf16 operands", bf16Note);
#pragma clang diagnostic pop
    }
    else
    {
      test.skip("fp16", ResultStatus::Unsupported,
                "BNNSMatMul needs macOS 13 / iOS 16", fp16Note);
      test.skip("bf16", ResultStatus::Unsupported,
                "BNNSMatMul needs macOS 13 / iOS 16", bf16Note);
    }

    test.skip("int8", ResultStatus::Unsupported,
              "BNNS exposes no int8 GEMM (quantized inference layers only)",
              int8Note);
  }

  return 0;
}

#endif // ENABLE_CPU && __APPLE__
