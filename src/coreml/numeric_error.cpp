#ifdef ENABLE_COREML

// coreml-numeric-error: what each weight format's speed row costs in
// accuracy, on this compute unit.
//
// A rate without its accuracy is half a number: int8 is fast because it
// discarded precision, and the speed row cannot say how much.  Each format
// is therefore also measured as relative RMS error against a reference
// built from *the same values the device saw* -- the stored codes widened,
// not the fp32 originals they were rounded from -- multiplied in double
// precision on the host.  The operand rounding is then identical on both
// sides and cancels, and what each row reports is the arithmetic plus the
// width the answer was kept in: the part that belongs to the compute unit.
//
// The fp32 and fp16 rows double as a precision detector.  Core ML's GPU
// accumulates fp16 in fp32 unless asked otherwise; whether the Neural
// Engine does is exactly what its fp16 row against the CPU's says.  And the
// fp32 row on the Neural Engine device reads whatever unit Core ML actually
// sent it to, which the plan names.

#include <coreml/coreml_peak.h>
#include "coreml_bench.h"
#include "coreml_model.h"
#include "coreml_session.h"

// The reference GEMM runs on Accelerate; the macro selects its current CBLAS
// interface rather than the deprecated one.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{

// Fixed size on every device: the error depends on the accumulation depth
// K, so it has to be the same K everywhere or the rows are not comparable.
constexpr int64_t kDim = 1024;

// What each weight layout gets to show its speed in: a few hundred
// multiplies, enough to tell a slow kernel from a fast one.
constexpr unsigned int kRaceBudgetUs = 200000;

struct Variant
{
  CoremlWeight w;
  const char *note;
};

const Variant kVariants[] = {
    {CoremlWeight::Fp16,
     "16-bit inputs and a 16-bit answer.  Around 200 ppm is an answer "
     "accumulated in fp32 and rounded once at the end; ten times that is a "
     "unit accumulating in fp16 all the way."},
    {CoremlWeight::Fp32,
     "Full precision; near zero, and the control the other rows are read "
     "against.  A nonzero figure here is a compute unit substituting a "
     "narrower type for the fp32 it was handed."},
    {CoremlWeight::Bf16,
     "bfloat16 has no arithmetic in Core ML; the row records the refusal so "
     "the accuracy table stays in step with the speed one."},
    {CoremlWeight::Int8Channel,
     "8-bit weights with a scale per output column, multiplied in 16-bit.  "
     "The quantization of the weights is not counted -- the reference uses "
     "the same codes at their exact scaled values -- so a unit that folds the "
     "scale out of the accumulation reads near its fp16 row, and one that "
     "rounds each weight to 16 bits first reads a little above it."},
    {CoremlWeight::Int4Block,
     "4-bit weights with a scale per block of 32, multiplied in 16-bit.  As "
     "for int8_weight: the codes are shared with the reference, so whatever "
     "separates this from the fp16 row is the decompression."},
    {CoremlWeight::Int4Lut,
     "4-bit table lookup, multiplied in 16-bit; the lookup is exact, so this "
     "row should match fp16."},
    {CoremlWeight::Fp8Block,
     "8-bit float weights with block scales, if any compute unit takes them."},
    {CoremlWeight::Int8Qdq,
     "8-bit weights and 8-bit activations, with the answer itself quantized "
     "to 8 bits.  Around 9000 ppm is what keeping the result in int8 costs on "
     "this data; the quantization of the operands is not counted."},
};

double relativeRmsPpm(const std::vector<float> &got, const std::vector<double> &ref)
{
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); i++)
  {
    const double d = (double)got[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  if (den <= 0.0)
    return -1.0;
  return std::sqrt(num / den) * 1.0e6;
}

} // namespace

int CoreMLPeak::runNumericError(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;
  const int spec = coremlSpecVersion();

  auto test = currentDeviceScope->beginTest(
      {"coreml_numeric_error", "Core ML matmul numeric error", "ppm", Category::Compute,
       "How far each weight format's answer drifts from a full-precision one, "
       "in parts per million, on a fixed 1024-cubed matrix multiply -- what "
       "the speed rows cost.  The reference multiplies the same stored values "
       "the compute unit was handed, in double precision, so this is the "
       "arithmetic and the width the answer was kept in, not the rounding of "
       "the inputs, which is the format's and identical everywhere.",
       TestShape::Heterogeneous, "weight format"});

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;
    const char *label = coremlWeightLabel(v.w);
    const int act = coremlActDtype(v.w);
    const int ioDtype = (act == CML_BF16) ? CML_FP16 : act;

    if (coremlSpecForWeight(v.w) > spec)
    {
      test.skip(label, ResultStatus::Unsupported,
                "needs " + coremlOsForSpec(coremlSpecForWeight(v.w)) +
                    " (model specification " + std::to_string(coremlSpecForWeight(v.w)) +
                    "); this OS accepts " + std::to_string(spec),
                v.note);
      continue;
    }

    // The activations: the same generator as the speed rows, rounded to the
    // width the model takes them in, which is the width the device sees.
    const std::string xRaw = coremlFillFloats(ioDtype, kDim * kDim, 0x243f6a88u);

    // The weights stored both ways, each timed, and the faster one's answer
    // is the row's: the speed rows race the two layouts (CoremlLayoutRace),
    // and a layout can change the kernel and with it the arithmetic -- the
    // M1 Pro's CPU multiplies blockwise int4 stored [in, out] in a slow
    // kernel accumulating in fp32 (272 ppm), and stored [out, in], seven
    // times faster, in the fp16 one its other rows read (3218).
    std::vector<float> weights;   // exactly what the device multiplies, either way
    std::unique_ptr<CoremlSession> sess[2];
    double rate[2] = {0.0, 0.0}, createUs[2] = {0.0, 0.0};
    std::string why[2];
    ResultStatus whyStatus[2] = {ResultStatus::Unsupported, ResultStatus::Unsupported};
    for (int t = 0; t < 2; t++)
    {
      std::string err;
      auto s = CoremlSession::create(
          dev, coremlPlainMatMulModel(spec, kDim, kDim, kDim, v.w, &weights, t), err);
      if (!s)
      {
        why[t] = err;
        continue;
      }
      if (!s->onDevice())
      {
        why[t] = coremlOffDeviceReason(dev, *s);
        continue;
      }
      void *xp = s->bindInput("x", ioDtype, {kDim, kDim}, xRaw.size(), err);
      if (!xp)
      {
        why[t] = err;
        whyStatus[t] = ResultStatus::Error;
        continue;
      }
      std::memcpy(xp, xRaw.data(), xRaw.size());
      const auto m = coremlMeasure(*s, warmupCount, kRaceBudgetUs, false, 0);
      if (m.meanUs <= 0.0)
      {
        why[t] = m.error;
        whyStatus[t] = m.status;
        continue;
      }
      rate[t] = 1.0 / m.meanUs;
      createUs[t] = coremlCreateUs(*s);
      CLPEAK_VLOG("coreml-numeric-error[%s/%s]: %s %.1f us per multiply, create %.2f s\n",
                  dev.displayName.c_str(), label, coremlLayoutName(t), m.meanUs,
                  createUs[t] / 1.0e6);
      sess[t] = std::move(s);
    }
    if (!sess[0] && !sess[1])
    {
      test.skip(label, whyStatus[0], why[0], v.note);
      continue;
    }
    const bool both = sess[0] && sess[1];
    const bool transposed = both ? CoremlLayoutRace::pick(rate, createUs) : (bool)sess[1];
    std::unique_ptr<CoremlSession> s = std::move(sess[transposed]);
    sess[!transposed].reset();
    std::string err;
    const std::string note =
        std::string(v.note) + "  The weights stored " + coremlLayoutName(transposed) +
        (both ? ": of the two orders, the one this compute unit multiplies faster -- where the "
                "two run alike, [in, out] unless [out, in] compiles several times faster."
              : "; stored the other way, this compute unit did not take them.");

    std::vector<double> a((size_t)kDim * kDim);
    for (int64_t i = 0; i < kDim * kDim; i++)
    {
      float f;
      if (ioDtype == CML_FP32)
        std::memcpy(&f, xRaw.data() + i * 4, 4);
      else
      {
        uint16_t h;
        std::memcpy(&h, xRaw.data() + i * 2, 2);
        f = coremlHalfToFloat(h);
      }
      if (v.w == CoremlWeight::Int8Qdq)
      {
        // The device quantizes the activations before multiplying; the
        // reference multiplies what comes back from that, so the operand
        // rounding cancels here as it does for the weights.
        const float scale = coremlHalfToFloat(coremlFloatToHalf(coremlQdqActScale(1.0f)));
        int q = (int)std::lround(f / scale);
        q = std::max(-128, std::min(127, q));
        f = scale * (float)q;
      }
      a[(size_t)i] = f;
    }

    if (!s->run(err))
    {
      test.skip(label, ResultStatus::Error, err, note);
      continue;
    }
    std::vector<uint8_t> raw;
    if (!s->outputBytes("y", raw, err))
    {
      test.skip(label, ResultStatus::Error, err, note);
      continue;
    }
    s.reset();

    std::vector<float> got((size_t)kDim * kDim);
    if (act == CML_FP32)
    {
      if (raw.size() != got.size() * 4)
      {
        test.skip(label, ResultStatus::Error, "unexpected output size", note);
        continue;
      }
      std::memcpy(got.data(), raw.data(), raw.size());
    }
    else
    {
      if (raw.size() != got.size() * 2)
      {
        test.skip(label, ResultStatus::Error, "unexpected output size", note);
        continue;
      }
      for (size_t i = 0; i < got.size(); i++)
      {
        uint16_t h;
        std::memcpy(&h, raw.data() + i * 2, 2);
        got[i] = coremlHalfToFloat(h);
      }
    }

    // The reference: A[M,K] * W[K,N] in double, on the host.
    std::vector<double> b(weights.begin(), weights.end());
    std::vector<double> ref((size_t)kDim * kDim, 0.0);
    // The current CBLAS entry point is iOS 16.4 / macOS 13.3; the backend
    // itself needs 17.4 / 14.4, so the fallback is never taken and exists to
    // keep the availability check honest.
    if (__builtin_available(macOS 13.3, iOS 16.4, *))
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)kDim, (int)kDim, (int)kDim, 1.0,
                  a.data(), (int)kDim, b.data(), (int)kDim, 0.0, ref.data(), (int)kDim);
    else
      for (int64_t i = 0; i < kDim; i++)
        for (int64_t k = 0; k < kDim; k++)
        {
          const double aik = a[(size_t)i * kDim + k];
          for (int64_t j = 0; j < kDim; j++)
            ref[(size_t)i * kDim + j] += aik * b[(size_t)k * kDim + j];
        }

    const double ppm = relativeRmsPpm(got, ref);
    if (ppm < 0.0)
    {
      test.skip(label, ResultStatus::Error, "reference result was all zero", note);
      continue;
    }
    // A NaN or an infinity anywhere in the compute unit's answer makes the
    // figure one too.  That is no error figure, and it cannot be recorded
    // either: the document spells it `nan`, which no JSON reader -- clpeak's
    // own included -- will load.  The reference multiplies values under 1 in
    // magnitude, in double, so the non-finite side is the compute unit's.
    if (!std::isfinite(ppm))
    {
      test.skip(label, ResultStatus::Error,
                "this compute unit's answer holds NaN or infinity where the host's "
                "double-precision reference is finite everywhere: it computed something "
                "other than this matmul, so there is no error figure to report",
                note);
      continue;
    }
    test.emit(label, (float)ppm, note.c_str());
  }

  test.end();
  return 0;
}

#endif // ENABLE_COREML
