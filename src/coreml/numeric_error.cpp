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
//
// Every measurement here is also the answer check the rate rows ask before
// they publish (CoreMLPeak::answerCheck, include/common/answer_check.h), in
// each weight layout, and for the convolution rows the fp16 and fp32
// products written as a 1x1 convolution.  A wrong answer keeps its layout
// out of the rate rows' races, withholds them where no layout answers
// right, and files this row as an Error carrying its figure.

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

// How this backend's answer checks name things (include/common/answer_check.h).
const clpeak::AnswerWords kWords = {"this compute unit's", "coreml_numeric_error", "the host's reference",
                                    "the host's double-precision reference"};

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

// The reference: A[M, K] * B[K, N] in double, on the host.
void referenceGemm(const std::vector<double> &a, const std::vector<double> &b, int64_t M, int64_t K,
                   int64_t N, std::vector<double> &ref)
{
  ref.assign((size_t)(M * N), 0.0);
  // The current CBLAS entry point is iOS 16.4 / macOS 13.3; the backend
  // itself needs 17.4 / 14.4, so the fallback is never taken and exists to
  // keep the availability check honest.
  if (__builtin_available(macOS 13.3, iOS 16.4, *))
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)M, (int)N, (int)K, 1.0, a.data(), (int)K,
                b.data(), (int)N, 0.0, ref.data(), (int)N);
  else
    for (int64_t i = 0; i < M; i++)
      for (int64_t k = 0; k < K; k++)
      {
        const double aik = a[(size_t)(i * K + k)];
        for (int64_t j = 0; j < N; j++)
          ref[(size_t)(i * N + j)] += aik * b[(size_t)(k * N + j)];
      }
}

// A session's answer, read back as floats: `dtype` is the output's (fp16 or
// fp32).  Empty with `error` set when it cannot be read.
std::vector<float> readAnswer(CoremlSession &s, int dtype, std::string &error)
{
  std::vector<uint8_t> raw;
  std::vector<float> got;
  if (!s.outputBytes("y", raw, error))
    return got;
  const size_t es = dtype == CML_FP32 ? 4 : 2;
  if (raw.size() != (size_t)kDim * kDim * es)
  {
    error = "unexpected output size";
    return got;
  }
  got.resize((size_t)kDim * kDim);
  for (size_t i = 0; i < got.size(); i++)
  {
    if (es == 4)
      std::memcpy(&got[i], raw.data() + i * 4, 4);
    else
    {
      uint16_t h;
      std::memcpy(&h, raw.data() + i * 2, 2);
      got[i] = coremlHalfToFloat(h);
    }
  }
  return got;
}

// The activations the device sees, widened: the input's own width, and for
// the quantized-activation format the codes it quantizes them to.
std::vector<double> widenInput(const std::string &xRaw, int ioDtype, CoremlWeight w)
{
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
    if (w == CoremlWeight::Int8Qdq)
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
  return a;
}

// The format's matmul in both weight layouts, each timed (the accuracy row
// reads the faster) and each answer measured.
void measureMatMul(const coreml_device_info_t &dev, CoremlWeight w, unsigned warmup,
                   CoreMLPeak::AnswerCheck &c)
{
  const int spec = coremlSpecVersion();
  const char *label = coremlWeightLabel(w);
  const int act = coremlActDtype(w);
  const int ioDtype = (act == CML_BF16) ? CML_FP16 : act;

  if (coremlSpecForWeight(w) > spec)
  {
    for (clpeak::AnswerCheck &l : c.layout)
    {
      l.status = ResultStatus::Unsupported;
      l.error = "needs " + coremlOsForSpec(coremlSpecForWeight(w)) + " (model specification " +
                std::to_string(coremlSpecForWeight(w)) + "); this OS accepts " + std::to_string(spec);
    }
    return;
  }

  // The activations: the same generator as the speed rows, rounded to the
  // width the model takes them in, which is the width the device sees.
  const std::string xRaw = coremlFillFloats(ioDtype, kDim * kDim, 0x243f6a88u);

  // The weights stored both ways, each timed: the speed rows race the two
  // layouts (CoremlLayoutRace), and a layout can change the kernel and with
  // it the arithmetic -- the M1 Pro's CPU multiplies blockwise int4 stored
  // [in, out] in a slow kernel accumulating in fp32 (272 ppm), and stored
  // [out, in], seven times faster, in the fp16 one its other rows read
  // (3218).
  std::vector<float> weights;   // exactly what the device multiplies, either way
  std::unique_ptr<CoremlSession> sess[2];
  double rate[2] = {0.0, 0.0}, createUs[2] = {0.0, 0.0};
  for (int t = 0; t < 2; t++)
  {
    clpeak::AnswerCheck &l = c.layout[t];
    l.status = ResultStatus::Unsupported;
    std::string err;
    auto s = CoremlSession::create(dev, coremlPlainMatMulModel(spec, kDim, kDim, kDim, w, &weights, t), err);
    if (!s)
    {
      l.error = err;
      continue;
    }
    if (!s->onDevice())
    {
      l.error = coremlOffDeviceReason(dev, *s);
      continue;
    }
    void *xp = s->bindInput("x", ioDtype, {kDim, kDim}, xRaw.size(), err);
    if (!xp)
    {
      l.error = err;
      l.status = ResultStatus::Error;
      continue;
    }
    std::memcpy(xp, xRaw.data(), xRaw.size());
    const auto m = coremlMeasure(*s, warmup, kRaceBudgetUs, false, 0);
    if (m.meanUs <= 0.0)
    {
      l.error = m.error;
      l.status = m.status;
      continue;
    }
    l.status = ResultStatus::Ok;
    rate[t] = 1.0 / m.meanUs;
    createUs[t] = coremlCreateUs(*s);
    CLPEAK_VLOG("coreml-numeric-error[%s/%s]: %s %.1f us per multiply, create %.2f s\n",
                dev.displayName.c_str(), label, coremlLayoutName(t), m.meanUs, createUs[t] / 1.0e6);
    sess[t] = std::move(s);
  }
  if (!sess[0] && !sess[1])
    return;
  const bool both = sess[0] && sess[1];
  const bool faster = both ? CoremlLayoutRace::pick(rate, createUs) : (bool)sess[1];

  // Each answer against one reference: A[M,K] * W[K,N] in double, on the
  // host, from the values both layouts multiply.
  const std::vector<double> a = widenInput(xRaw, ioDtype, w);
  const std::vector<double> b(weights.begin(), weights.end());
  std::vector<double> ref;
  referenceGemm(a, b, kDim, kDim, kDim, ref);
  for (int t = 0; t < 2; t++)
  {
    if (!sess[t])
      continue;
    clpeak::AnswerCheck &l = c.layout[t];
    std::string err;
    std::vector<float> got;
    if (sess[t]->run(err))
      got = readAnswer(*sess[t], act, err);
    sess[t].reset();
    if (got.empty())
    {
      l.status = ResultStatus::Error;
      l.error = err;
      continue;
    }
    clpeak::judgeAnswer(l, got.data(), ref.data(), got.size(), kWords);
    if (l.ppm >= 0.0)
      CLPEAK_VLOG("coreml-numeric-error[%s/%s]: %s %.2f ppm%s\n", dev.displayName.c_str(), label,
                  coremlLayoutName(t), l.ppm, l.wrong() ? ", a wrong answer" : "");
  }

  // The row reads the faster layout of those that answer right; with none
  // right, the faster of those measured, whose wrong answer it reports.
  auto measured = [&](int t) { return c.layout[t].ppm >= 0.0 || c.layout[t].nonFinite; };
  auto right = [&](int t) { return measured(t) && !c.layout[t].wrong(); };
  const int fast = faster ? 1 : 0, slow = 1 - fast;
  c.read = right(fast) ? fast : right(slow) ? slow : measured(fast) ? fast : measured(slow) ? slow : fast;
  const int other = 1 - c.read;
  c.note = std::string("  The weights stored ") + coremlLayoutName(c.read);
  const std::string leftOut = ", so the rate rows leave that order out.";
  if (!both)
    c.note += "; stored the other way, this compute unit did not take them.";
  else if (c.read == fast)
  {
    c.note += ": of the two orders, the one this compute unit multiplies faster -- where the "
              "two run alike, [in, out] unless [out, in] compiles several times faster.";
    if (right(c.read) && c.layout[other].wrong())
      c.note += std::string("  Stored ") + coremlLayoutName(other) + ", its answer is wrong" + leftOut;
  }
  else
    c.note += std::string("; stored ") + coremlLayoutName(other) + ", the order it multiplies faster, " +
              (c.layout[other].wrong() ? "its answer is wrong" + leftOut : std::string("its answer could not be read."));
}

// The format's product written as a 1x1 convolution, which the convolution
// rows ask: one layout, untimed.
void measureConv1x1(const coreml_device_info_t &dev, CoremlWeight w, unsigned warmup,
                    CoreMLPeak::AnswerCheck &c)
{
  const int spec = coremlSpecVersion();
  const int dtype = coremlActDtype(w);
  clpeak::AnswerCheck &l = c.layout[0];
  l.status = ResultStatus::Unsupported;
  std::vector<float> weights;
  std::string err;
  auto s = CoremlSession::create(dev, coremlPlainConv1x1Model(spec, kDim, kDim, kDim, dtype, &weights), err);
  if (!s)
  {
    l.error = err;
    return;
  }
  if (!s->onDevice())
  {
    l.error = coremlOffDeviceReason(dev, *s);
    return;
  }
  c.read = 0;
  const std::string xRaw = coremlFillFloats(dtype, kDim * kDim, 0x243f6a88u);
  void *xp = s->bindInput("x", dtype, {1, kDim, 32, 32}, xRaw.size(), err);
  std::vector<float> got;
  if (xp)
  {
    std::memcpy(xp, xRaw.data(), xRaw.size());
    if (s->timeRuns(1 + warmup, err) > 0.0)
      got = readAnswer(*s, dtype, err);
  }
  s.reset();
  if (got.empty())
  {
    l.status = ResultStatus::Error;
    l.error = err;
    return;
  }
  // y[n][p] = sum_k W[n][k] x[k][p]: the weights [N, K] times the
  // activations as [K, H * W].
  const std::vector<double> a = widenInput(xRaw, dtype, w);
  const std::vector<double> b(weights.begin(), weights.end());
  std::vector<double> ref;
  referenceGemm(b, a, kDim, kDim, kDim, ref);
  clpeak::judgeAnswer(l, got.data(), ref.data(), got.size(), kWords);
  if (l.ppm >= 0.0)
    CLPEAK_VLOG("coreml-numeric-error[%s/%s as a 1x1 convolution]: %.2f ppm%s\n", dev.displayName.c_str(),
                coremlWeightLabel(w), l.ppm, l.wrong() ? ", a wrong answer" : "");
}

} // namespace

const CoreMLPeak::AnswerCheck &CoreMLPeak::answerCheck(const coreml_device_info_t &dev, CoremlWeight w,
                                                       bool conv1x1)
{
  const auto key = std::make_tuple((int)dev.kind, dev.gpuIndex, (int)w, conv1x1);
  auto found = answerChecks_.find(key);
  if (found != answerChecks_.end())
    return found->second;
  AnswerCheck &c = answerChecks_[key];
  if (conv1x1)
    measureConv1x1(dev, w, warmupCount, c);
  else
    measureMatMul(dev, w, warmupCount, c);
  return c;
}

std::string CoreMLPeak::wrongAnswer(const coreml_device_info_t &dev, CoremlWeight w, bool transposed,
                                    bool conv1x1)
{
  const AnswerCheck &c = answerCheck(dev, w, conv1x1);
  const std::string what = std::string(coremlWeightLabel(w)) +
                           (conv1x1 ? " written as a 1x1 convolution"
                                    : std::string(" stored ") + coremlLayoutName(transposed));
  return clpeak::wrongAnswerReason(c.layout[conv1x1 ? 0 : (int)transposed], kWords, what);
}

int CoreMLPeak::runNumericError(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;

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
    const AnswerCheck &c = answerCheck(dev, v.w);
    if (c.read < 0)
    {
      test.skip(label, c.layout[0].status, c.layout[0].error, v.note);
      continue;
    }
    const std::string note = std::string(v.note) + c.note;
    const clpeak::AnswerCheck &r = c.layout[c.read];
    // A wrong answer in every layout measured is no precision figure: the
    // row is an Error carrying it, beside the rate rows it withheld.
    if (r.wrong())
    {
      const clpeak::AnswerCheck &o = c.layout[1 - c.read];
      const bool otherWrong = o.ppm >= 0.0 || o.nonFinite;
      test.skip(label, ResultStatus::Error,
                clpeak::wrongAnswerRowReason(
                    r, kWords,
                    otherWrong ? std::string("and stored ") + coremlLayoutName(1 - c.read) +
                                     " it is wrong too, so its rate rows are withheld"
                               : std::string("so its rate rows are withheld")),
                note);
      continue;
    }
    if (r.ppm < 0.0)
    {
      test.skip(label, r.status, r.error, note);
      continue;
    }
    test.emit(label, (float)r.ppm, note.c_str());
  }

  test.end();
  return 0;
}

#endif // ENABLE_COREML
