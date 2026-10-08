#ifdef ENABLE_LITERT

// litert-numeric-error: what each format's speed row costs in accuracy, on
// this accelerator.
//
// A rate without its accuracy is half a number: int8 is fast because it
// discarded precision, and the speed row cannot say how much.  Each format
// is therefore also measured as relative RMS error against a reference
// built from *the values the accelerator was handed* -- the activations as
// they went in, the stored weight codes at their exact scaled values --
// multiplied in double precision on the host.  The rounding of the stored
// operands is then identical on both sides and cancels; what a row reports
// is the arithmetic and the width the answer was kept in.
//
// One exception is deliberate.  On the CPU, XNNPACK runs the weight-only
// formats (int8_weight, int4_weight) by quantizing the float activations to
// int8 on the fly, inside the kernel, where no reference can see the codes.
// That rounding is therefore counted for those rows, and it is the right
// thing to count: it is what the format costs on that accelerator, and the
// GPU's row for the same format -- which unpacks the weights to half and
// multiplies in float -- reads an order of magnitude lower beside it.
//
// The fp32 and fp16 rows double as a precision detector: the GPU accelerator
// computes an fp32 graph in fp16 unless told otherwise, and its fp32 row is
// the check that asking worked.

#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace
{

// Fixed size on every device: the error depends on the accumulation depth
// K, so it has to be the same K everywhere or the rows are not comparable.
constexpr int64_t kDim = 1024;

struct Variant
{
  LitertFormat f;
  const char *note;
};

const Variant kVariants[] = {
    {LitertFormat::Fp32, "Full 32-bit precision: hundreds of ppm would mean a narrower type was used."},
    {LitertFormat::Fp16,
     "16-bit weights and arithmetic: about 200 ppm is fp32 accumulation, ten times that is fp16 "
     "accumulation."},
    {LitertFormat::Fp16Acc32, "The GPU's fp16 policy with fp32 accumulation."},
    {LitertFormat::Bf16, "bfloat16 weights and arithmetic."},
    {LitertFormat::Int8Qdq,
     "Full-integer int8 with the answer kept in 8 bits: about 9000 ppm on this data."},
    {LitertFormat::Int16x8,
     "Full-integer int16x8: 16-bit activations over 8-bit weights, the answer kept in 16 bits."},
    {LitertFormat::Int8Weight,
     "8-bit weights, one scale per output column, against float activations, counting the activations' "
     "rounding where the kernel quantizes them."},
    {LitertFormat::Int4Weight,
     "4-bit weights in blocks of 32 against float activations, counting the activations' rounding where "
     "the kernel quantizes them."},
    {LitertFormat::Fp8Weight,
     "8-bit float (E4M3) weights, one scale per output column, against float activations."},
};

// How this backend's answer checks name things (include/common/answer_check.h).
const clpeak::AnswerWords kWords = {"this accelerator's", "litert_numeric_error", "the host's reference",
                                    "the host's double-precision reference"};

// y[m][n] = sum_k x[m][k] * w[n][k], in double, across the host's threads.
void referenceGemm(const std::vector<double> &x, const std::vector<float> &w, int64_t M, int64_t K,
                   int64_t N, std::vector<double> &y)
{
  y.assign((size_t)(M * N), 0.0);
  unsigned threads = std::thread::hardware_concurrency();
  if (threads == 0)
    threads = 1;
  threads = std::min<unsigned>(threads, (unsigned)M);
  std::vector<std::thread> pool;
  for (unsigned t = 0; t < threads; t++)
  {
    pool.emplace_back([&, t]() {
      for (int64_t m = t; m < M; m += threads)
      {
        const double *xr = &x[(size_t)(m * K)];
        for (int64_t n = 0; n < N; n++)
        {
          const float *wr = &w[(size_t)(n * K)];
          double acc = 0.0;
          for (int64_t k = 0; k < K; k++)
            acc += xr[k] * (double)wr[k];
          y[(size_t)(m * N + n)] = acc;
        }
      }
    });
  }
  for (auto &th : pool)
    th.join();
}

// The marker every byte of an answer's output is set to before the run.  An
// element still holding it was never written: as int8 it is -128, past the
// 4-sigma code the scales put at 127, and as int16, fp16 and fp32 a value no
// product here can take.
constexpr uint8_t kUnwrittenByte = 0x80;
// A share of the output past this still holding the marker is an answer not
// written, not a coincidence.
constexpr double kUnwrittenShare = 0.5;

// How much of an output still holds the marker, element by element.
double unwrittenShare(const std::vector<uint8_t> &out, size_t elemBytes)
{
  if (out.empty() || elemBytes == 0)
    return -1.0;
  const size_t n = out.size() / elemBytes;
  size_t marked = 0;
  for (size_t e = 0; e < n; e++)
  {
    bool all = true;
    for (size_t b = 0; b < elemBytes && all; b++)
      all = out[e * elemBytes + b] == kUnwrittenByte;
    marked += all;
  }
  return n ? (double)marked / (double)n : -1.0;
}

// An output's elements as the values they stand for: `io` is the type the
// graph returns, `act` the plan's activation type -- an integer answer comes
// back as codes at the output scale, or under floatIo already dequantized.
bool decodeOutput(const std::vector<uint8_t> &out, clpeak_tflite::TfType io, clpeak_tflite::TfType act,
                  std::vector<float> &got)
{
  using clpeak_tflite::TfType;
  switch (io)
  {
  case TfType::F32:
    if (out.size() != got.size() * 4)
      return false;
    std::memcpy(got.data(), out.data(), out.size());
    return true;
  case TfType::F16:
  case TfType::BF16:
    if (out.size() != got.size() * 2)
      return false;
    for (size_t i = 0; i < got.size(); i++)
    {
      uint16_t h;
      std::memcpy(&h, &out[i * 2], 2);
      got[i] = io == TfType::F16 ? litertHalfToFloat(h) : litertBf16ToFloat(h);
    }
    return true;
  case TfType::I8:
  {
    if (out.size() != got.size())
      return false;
    const float scale = litertOutScale(kDim, 8);
    for (size_t i = 0; i < got.size(); i++)
      got[i] = scale * (float)(int8_t)out[i];
    return true;
  }
  case TfType::I16:
  {
    if (out.size() != got.size() * 2)
      return false;
    const float scale = litertOutScale(kDim, 16);
    for (size_t i = 0; i < got.size(); i++)
    {
      int16_t code;
      std::memcpy(&code, &out[i * 2], 2);
      got[i] = scale * (float)code;
    }
    return true;
  }
  default:
    (void)act;
    return false;
  }
}

// What the marker says of a wrong answer: never written (it survived two
// runs), written only after the run returned (a second run's read was
// right), or neither -- an answer written, and wrong.
enum class Miss { Wrong, Never, Late };
Miss missOf(const LitertPeak::AnswerCheck &c)
{
  if (c.unwritten >= kUnwrittenShare && c.unwrittenRerun >= kUnwrittenShare)
    return Miss::Never;
  if (c.unwritten >= kUnwrittenShare && c.rerunPpm >= 0.0 && c.rerunPpm < clpeak::kWrongAnswerPpm)
    return Miss::Late;
  return Miss::Wrong;
}

// What the rate rows do without a form whose answer is wrong (gemm.cpp,
// conv.cpp, block.cpp), as the accuracy row says it.
constexpr const char *kRowConsequence =
    "so the rate rows leave this form of the format out, and are withheld where no form of it answers right";

// The reason the accuracy row of a wrong answer is filed under, its figure
// included: what the marker said of it, or what it read.
std::string wrongRowReason(const LitertPeak::AnswerCheck &c)
{
  char pct[32];
  const std::string figure =
      c.ppm >= 0.0 ? " (" + std::to_string((long long)c.ppm) + " ppm against " + kWords.refFull + ")" : "";
  if (missOf(c) == Miss::Never)
  {
    std::snprintf(pct, sizeof pct, "%.0f%%", 100.0 * std::min(c.unwritten, c.unwrittenRerun));
    return "this accelerator never wrote its answer" + figure + ": after two runs, " + pct +
           " of the output still held what clpeak wrote there before the first -- a runtime that does not "
           "return the result, not one that loses precision -- " + kRowConsequence;
  }
  if (missOf(c) == Miss::Late)
    return "this accelerator's answer landed only after the run that computed it had returned" + figure +
           ": the output still held what clpeak wrote there before the run, and a second run's read was "
           "right -- its timings would not cover the work, " + kRowConsequence;
  return clpeak::wrongAnswerRowReason(c, kWords, kRowConsequence);
}

// A form, in the words a reason or a note gives it after the format's label.
std::string formWords(const LitertForm &form)
{
  std::string s;
  if (form.conv1x1)
    s += " written as a 1x1 convolution";
  if (form.gpuInt8Kernels)
    s += std::string(s.empty() ? "" : ",") + " with the GPU's 8-bit kernels allowed";
  if (form.floatIo)
    s += std::string(s.empty() ? "" : ",") + " with float inputs and outputs";
  return s;
}

} // namespace

const LitertPeak::AnswerCheck &LitertPeak::answerCheck(const LitertRuntime &rt, const litert_device_info_t &dev,
                                                       LitertFormat f, const LitertForm &form)
{
  using clpeak_tflite::TfType;
  const LitertPlan base = litertPlanFor(f, dev.accel);
  // A choice the plan does not have is the form it does have, so one graph
  // and one session is checked once whichever way it is asked for.
  LitertForm k = form;
  k.gpuInt8Kernels = k.gpuInt8Kernels && base.gpuInt8KernelChoice;
  k.floatIo = k.floatIo && litertIoType(litertFormPlan(base, k)) != base.act;
  const auto key = std::make_tuple((int)dev.accel, (int)f, k.conv1x1, k.gpuInt8Kernels, k.floatIo);
  auto found = answerChecks_.find(key);
  if (found != answerChecks_.end())
    return found->second;
  AnswerCheck &c = answerChecks_[key];
  if (!base.applies)
  {
    c.status = ResultStatus::Unsupported;
    c.error = base.whyNot;
    return c;
  }
  const LitertPlan plan = litertFormPlan(base, k);
  const TfType io = litertIoType(plan);
  const std::string label = std::string(litertFormatLabel(f)) + formWords(k);

  std::vector<float> weights;   // exactly what the accelerator multiplies
  std::string err;
  auto s = LitertSession::create(rt, dev,
                                 litertPlainMatMulModel(plan, kDim, kDim, kDim, &weights, 0x85a308d3u, k.conv1x1),
                                 litertConfigFor(plan, k), err);
  if (!s)
  {
    c.status = ResultStatus::Unsupported;
    c.error = err;
    return c;
  }
  if (!s->onDevice())
  {
    c.status = ResultStatus::Unsupported;
    c.error = s->offDevice();
    return c;
  }

  // The activations: the same generator as the speed rows, rounded to the
  // width the model takes them in, which is the width the device sees -- an
  // integer plan's codes, handed over as codes or, under floatIo, as the
  // float values they stand for, which its QUANTIZE maps back exactly.
  const int64_t count = kDim * kDim;
  std::vector<double> a((size_t)count);
  std::string raw;
  const int abits = (int)clpeak_tflite::tfliteElementBits(plan.act);
  switch (plan.act)
  {
  case TfType::F32:
    raw.resize((size_t)count * 4);
    for (int64_t i = 0; i < kDim; i++)
      for (int64_t j = 0; j < kDim; j++)
      {
        float v = litertValueAt(i, j, 0x243f6a88u);
        // The GPU's fp16 policies round every operand to half; done here
        // first, the conversion is exact and the rounding cancels.
        if (plan.halfRounded)
          v = litertHalfToFloat(litertFloatToHalf(v));
        std::memcpy(&raw[(size_t)(i * kDim + j) * 4], &v, 4);
        a[(size_t)(i * kDim + j)] = v;
      }
    break;
  case TfType::F16:
  case TfType::BF16:
    raw.resize((size_t)count * 2);
    for (int64_t i = 0; i < kDim; i++)
      for (int64_t j = 0; j < kDim; j++)
      {
        const float v = litertValueAt(i, j, 0x243f6a88u);
        const uint16_t h = plan.act == TfType::F16 ? litertFloatToHalf(v) : litertFloatToBf16(v);
        std::memcpy(&raw[(size_t)(i * kDim + j) * 2], &h, 2);
        a[(size_t)(i * kDim + j)] = plan.act == TfType::F16 ? litertHalfToFloat(h) : litertBf16ToFloat(h);
      }
    break;
  case TfType::I8:
  case TfType::I16:
  {
    const float scale = litertActScale(abits);
    const long qmax = (1L << (abits - 1)) - 1;
    const size_t eb = io == TfType::F32 ? 4 : (size_t)(abits / 8);
    raw.resize((size_t)count * eb);
    for (int64_t i = 0; i < kDim; i++)
      for (int64_t j = 0; j < kDim; j++)
      {
        const float v = litertValueAt(i, j, 0x243f6a88u);
        long q = std::lround(v / scale);
        q = std::max(-qmax, std::min(qmax, q));
        char *dst = &raw[(size_t)(i * kDim + j) * eb];
        if (io == TfType::F32)
        {
          const float fv = scale * (float)q;
          std::memcpy(dst, &fv, 4);
        }
        else if (abits == 8)
        {
          const int8_t code = (int8_t)q;
          std::memcpy(dst, &code, 1);
        }
        else
        {
          const int16_t code = (int16_t)q;
          std::memcpy(dst, &code, 2);
        }
        a[(size_t)(i * kDim + j)] = scale * (double)q;
      }
    break;
  }
  default:
    c.status = ResultStatus::Error;
    c.error = "unexpected activation type";
    return c;
  }

  // The marker goes in first, so an answer never written reads as one
  // (AnswerCheck::unwritten).  A buffer that cannot be marked leaves the
  // check as it was.
  std::string markErr;
  const bool marked = s->fillOutput(0, kUnwrittenByte, markErr);
  std::vector<uint8_t> out;
  if (!s->writeInput(0, raw.data(), raw.size(), err) || !s->run(err) || !s->outputBytes(0, out, err))
  {
    // A kernel that declines the type at its first inference ("failed to
    // prepare") is a capability, as in every rate row.
    c.status = litertFailureStatus(err);
    c.error = err;
    return c;
  }
  const size_t elemBytes = litertElemBytes(io, 1);
  if (marked)
    c.unwritten = unwrittenShare(out, elemBytes);

  std::vector<float> got((size_t)count);
  if (!decodeOutput(out, io, plan.act, got))
  {
    c.status = ResultStatus::Error;
    c.error = "unexpected output size";
    return c;
  }

  std::vector<double> ref;
  referenceGemm(a, weights, kDim, kDim, kDim, ref);
  // The reference multiplies values under 1 in magnitude, in double, so a NaN
  // or an infinity in the figure is the accelerator's; and no sum here can
  // pass 256, which overflows no width a kernel accumulates in.  It is a
  // wrong answer, not lost precision, and wrongAnswer() refuses the format's
  // rates for it as it does for one past the line.
  clpeak::judgeAnswer(c, got.data(), ref.data(), got.size(), kWords);
  if (c.status != ResultStatus::Ok && !c.nonFinite)
    return c;

  // A wrong answer is run once more and read again, the marker left as the
  // first run left it: an answer never written still reads as the marker,
  // and one written only after the run returned reads right this time.
  if (c.wrong())
  {
    std::vector<uint8_t> again;
    std::vector<float> got2((size_t)count);
    std::string rerr;
    if (s->run(rerr) && s->outputBytes(0, again, rerr) && decodeOutput(again, io, plan.act, got2))
    {
      if (marked)
        c.unwrittenRerun = unwrittenShare(again, elemBytes);
      const double p2 = clpeak::answerPpm(got2.data(), ref.data(), got2.size());
      c.rerunPpm = std::isfinite(p2) ? p2 : -1.0;
    }
    CLPEAK_VLOG("litert-numeric-error[%s/%s]: wrong answer, %.0f ppm%s; %.1f%% of the output still held the "
                "marker after the run, %.1f%% after a second run, which read %.0f ppm\n",
                dev.displayName.c_str(), label.c_str(), c.ppm, c.nonFinite ? " (NaN or infinity)" : "",
                100.0 * c.unwritten, 100.0 * c.unwrittenRerun, c.rerunPpm);
  }
  else
    CLPEAK_VLOG("litert-numeric-error[%s/%s]: %.2f ppm\n", dev.displayName.c_str(), label.c_str(), c.ppm);
  return c;
}

LitertForm LitertPeak::resolveIo(const LitertRuntime &rt, const litert_device_info_t &dev, LitertFormat f,
                                 LitertForm form)
{
  const LitertPlan plan = litertPlanFor(f, dev.accel);
  form.floatIo = false;
  if (!plan.applies || litertIoType(litertFormPlan(plan, LitertForm{false, false, true})) == plan.act)
    return form;
  auto right = [](const AnswerCheck &c) { return c.status == ResultStatus::Ok && c.ppm >= 0.0 && !c.wrong(); };
  if (right(answerCheck(rt, dev, f, form)))
    return form;
  LitertForm floated = form;
  floated.floatIo = true;
  if (right(answerCheck(rt, dev, f, floated)))
    return floated;
  return form;
}

std::string LitertPeak::wrongAnswer(const LitertRuntime &rt, const litert_device_info_t &dev, LitertFormat f,
                                    const LitertForm &form)
{
  const AnswerCheck &c = answerCheck(rt, dev, f, form);
  if (!c.wrong())
    return std::string();
  const std::string what = std::string(litertFormatLabel(f)) + formWords(form);
  char pct[32];
  // The marker clpeak wrote before the run said where the answer went.
  if (missOf(c) == Miss::Never)
  {
    std::snprintf(pct, sizeof pct, "%.0f%%", 100.0 * std::min(c.unwritten, c.unwrittenRerun));
    return "this accelerator never wrote its answer for " + what + ": after two runs, " + pct +
           " of the output still held what clpeak wrote there before the first -- a runtime that does "
           "not return the result, not one that loses precision -- so its rate is not reported";
  }
  if (missOf(c) == Miss::Late)
    return "this accelerator's answer for " + what + " landed only after the run that computed it had "
           "returned: the output still held what clpeak wrote there before the run, and a second run's "
           "read was right -- its timings would not cover the work, so its rate is not reported";
  return clpeak::wrongAnswerReason(c, kWords, what);
}

int LitertPeak::runNumericError(const LitertRuntime &rt, const litert_device_info_t &dev,
                                benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"litert_numeric_error", "LiteRT matmul numeric error", "ppm", Category::Compute,
       "How far each format's answer on a 1024-cubed matmul drifts from a "
       "double-precision reference, in parts per million.  The reference "
       "multiplies the exact values this accelerator was given, so only the "
       "arithmetic and the width the answer was kept in remain.",
       TestShape::Heterogeneous, "model format"});

  // What another form of the same product read, for the clause that quotes
  // it: its figure, or why it has none.
  auto reading = [&](LitertFormat f, const LitertForm &form) -> std::string {
    const AnswerCheck &c = answerCheck(rt, dev, f, form);
    if (c.wrong() && missOf(c) == Miss::Never)
      return "no answer";
    if (c.wrong() && missOf(c) == Miss::Late)
      return "a late answer";
    if (c.wrong())
      return "a wrong answer";
    if (c.ppm < 0.0)
      return "nothing";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.0f ppm", c.ppm);
    return buf;
  };

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;
    const char *label = litertFormatLabel(v.f);
    const LitertPlan plan = litertPlanFor(v.f, dev.accel);
    // The row reads FULLY_CONNECTED with the GPU's 8-bit kernels disallowed,
    // in the inputs and outputs the format answers right with (resolveIo).
    const LitertForm form = resolveIo(rt, dev, v.f, LitertForm());
    const AnswerCheck &c = answerCheck(rt, dev, v.f, form);
    std::string note = v.note;
    if (c.ppm < 0.0 && !c.wrong())
    {
      test.skip(label, c.status, c.error, note);
      continue;
    }
    // One sentence after the format's: an integer format the accelerator
    // answered only with float inputs and outputs, and what its own int8 or
    // int16 ones gave; then the other forms the rate rows race -- int8's
    // layers as 1x1 convolutions (gemm.cpp), and on the GPU its 8-bit kernels.
    std::string extra;
    if (form.floatIo)
    {
      LitertForm own = form;
      own.floatIo = false;
      extra = "Float inputs and outputs, as " +
              std::string(plan.act == clpeak_tflite::TfType::I16 ? "int16" : "int8") + " ones gave " +
              reading(v.f, own);
    }
    const bool conv = v.f == LitertFormat::Int8Qdq;
    std::vector<std::pair<std::string, std::string>> alts;   // (reading, how)
    if (conv)
      alts.push_back({reading(v.f, resolveIo(rt, dev, v.f, LitertForm{true, false, false})), "as a 1x1 convolution"});
    if (plan.gpuInt8KernelChoice)
      alts.push_back({reading(v.f, resolveIo(rt, dev, v.f, LitertForm{false, true, false})),
                      "with 8-bit kernels"});
    if (conv && plan.gpuInt8KernelChoice)
      alts.push_back({reading(v.f, resolveIo(rt, dev, v.f, LitertForm{true, true, false})), "both ways"});
    if (!alts.empty())
    {
      extra += extra.empty() ? "Other forms gave " : "; other forms gave ";
      for (size_t i = 0; i < alts.size(); i++)
        extra += std::string(i == 0 ? "" : (i + 1 == alts.size() ? " and " : ", ")) + alts[i].first + " " +
                 alts[i].second;
    }
    if (!extra.empty())
      note += "  " + extra + ".";
    // A wrong answer is no precision figure: the row is an Error carrying
    // it, beside the rate rows it withheld.
    if (c.wrong())
    {
      test.skip(label, ResultStatus::Error, wrongRowReason(c), note);
      continue;
    }
    test.emit(label, (float)c.ppm, note.c_str());
  }

  test.end();
  return 0;
}

#endif // ENABLE_LITERT
