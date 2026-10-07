#ifndef CLPEAK_ANSWER_CHECK_H
#define CLPEAK_ANSWER_CHECK_H

// Whether an ML runtime's answer is right: the line, the figure and the
// words the ONNX Runtime, Core ML and LiteRT backends share.
//
// A fast kernel can be a wrong one -- LiteRT's OpenCL accelerator ran an int8
// matmul at a TOPS on a Galaxy S24 and returned 2.5 million ppm -- and a
// timing says nothing about what was computed.  So each backend's accuracy
// matmul (its numeric_error.cpp) is also a check every rate row asks before
// it publishes: measured once per device, format and form -- a weight layout,
// a 1x1-convolution spelling -- on first use, since the rate tests run before
// the accuracy one.  An answer at or past kWrongAnswerPpm, or with NaN or
// infinity in it, withholds that format's gemm, conv and block rows as an
// Error naming the figure (wrongAnswerReason), and its accuracy row is an
// Error carrying the figure too (wrongAnswerRowReason), so the two halves of
// the pair stand together.  A check that could not be measured -- refused,
// sent to another unit, a quantized matmul left unfused -- gates nothing.

#include <common/run_document.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>

namespace clpeak
{

// A relative RMS error at or past this is a wrong answer, not a loss of
// precision.  The line sits between two populations.  The costliest right
// answers are an fp4 answer, whose rounding alone costs about 12% at the
// accuracy matmul's output scale, and the int8 matmul of a pre-VNNI x86 CPU,
// whose 16-bit pair sums saturate and lose 18% on a Zen 2 -- a measured cost,
// published beside its rate (src/onnx/AGENTS.md).  A wrong answer reads far
// higher: exactly 100% for one that is zero everywhere, 141% or more for one
// unrelated to the product, 252% for the full-range int8 the S24 returned.
constexpr double kWrongAnswerPpm = 500000.0;

// One answer, measured: the relative RMS error of a matmul's answer against
// the reference its backend builds from the same operands, or why there is
// none.
struct AnswerCheck
{
  // In parts per million; -1 when it could not be measured, and then
  // `status` and `error` say why.
  double ppm = -1.0;
  // The answer held NaN or infinity where the reference is finite
  // everywhere: no figure, and the most wrong answer there is.
  bool nonFinite = false;
  // Every element of the answer was exactly zero; the figure is 1,000,000.
  bool zero = false;
  ResultStatus status = ResultStatus::Ok;
  std::string error;

  bool wrong() const { return nonFinite || ppm >= kWrongAnswerPpm; }
};

// How a backend's reasons name things.
struct AnswerWords
{
  const char *who;     // the answering unit, possessive: "this provider's"
  const char *test;    // the test that measures the answer: "onnx_numeric_error"
  const char *ref;     // its reference, in a rate row's reason: "the host's reference"
  const char *refFull; // and in the accuracy row's: "the host's double-precision reference"
};

// Relative RMS error of `got` against `ref`, `n` elements each, in ppm; -1
// when the reference is zero everywhere.  NaN or infinity in `got` makes it
// non-finite.
template <class G, class R>
double answerPpm(const G *got, const R *ref, size_t n)
{
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < n; i++)
  {
    const double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  if (den <= 0.0)
    return -1.0;
  return std::sqrt(num / den) * 1.0e6;
}

// Measure an answer into `c`: its figure and whether it is zero everywhere,
// or a non-finite answer (status Error, no figure), or a reference that is
// zero everywhere, which no operands here produce (status Error).
template <class G, class R>
void judgeAnswer(AnswerCheck &c, const G *got, const R *ref, size_t n, const AnswerWords &w)
{
  const double ppm = answerPpm(got, ref, n);
  if (ppm < 0.0)
  {
    c.status = ResultStatus::Error;
    c.error = "reference result was all zero";
    return;
  }
  // A NaN or an infinity anywhere in the answer makes the figure one too.
  // That is no error figure, and it cannot be recorded either: the document
  // would spell it `nan`, which no JSON reader -- clpeak's own included --
  // will load.
  if (!std::isfinite(ppm))
  {
    c.nonFinite = true;
    c.status = ResultStatus::Error;
    c.error = std::string(w.who) + " answer holds NaN or infinity where " + w.refFull +
              " is finite everywhere: it computed something other than this matmul, so there "
              "is no error figure to report";
    return;
  }
  c.ppm = ppm;
  bool zero = true;
  for (size_t i = 0; i < n && zero; i++)
    zero = (double)got[i] == 0.0;
  c.zero = zero;
}

namespace detail
{
inline std::string answerFigure(const AnswerCheck &c, const char *prefix, const char *ref)
{
  return std::string("(") + prefix + std::to_string((long long)c.ppm) + " ppm against " + ref + ")";
}
inline std::string answerPercent(const AnswerCheck &c)
{
  char pct[32];
  std::snprintf(pct, sizeof pct, "%.0f%%", c.ppm / 10000.0);
  return pct;
}
constexpr const char *kKernelCause =
    "a kernel that does not compute the format, not one that loses precision";
constexpr const char *kZeroCause = "an answer not computed or not returned, not one that loses precision";
} // namespace detail

// The reason a rate row is refused with, for a wrong answer `c` for `what`
// (the format, and the form where it has one); empty for a right one.
inline std::string wrongAnswerReason(const AnswerCheck &c, const AnswerWords &w, const std::string &what)
{
  if (!c.wrong())
    return std::string();
  const std::string head = std::string(w.who) + " answer for " + what;
  const std::string prefix = std::string(w.test) + ": ";
  if (c.nonFinite)
    return head + " holds NaN or infinity where " + w.ref + " is finite everywhere (" + w.test + ") -- " +
           detail::kKernelCause + " -- so its rate is not reported";
  if (c.zero)
    return head + " is zero everywhere " + detail::answerFigure(c, prefix.c_str(), w.ref) + " -- " +
           detail::kZeroCause + " -- so its rate is not reported";
  return head + " is wrong by " + detail::answerPercent(c) + " " + detail::answerFigure(c, prefix.c_str(), w.ref) +
         " -- " + detail::kKernelCause + " -- so its rate is not reported";
}

// The reason the accuracy row of a wrong answer files its Error under, the
// figure included; `consequence` says what became of the rate rows ("so its
// rate rows are withheld").  Empty for a right answer.
inline std::string wrongAnswerRowReason(const AnswerCheck &c, const AnswerWords &w,
                                        const std::string &consequence)
{
  if (!c.wrong())
    return std::string();
  if (c.nonFinite)
    return c.error;
  const std::string head = std::string(w.who) + " answer";
  if (c.zero)
    return head + " is zero everywhere " + detail::answerFigure(c, "", w.refFull) + " -- " + detail::kZeroCause +
           " -- " + consequence;
  return head + " is wrong by " + detail::answerPercent(c) + " " + detail::answerFigure(c, "", w.refFull) + " -- " +
         detail::kKernelCause + " -- " + consequence;
}

} // namespace clpeak

#endif // CLPEAK_ANSWER_CHECK_H
