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
    {LitertFormat::Fp32,
     "Full precision; near zero, and the control the other rows are read "
     "against.  A figure in the hundreds of ppm here is an accelerator "
     "substituting a narrower type for the fp32 it was asked for."},
    {LitertFormat::Fp16,
     "16-bit inputs and a 16-bit answer.  Around 200 ppm is an answer "
     "accumulated in fp32 and rounded once at the end; ten times that is a "
     "unit accumulating in fp16 all the way."},
    {LitertFormat::Fp16Acc32,
     "The GPU's fp16 policy with fp32 accumulation, if it changes anything: "
     "the difference from the fp16 row is what the accumulator width was "
     "worth."},
    {LitertFormat::Bf16,
     "bfloat16 has three fewer mantissa bits than fp16; eight times the fp16 "
     "figure is what that costs, if any kernel takes it."},
    {LitertFormat::Int8Qdq,
     "8-bit weights and 8-bit activations, with the answer itself quantized "
     "to 8 bits.  Around 9000 ppm is what keeping the result in int8 costs on "
     "this data; the quantization of the operands is not counted."},
    {LitertFormat::Int16x8,
     "16-bit activations over 8-bit weights, the answer kept in 16 bits; the "
     "result quantization costs about 256 times less than int8_qdq's."},
    {LitertFormat::Int8Weight,
     "8-bit weights against float activations.  On an accelerator that "
     "multiplies in float this reads near the fp16 or fp32 row; on XNNPACK, "
     "which quantizes the activations to int8 as it goes, that rounding is "
     "counted, and this is what dynamic-range quantization costs there."},
    {LitertFormat::Int4Weight,
     "4-bit blockwise weights against float activations.  The codes are "
     "shared with the reference, so whatever separates this from int8_weight "
     "is the accelerator's arithmetic, not the format."},
    {LitertFormat::Fp8Weight,
     "8-bit float weights, if any kernel takes them."},
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

} // namespace

int LitertPeak::runNumericError(const LitertRuntime &rt, const litert_device_info_t &dev,
                                benchmark_config_t &cfg)
{
  (void)cfg;
  using clpeak_tflite::TfType;

  auto test = currentDeviceScope->beginTest(
      {"litert_numeric_error", "LiteRT matmul numeric error", "ppm", Category::Compute,
       "How far each format's answer drifts from a full-precision one, in parts "
       "per million, on a fixed 1024-cubed matrix multiply -- what the speed "
       "rows cost.  The reference multiplies the values the accelerator was "
       "handed, in double precision on the host, so this is the arithmetic and "
       "the width the answer was kept in; where an accelerator quantizes the "
       "activations itself, inside the kernel, that rounding is counted too.",
       TestShape::Heterogeneous, "model format"});

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;
    const char *label = litertFormatLabel(v.f);
    const LitertPlan plan = litertPlanFor(v.f, dev.accel);
    if (!plan.applies)
    {
      test.skip(label, ResultStatus::Unsupported, plan.whyNot, v.note);
      continue;
    }

    std::vector<float> weights;   // exactly what the accelerator multiplies
    std::string err;
    auto s = LitertSession::create(rt, dev, litertPlainMatMulModel(plan, kDim, kDim, kDim, &weights),
                                   litertConfigFor(plan), err);
    if (!s)
    {
      test.skip(label, ResultStatus::Unsupported, err, v.note);
      continue;
    }
    if (!s->onDevice())
    {
      test.skip(label, ResultStatus::Unsupported, s->offDevice(), v.note);
      continue;
    }

    // The activations: the same generator as the speed rows, rounded to the
    // width the model takes them in, which is the width the device sees.
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
          const float f = litertValueAt(i, j, 0x243f6a88u);
          std::memcpy(&raw[(size_t)(i * kDim + j) * 4], &f, 4);
          a[(size_t)(i * kDim + j)] = f;
        }
      break;
    case TfType::F16:
    case TfType::BF16:
      raw.resize((size_t)count * 2);
      for (int64_t i = 0; i < kDim; i++)
        for (int64_t j = 0; j < kDim; j++)
        {
          const float f = litertValueAt(i, j, 0x243f6a88u);
          const uint16_t h = plan.act == TfType::F16 ? litertFloatToHalf(f) : litertFloatToBf16(f);
          std::memcpy(&raw[(size_t)(i * kDim + j) * 2], &h, 2);
          a[(size_t)(i * kDim + j)] = plan.act == TfType::F16 ? litertHalfToFloat(h) : litertBf16ToFloat(h);
        }
      break;
    case TfType::I8:
    case TfType::I16:
    {
      const float scale = litertActScale(abits);
      const long qmax = (1L << (abits - 1)) - 1;
      raw.resize((size_t)count * (abits / 8));
      for (int64_t i = 0; i < kDim; i++)
        for (int64_t j = 0; j < kDim; j++)
        {
          const float f = litertValueAt(i, j, 0x243f6a88u);
          long q = std::lround(f / scale);
          q = std::max(-qmax, std::min(qmax, q));
          if (abits == 8)
          {
            const int8_t c = (int8_t)q;
            std::memcpy(&raw[(size_t)(i * kDim + j)], &c, 1);
          }
          else
          {
            const int16_t c = (int16_t)q;
            std::memcpy(&raw[(size_t)(i * kDim + j) * 2], &c, 2);
          }
          a[(size_t)(i * kDim + j)] = scale * (double)q;
        }
      break;
    }
    default:
      test.skip(label, ResultStatus::Error, "unexpected activation type", v.note);
      continue;
    }

    if (!s->writeInput(0, raw.data(), raw.size(), err) || !s->run(err))
    {
      test.skip(label, ResultStatus::Error, err, v.note);
      continue;
    }
    std::vector<uint8_t> out;
    if (!s->outputBytes(0, out, err))
    {
      test.skip(label, ResultStatus::Error, err, v.note);
      continue;
    }
    s.reset();

    std::vector<float> got((size_t)count);
    bool sized = true;
    switch (plan.act)
    {
    case TfType::F32:
      sized = out.size() == got.size() * 4;
      if (sized)
        std::memcpy(got.data(), out.data(), out.size());
      break;
    case TfType::F16:
    case TfType::BF16:
      sized = out.size() == got.size() * 2;
      if (sized)
        for (size_t i = 0; i < got.size(); i++)
        {
          uint16_t h;
          std::memcpy(&h, &out[i * 2], 2);
          got[i] = plan.act == TfType::F16 ? litertHalfToFloat(h) : litertBf16ToFloat(h);
        }
      break;
    case TfType::I8:
    {
      sized = out.size() == got.size();
      const float scale = litertOutScale(kDim, 8);
      if (sized)
        for (size_t i = 0; i < got.size(); i++)
          got[i] = scale * (float)(int8_t)out[i];
      break;
    }
    case TfType::I16:
    {
      sized = out.size() == got.size() * 2;
      const float scale = litertOutScale(kDim, 16);
      if (sized)
        for (size_t i = 0; i < got.size(); i++)
        {
          int16_t c;
          std::memcpy(&c, &out[i * 2], 2);
          got[i] = scale * (float)c;
        }
      break;
    }
    default:
      sized = false;
    }
    if (!sized)
    {
      test.skip(label, ResultStatus::Error, "unexpected output size", v.note);
      continue;
    }

    std::vector<double> ref;
    referenceGemm(a, weights, kDim, kDim, kDim, ref);
    const double ppm = relativeRmsPpm(got, ref);
    if (ppm < 0.0 || !std::isfinite(ppm))
    {
      test.skip(label, ResultStatus::Error, "the result was not finite", v.note);
      continue;
    }
    CLPEAK_VLOG("litert-numeric-error[%s/%s]: %.2f ppm\n", dev.displayName.c_str(), label, ppm);
    test.emit(label, (float)ppm, v.note);
  }

  test.end();
  return 0;
}

#endif // ENABLE_LITERT
