#ifdef ENABLE_ONNX

// onnx-numeric-error: how much accuracy each datatype actually costs on this
// execution provider, measured as relative RMS error against an fp32
// reference computed on the CPU EP from the very same values.
//
// A TOPS figure without this is half a number: int8 is fast because it threw
// precision away, and how much it threw away is not visible from the speed.
//
// The fp32 row does a second job.  clpeak's CPU-fallback guard works at
// ORT's partitioning level, so it cannot see an EP that accepts a node and
// then quietly computes it at lower precision internally -- Core ML running
// an fp32 graph on the fp16 Neural Engine is the standard example.  An fp32
// row reading far above the low single digits of ppm is that downgrade
// showing up as a measurement instead of a footnote.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace
{

// Fixed size for every provider: the error being measured depends on the
// accumulation depth K, so it has to be the same K everywhere or the rows
// are not comparable.  1024 is deep enough for fp16 accumulation error to
// be real and small enough that the fp32 CPU reference stays quick.
constexpr int64_t kDim = 1024;

struct Variant
{
  int         dtype;
  bool        qdq;
  const char *label;
  const char *note;
};

size_t dtypeSize(int dtype)
{
  switch (dtype)
  {
  case ONNX_DT_FLOAT:                          return 4;
  case ONNX_DT_FLOAT16: case ONNX_DT_BFLOAT16: return 2;
  default:                                     return 1;
  }
}

// Same generator as the GEMM test, so the two tests describe the same work.
void fillTensor(std::string &raw, int dtype, int64_t count, uint32_t seed)
{
  uint32_t s = seed;
  raw.resize((size_t)count * dtypeSize(dtype));
  float    *f = reinterpret_cast<float *>(&raw[0]);
  uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
  int8_t   *q = reinterpret_cast<int8_t *>(&raw[0]);
  uint8_t  *u = reinterpret_cast<uint8_t *>(&raw[0]);
  for (int64_t i = 0; i < count; i++)
  {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float v = (float)(s >> 8) / 16777216.0f - 0.5f;
    switch (dtype)
    {
    case ONNX_DT_FLOAT:    f[i] = v; break;
    case ONNX_DT_FLOAT16:  h[i] = floatToHalf(v); break;
    case ONNX_DT_BFLOAT16: h[i] = floatToBf16(v); break;
    case ONNX_DT_UINT8:    u[i] = (uint8_t)(v * 254.0f + 128.0f); break;
    default:               q[i] = (int8_t)(v * 254.0f); break;
    }
  }
}

float qdqOutputScale(int64_t K)
{
  return (float)(4.0 * std::sqrt((double)K) / 3.0 / 127.0);
}

// Exact widening of a stored tensor to fp32 -- these are the values the
// reduced-precision run actually saw, so the reference must use them and not
// the fp32 originals they were rounded from.
std::vector<float> widen(const std::string &raw, int dtype, int64_t count,
                         float scale)
{
  std::vector<float> out((size_t)count);
  const float    *f = reinterpret_cast<const float *>(raw.data());
  const uint16_t *h = reinterpret_cast<const uint16_t *>(raw.data());
  const int8_t   *q = reinterpret_cast<const int8_t *>(raw.data());
  const uint8_t  *u = reinterpret_cast<const uint8_t *>(raw.data());
  for (int64_t i = 0; i < count; i++)
  {
    switch (dtype)
    {
    case ONNX_DT_FLOAT:   out[i] = f[i]; break;
    case ONNX_DT_FLOAT16: out[i] = halfToFloat(h[i]); break;
    case ONNX_DT_UINT8:   out[i] = ((float)u[i] - 128.0f) * scale; break;
    default:              out[i] = (float)q[i] * scale; break;
    }
  }
  return out;
}

// Run one already-built model on one EP, returning the raw output bytes.
// `error` is set (and the vector left empty) on any failure.
std::vector<uint8_t> runOnce(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                             const std::string &modelBytes,
                             const char *inName, const char *outName,
                             const void *inData, size_t inBytes,
                             int ioDtype, int64_t rows, int64_t cols,
                             std::string &error)
{
  std::vector<uint8_t> out;

  auto ses = onnxCreateSession(rt, ep, modelBytes);
  if (!ses.session)
  {
    error = ses.error;
    return out;
  }

  out.resize((size_t)rows * cols * dtypeSize(ioDtype));

  OrtMemoryInfo *mi = nullptr;
  OrtValue *inVal = nullptr, *outVal = nullptr;
  OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault, &mi);
  if (!st)
  {
    const int64_t inShape[2]  = {rows, cols};
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, const_cast<void *>(inData), inBytes, inShape, 2,
        (ONNXTensorElementDataType)ioDtype, &inVal);
  }
  if (!st)
  {
    const int64_t outShape[2] = {rows, cols};
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, out.data(), out.size(), outShape, 2,
        (ONNXTensorElementDataType)ioDtype, &outVal);
  }
  if (!st)
  {
    const char *ins[]  = {inName};
    const char *outs[] = {outName};
    st = rt.api->Run(ses.session, nullptr, ins,
                     (const OrtValue *const *)&inVal, 1, outs, 1, &outVal);
  }
  if (st)
  {
    error = onnxStatusText(rt, st);
    out.clear();
  }

  if (inVal)  rt.api->ReleaseValue(inVal);
  if (outVal) rt.api->ReleaseValue(outVal);
  if (mi)     rt.api->ReleaseMemoryInfo(mi);
  rt.api->ReleaseSession(ses.session);
  return out;
}

// Relative RMS error in parts per million.  RMS rather than max-abs: one
// unlucky cancellation in a million elements says much less about a
// datatype than the error across the whole result does.
double relativeRmsPpm(const std::vector<float> &got,
                      const std::vector<float> &ref)
{
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < got.size(); i++)
  {
    const double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  if (den <= 0.0)
    return -1.0;
  return std::sqrt(num / den) * 1.0e6;
}

} // namespace

int OnnxPeak::runNumericError(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                              benchmark_config_t &cfg)
{
  (void)cfg;

  static const Variant kVariants[] = {
    {ONNX_DT_FLOAT,   false, "fp32",
     "Should be near zero -- an fp32 graph compared against fp32.  A reading "
     "well above single digits means the provider is not really computing in "
     "fp32, but quietly downgrading internally."},
    {ONNX_DT_FLOAT16, false, "fp16",
     "What half precision costs: fp16 inputs and fp16 accumulation against "
     "the fp32 answer."},
    {ONNX_DT_INT8, true, "int8_qdq",
     "What quantization costs, end to end -- quantizing the inputs, the "
     "integer matmul, and re-quantizing the result."},
  };

  auto test = currentDeviceScope->beginTest(
      {"onnx-numeric-error", "ONNX MatMul numeric error", "ppm",
       Category::FpCompute,
       "How far each datatype's answer drifts from a full-precision one, in "
       "parts per million, on a fixed 1024x1024x1024 matrix multiply.  Speed "
       "rows alone cannot be compared honestly across datatypes: int8 is fast "
       "because it discarded precision, and this is how much.  The fp32 row "
       "also catches a provider that accepts a full-precision graph and then "
       "computes it at lower precision behind your back."});

  // The reference always runs on the CPU EP in fp32 -- it is the one path
  // present on every machine and the one whose arithmetic is not in question.
  onnx_ep_info_t cpuEp;
  cpuEp.providerKey = "CPUExecutionProvider";
  cpuEp.displayName = "ONNX Runtime CPU";
  cpuEp.deviceType  = DeviceType::Cpu;

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;

    logger::EmitOptions o;
    o.description = v.note;

    const float qScale = 1.0f / 127.0f;
    const float cScale = qdqOutputScale(kDim);

    // Quantized activations may be signed or unsigned; providers disagree
    // about which they accept, so try both.
    //
    // Unsigned first, which is the opposite of the throughput row's order and
    // deliberate.  That row picks by which scheme *fuses*; this one can only
    // pick by which one runs, and on x86 both run while only the unsigned one
    // fuses.  Choosing signed there would quietly measure a floating-point
    // multiply and report the mild error of one, hiding what unsigned int8
    // actually costs on that hardware -- the 18% loss from int16 accumulators
    // saturating, which is the whole reason this row exists.
    static const int kActDtypes[] = {ONNX_DT_UINT8, ONNX_DT_INT8};

    int aDtype = v.dtype;
    std::string aRaw, bRaw, err;
    std::vector<uint8_t> raw;
    const char *inName = "A", *outName = "C";

    if (v.qdq)
    {
      for (int cand : kActDtypes)
      {
        aDtype = cand;
        fillTensor(aRaw, aDtype, kDim * kDim, 0x9e3779b9u);
        fillTensor(bRaw, v.dtype, kDim * kDim, 0x243f6a88u);
        std::string model = onnxQdqMatMulModel(kDim, kDim, kDim, bRaw,
                                               qScale, qScale, cScale, aDtype);
        inName  = "A_q";
        outName = "C_q";
        err.clear();
        raw = runOnce(rt, ep, model, inName, outName, aRaw.data(), aRaw.size(),
                      aDtype, kDim, kDim, err);
        if (!raw.empty())
          break;
      }
    }
    else
    {
      fillTensor(aRaw, aDtype, kDim * kDim, 0x9e3779b9u);
      fillTensor(bRaw, v.dtype, kDim * kDim, 0x243f6a88u);
      std::string model = onnxMatMulModel(kDim, kDim, kDim, v.dtype, bRaw);
      raw = runOnce(rt, ep, model, inName, outName, aRaw.data(), aRaw.size(),
                    aDtype, kDim, kDim, err);
    }
    if (v.qdq)
      o.description += (aDtype == ONNX_DT_UINT8)
                           ? "  Measured with unsigned activations."
                           : "  Measured with signed activations.";

    if (raw.empty())
    {
      test.skip(v.label, ResultStatus::Unsupported,
                err.empty() ? "run failed" : err, o.description);
      continue;
    }

    // The provider's answer, widened to fp32.  For the QDQ path that means
    // undoing the output quantization -- the same thing a real pipeline's
    // next layer does.
    std::vector<float> got = widen(
        std::string(reinterpret_cast<const char *>(raw.data()), raw.size()),
        aDtype, kDim * kDim, cScale);

    // The reference: the same values the provider saw, widened to fp32 and
    // multiplied on the CPU EP.
    std::vector<float> aRef = widen(aRaw, aDtype, kDim * kDim, qScale);
    std::vector<float> bRef = widen(bRaw, v.dtype, kDim * kDim, qScale);

    std::string refWeights(reinterpret_cast<const char *>(bRef.data()),
                           bRef.size() * sizeof(float));
    std::string refModel = onnxMatMulModel(kDim, kDim, kDim, ONNX_DT_FLOAT,
                                           refWeights);
    std::string refErr;
    auto refRaw = runOnce(rt, cpuEp, refModel, "A", "C", aRef.data(),
                          aRef.size() * sizeof(float), ONNX_DT_FLOAT,
                          kDim, kDim, refErr);
    if (refRaw.empty())
    {
      test.skip(v.label, ResultStatus::Error,
                "fp32 reference failed: " + refErr, o.description);
      continue;
    }

    std::vector<float> ref((size_t)kDim * kDim);
    std::memcpy(ref.data(), refRaw.data(), refRaw.size());

    double ppm = relativeRmsPpm(got, ref);
    if (ppm < 0.0)
    {
      test.skip(v.label, ResultStatus::Error, "reference result was all zero",
                o.description);
      continue;
    }
    test.emit(v.label, (float)ppm, o);
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
