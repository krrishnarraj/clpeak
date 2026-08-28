#ifdef ENABLE_ONNX

// onnx-numeric-error: how much accuracy each datatype actually costs on this
// execution provider, measured as relative RMS error against an fp32
// reference computed on the CPU EP from the very same values.
//
// A TOPS figure without this is half a number: int8 is fast because it threw
// precision away, and how much it threw away is not visible from the speed.
//
// The int8 row carries a second fact of its own: whether the provider fused a
// quantized matmul at all.  A provider that declined dequantizes the operands
// and multiplies in floating point, and the error that comes back is then the
// quantization scheme's rather than the integer unit's -- the same distinction
// onnx-gemm-int refuses to publish a rate without, and the two tests can
// disagree because they choose the activation signedness on different grounds.
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
#include <string>
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

// Only ever asked about the types that cross the graph boundary, which are
// fp32 and the plain float widths -- the quantized ones are handed over as
// fp32 and quantized on device.  Sub-byte types need onnxElemBytes(), which
// can express a half; this cannot.
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
  raw.assign((size_t)onnxElemBytes(dtype, count), '\0');
  float    *f = reinterpret_cast<float *>(&raw[0]);
  uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
  for (int64_t i = 0; i < count; i++)
  {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    float v = (float)(s >> 8) / 16777216.0f - 0.5f;
    switch (dtype)
    {
    case ONNX_DT_FLOAT:    f[i] = v; break;
    case ONNX_DT_FLOAT16:  h[i] = floatToHalf(v); break;
    case ONNX_DT_BFLOAT16: h[i] = floatToBf16(v); break;
    default:               onnxStoreQuantElem(&raw[0], i, dtype, v * 2.0f);
                           break;
    }
  }
}

float qdqOutputScale(int64_t K, int outDtype)
{
  const double top = (outDtype == ONNX_DT_FLOAT4E2M1) ? 6.0 : 127.0;
  return (float)(4.0 * std::sqrt((double)K) / 3.0 / top);
}

// Exact widening of a stored tensor to fp32 -- these are the values the
// reduced-precision run actually saw, so the reference must use them and not
// the fp32 originals they were rounded from.
//
// The switch is exhaustive and sits outside the loop.  It used to be inside
// with the integer path as `default:`, which meant any float type it had not
// been taught -- bfloat16 was the first -- was silently read as int8 and
// scaled, producing an error figure for a tensor that had been reinterpreted
// rather than widened.  An unknown type now returns empty and the caller says
// so.
std::vector<float> widen(const std::string &raw, int dtype, int64_t count,
                         float scale)
{
  std::vector<float> out;
  const float    *f = reinterpret_cast<const float *>(raw.data());
  const uint16_t *h = reinterpret_cast<const uint16_t *>(raw.data());
  const int8_t   *q = reinterpret_cast<const int8_t *>(raw.data());
  const uint8_t  *u = reinterpret_cast<const uint8_t *>(raw.data());

  switch (dtype)
  {
  case ONNX_DT_FLOAT:
    out.assign(f, f + count);
    break;
  case ONNX_DT_FLOAT16:
    out.resize((size_t)count);
    for (int64_t i = 0; i < count; i++) out[i] = halfToFloat(h[i]);
    break;
  case ONNX_DT_BFLOAT16:
    out.resize((size_t)count);
    for (int64_t i = 0; i < count; i++) out[i] = bf16ToFloat(h[i]);
    break;
  case ONNX_DT_UINT8:
    out.resize((size_t)count);
    for (int64_t i = 0; i < count; i++)
      out[i] = ((float)u[i] - 128.0f) * scale;   // zero point 128
    break;
  case ONNX_DT_INT8:
    out.resize((size_t)count);
    for (int64_t i = 0; i < count; i++) out[i] = (float)q[i] * scale;
    break;
  case ONNX_DT_FLOAT8E4M3FN:
    out.resize((size_t)count);
    for (int64_t i = 0; i < count; i++) out[i] = fp8E4M3ToFloat(u[i]) * scale;
    break;
  case ONNX_DT_FLOAT8E5M2:
    out.resize((size_t)count);
    for (int64_t i = 0; i < count; i++) out[i] = fp8E5M2ToFloat(u[i]) * scale;
    break;
  case ONNX_DT_FLOAT4E2M1:
    out.resize((size_t)count);
    for (int64_t i = 0; i < count; i++)
      out[i] = fp4E2M1ToFloat(onnxLoadNibble(raw.data(), i)) * scale;
    break;
  default:
    break;   // empty: caller reports the type as unhandled
  }
  return out;
}

// Run one already-built model on one EP, returning the raw output bytes.
// `error` is set (and the vector left empty) on any failure.  When `ops` is
// given the session is profiled and the kernels the provider actually ran are
// written to it.
std::vector<uint8_t> runOnce(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                             const std::string &modelBytes,
                             const char *inName, const char *outName,
                             const void *inData, size_t inBytes,
                             int ioDtype, int64_t rows, int64_t cols,
                             std::string &error,
                             std::vector<std::string> *ops = nullptr,
                             bool keepQdqUnfused = false)
{
  std::vector<uint8_t> out;

  auto ses = onnxCreateSession(rt, ep, modelBytes,
                               /*keepConstantsUnfolded=*/false,
                               /*profile=*/ops != nullptr, keepQdqUnfused);
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
  if (ops && !out.empty())
    *ops = onnxCollectExecutedOps(rt, ses.session);

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
    {ONNX_DT_BFLOAT16, false, "bf16",
     "The other 16-bit float: three fewer mantissa bits than fp16 in exchange "
     "for fp32's exponent range.  Expect a larger error than fp16 on values "
     "like these, which all sit comfortably inside both ranges -- bf16 buys "
     "headroom against overflow, not precision, and this shows the price."},
    {ONNX_DT_FLOAT8E4M3FN, true, "fp8_e4m3",
     "The same for 8-bit floating point, in the precision-favouring variant.  "
     "Read it beside int8: both are eight bits holding the same product, and "
     "which one loses less depends entirely on how the values are spread.  "
     "These are uniform over a fixed range, which is int8's best case and "
     "float8's worst -- float8 buys dynamic range, and uniform data has none "
     "to spend."},
    {ONNX_DT_FLOAT8E5M2, true, "fp8_e5m2",
     "The same in the range-favouring variant, with one fewer mantissa bit.  "
     "It should be about twice fp8_e4m3's error on data like this."},
    {ONNX_DT_FLOAT4E2M1, true, "fp4_e2m1",
     "The same for 4-bit floating point, which has eight magnitudes to hold "
     "the answer in.  Expect roughly four times fp8_e4m3's error: two fewer "
     "mantissa bits."},
    {ONNX_DT_INT8, true, "int8_qdq",
     "What the integer path costs once the operands are already quantized: "
     "the multiply, and storing the result back in 8 bits.  Rounding the "
     "operands is not counted -- the reference multiplies the same quantized "
     "values, so that part cancels and what is left is this hardware's doing.  "
     "Whether the multiply ran in integers at all is a separate question, and "
     "this row says which it was."},
  };

  auto test = currentDeviceScope->beginTest(
      {"onnx-numeric-error", "ONNX MatMul numeric error", "ppm",
       Category::FpCompute,
       "How far each datatype's answer drifts from a full-precision one, in "
       "parts per million, on a fixed 1024x1024x1024 matrix multiply.  Speed "
       "rows alone cannot be compared honestly across datatypes: a format is "
       "fast because it discarded precision, and this is how much.  The "
       "reference multiplies the very same operands the device was given, so "
       "what is measured is the arithmetic and the width the answer is kept "
       "in -- not the rounding of the inputs, which is a property of the "
       "format rather than of the hardware and would be identical everywhere.  "
       "The fp32 row also catches a provider that accepts a full-precision "
       "graph and then computes it at lower precision behind your back."});

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

    // The same gate onnx-gemm applies.  Without it a datatype newer than the
    // runtime reports whatever ORT says about IR versions, which names neither
    // the datatype nor the fix.
    if (std::string why = onnxDtypeUnsupportedReason(rt, v.dtype); !why.empty())
    {
      test.skip(v.label, ResultStatus::Unsupported, why, o.description);
      continue;
    }

    const float cScale = qdqOutputScale(kDim, v.dtype);

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
    // int8 alone has two spellings to choose between; every other quantized
    // format uses its own type for both operands.
    const int kInt8Acts[] = {ONNX_DT_UINT8, ONNX_DT_INT8};
    const int kSameAct[]  = {v.dtype};
    const int   *actCands = (v.dtype == ONNX_DT_INT8) ? kInt8Acts : kSameAct;
    const size_t nActCands = (v.dtype == ONNX_DT_INT8) ? 2u : 1u;

    int aDtype = v.dtype;
    std::string aRaw, bRaw, err;
    std::vector<uint8_t> raw;
    std::vector<float> aDeq;
    std::vector<std::string> ranOps;
    const char *inName = "A", *outName = "C";

    if (v.qdq)
    {
      // The quantized type never crosses the graph boundary.  The input is
      // handed over as the fp32 values it dequantizes to and quantized on
      // device; the result is dequantized before it leaves.  That costs
      // nothing in accuracy -- every value passed in is already exactly
      // representable in the target type, so the added QuantizeLinear
      // round-trips it -- and it is the only way an EP that implements a type
      // internally but refuses it at its boundary can be measured at all.
      // TensorRT is exactly that: it imports float8 initializers and answers
      // "input onnx tensor data type: 17 not supported" for a float8 input.
      for (size_t ci = 0; ci < nActCands; ci++)
      {
        aDtype = actCands[ci];
        fillTensor(aRaw, aDtype, kDim * kDim, 0x9e3779b9u);
        fillTensor(bRaw, v.dtype, kDim * kDim, 0x243f6a88u);
        std::string model = onnxQdqMatMulModel(
            kDim, kDim, kDim, bRaw, onnxQuantScaleFor(aDtype),
            onnxQuantScaleFor(v.dtype), cScale, aDtype, v.dtype,
            /*floatIo=*/true);
        inName  = "A";
        outName = "C";
        err.clear();
        ranOps.clear();
        // Anything QLinearMatMul cannot carry must not be fused into it.
        const bool unfusable = !onnxQdqFusionIsLegal(aDtype) ||
                               !onnxQdqFusionIsLegal(v.dtype);
        // What the device is given is the dequantized form of what was
        // quantized here, so the reference below and the run see one set of
        // values.
        aDeq = widen(aRaw, aDtype, kDim * kDim, onnxQuantScaleFor(aDtype));
        raw = runOnce(rt, ep, model, inName, outName, aDeq.data(),
                      aDeq.size() * sizeof(float), ONNX_DT_FLOAT, kDim, kDim,
                      err, &ranOps, unfusable);
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
    {
      if (v.dtype == ONNX_DT_INT8)
        o.description += (aDtype == ONNX_DT_UINT8)
                             ? "  Measured with unsigned activations."
                             : "  Measured with signed activations.";
      // Whether the provider fused a quantized matmul decides what this row
      // is the error *of*.  onnx-gemm-int refuses to publish a rate when the
      // fusion did not happen, because a float multiply is not an int8 rate;
      // the error is still worth reporting either way -- quantizing and
      // dequantizing costs precision whatever the multiply ran as -- but it
      // must not be read as the cost of integer arithmetic when no integer
      // arithmetic took place.  The scheme here is chosen by which one runs
      // rather than which one fuses (see above), so the two tests can and do
      // disagree on the same provider.
      if (!raw.empty() && !ranOps.empty() && !onnxOpsRanQuantizedMatMul(ranOps))
      {
        std::string joined;
        for (const auto &op : ranOps)
          joined += (joined.empty() ? "" : ", ") + op;
        o.description +=
            "  This provider did not fuse a quantized matmul: it dequantized "
            "the operands and multiplied in floating point (ran: " + joined +
            "), so this is what the quantization scheme costs rather than "
            "what this hardware's own quantized arithmetic costs.";
        CLPEAK_VLOG("onnx-numeric-error[%s/%s]: no fused quantized matmul, "
                    "executed %s\n", ep.providerKey.c_str(), v.label,
                    joined.c_str());
      }
    }

    if (raw.empty())
    {
      test.skip(v.label, ResultStatus::Unsupported,
                err.empty() ? "run failed" : err, o.description);
      continue;
    }

    // The provider's answer, widened to fp32.  For the QDQ path that means
    // undoing the output quantization -- the same thing a real pipeline's
    // next layer does.
    // The quantized path returns fp32 already dequantized; the plain one
    // returns its own dtype and is widened here.
    std::vector<float> got;
    if (v.qdq)
    {
      got.resize((size_t)kDim * kDim);
      std::memcpy(got.data(), raw.data(), got.size() * sizeof(float));
    }
    else
    {
      got = widen(std::string(reinterpret_cast<const char *>(raw.data()),
                              raw.size()),
                  aDtype, kDim * kDim, cScale);
    }

    // The reference: the same values the provider saw, widened to fp32 and
    // multiplied on the CPU EP.
    std::vector<float> aRef = v.qdq
        ? aDeq
        : widen(aRaw, aDtype, kDim * kDim, onnxQuantScaleFor(aDtype));
    std::vector<float> bRef = widen(bRaw, v.dtype, kDim * kDim,
                                    onnxQuantScaleFor(v.dtype));
    if (got.empty() || aRef.empty() || bRef.empty())
    {
      test.skip(v.label, ResultStatus::Error,
                "this datatype has no widening to fp32 here, so no reference "
                "can be built for it", o.description);
      continue;
    }

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
