#ifdef ENABLE_ONNX

#include "gemm_setup.h"

#include <onnx/onnx_peak.h>

#include <chrono>
#include <cmath>
#include <cstring>

namespace onnxgemm
{

const Variant kFpVariants[] = {
    {ONNX_DT_FLOAT, false, "fp32",
     "FP32 graph inputs and outputs.  A provider may use a narrower internal "
     "format; read the fp32 numeric-error row beside this one to see whether "
     "it did.  Many providers cannot run this on their matrix hardware at "
     "all, or route it away from that hardware -- that is a finding, not "
     "a failure.",
     0, false},
    {ONNX_DT_FLOAT16, false, "fp16",
     "16-bit floats, the native currency of most matrix hardware.", 0, false},
    {ONNX_DT_BFLOAT16, false, "bf16",
     "The 16-bit float with fp32's exponent range and three fewer mantissa "
     "bits.  Modern matrix hardware usually runs it at the fp16 rate; a "
     "provider that falls well short of its own fp16 row is emulating it, "
     "and one that refuses it outright has no bf16 path at all.",
     0, false},
    {ONNX_DT_FLOAT8E4M3FN, true, "fp8_e4m3",
     "8-bit floats in QDQ form, in the variant that spends its bits on "
     "precision: four exponent bits and three of mantissa, reaching 448.  "
     "This is the format quantized inference actually uses when it moves below "
     "16 bits without going to integers.",
     0, false},
    {ONNX_DT_FLOAT8E5M2, true, "fp8_e5m2",
     "The other 8-bit float, trading a mantissa bit for an exponent one: it "
     "reaches 57344 and rounds more coarsely.  Hardware usually runs both at "
     "the same rate, so a difference between these two rows is the provider "
     "choosing different machinery, and the accuracy rows say what each costs.",
     0, false},
    {ONNX_DT_FLOAT4E2M1, true, "fp4_e2m1",
     "4-bit floating point on both operands: two exponent bits, one of "
     "mantissa, eight magnitudes in all and a largest value of 6.  This is the "
     "narrowest format current tensor cores implement, and unlike int4 there "
     "is a chance a provider fuses it into a real 4-bit multiply rather than "
     "unpacking it -- the row says which happened.",
     0, false},
    {ONNX_DT_FLOAT4E2M1, false, "nvfp4",
     "NVIDIA's 4-bit block format on both operands: E2M1 values, an 8-bit "
     "float scale for every 16 of them along the reduction axis, and one more "
     "scale for the whole tensor.  Two levels are what let four bits carry a "
     "real model, and this is the arrangement a float4 tensor core expects -- "
     "so unlike every other narrow row here, a number in it would be genuine "
     "4-bit arithmetic rather than four bits unpacked into something wider.",
     /*blockSize=*/16, /*nvfp4=*/true},
    {ONNX_DT_FLOAT4E2M1, false, "fp4_weight",
     "The same 4-bit float used only for the weights, one scale per 32 of "
     "them, against 16-bit activations.  Directly comparable with the int4 "
     "row above it: identical geometry, identical block size, and the only "
     "difference is whether those four bits are spent on a float or an "
     "integer.",
     /*blockSize=*/32, false},
    {ONNX_DT_INT4, false, "int4_weight",
     "4-bit weights with one scale per 32 of them, against 16-bit activations "
     "-- the form quantized language models actually ship in.  The arithmetic "
     "is still 16-bit, because ONNX has no 4-bit multiply and the weights are "
     "unpacked on the way in, so this row is reported in TFLOPS and what four "
     "bits buys is a quarter of the weight traffic rather than a faster "
     "multiply.  On a square problem like this one that mostly shows up as "
     "matching the fp16 row; a provider well below it is unpacking badly.",
     /*blockSize=*/32, false},
};
const size_t kFpVariantCount = sizeof(kFpVariants) / sizeof(kFpVariants[0]);

const Variant kIntVariants[] = {
    {ONNX_DT_INT8, true, "int8_qdq",
     "8-bit integers in QDQ form -- quantized in, quantized out, the shape "
     "quantized inference actually ships in.  This is what vendors usually "
     "quote headline TOPS figures for.",
     0, false},
};
const size_t kIntVariantCount = sizeof(kIntVariants) / sizeof(kIntVariants[0]);

// int8 has two spellings and no provider takes both.  The float8 formats have
// one: activations and weights share the type, and there is no signed/unsigned
// question because they are signed floats.
static const QuantScheme kInt8Schemes[] = {
    {ONNX_DT_INT8, ONNX_DT_INT8, "signed activations"},    // TensorRT, ARM
    {ONNX_DT_UINT8, ONNX_DT_INT8, "unsigned activations"}, // x86 without VNNI
};

size_t schemesFor(const Variant &v, QuantScheme out[2])
{
  if (v.nvfp4)
  {
    out[0] = {v.dtype, v.dtype, "both operands blocked, with a global scale"};
    return 1;
  }
  if (v.blockSize > 0)
  {
    // Weight-only has no activation type to choose: the activations are fp16
    // and only the weights are narrow.
    out[0] = {ONNX_DT_FLOAT16, v.dtype, "16-bit activations against blocked weights"};
    return 1;
  }
  if (v.dtype == ONNX_DT_INT8)
  {
    out[0] = kInt8Schemes[0];
    out[1] = kInt8Schemes[1];
    return 2;
  }
  out[0] = {v.dtype, v.dtype, "matching activations and weights"};
  return 1;
}

std::vector<OnnxLiveShape> liveShapesFor(const Variant &v)
{
  // Result-scaled first everywhere: it is the fastest shape on a provider
  // that does not fold (a live operand costs a pass, and on the ANE compiles
  // to a ~20% slower program), so it is what most devices should measure.
  // The live forms follow as fallbacks the ladder drops to only when the
  // result-scaled ladder is caught folding -- which happens on the vendor
  // compilers (QNN, OpenVINO GPU/NPU) and nowhere else observed.
  //
  // NVFP4 and the float8/float4 QDQ rows have no live fallback: their
  // activations are quantized codes whose dequantize must sit directly
  // against the matmul to fuse, Add has no float8/float4 type constraint,
  // and the one provider that runs them (TensorRT) does not fold, so the
  // timing guard is their only backstop.
  if (v.nvfp4 || (v.qdq && !onnxQdqFusionIsLegal(v.dtype)))
    return {OnnxLiveShape::ResultScaled};

  // int8 QDQ fallbacks: the quantized add (a quantized node unit, which a
  // QDQ-only backend accepts where it refuses a bare int8 Add), then the
  // bare add, then the block's float-scale-and-quantize shape.
  if (v.qdq)
    return {OnnxLiveShape::ResultScaled, OnnxLiveShape::QdqAdd0,
            OnnxLiveShape::Add0, OnnxLiveShape::OperandScaled};

  // Plain floats and the weight-only rows: scale the fp operand as the
  // fallback.
  return {OnnxLiveShape::ResultScaled, OnnxLiveShape::OperandScaled};
}

uint64_t operandBytes(const Variant &v, int64_t D, OnnxLiveShape shape)
{
  const uint64_t elems = (uint64_t)D * (uint64_t)D;
  if (v.nvfp4)
    return 2ull * (onnxElemBytes(ONNX_DT_FLOAT4E2M1, (int64_t)elems) +
                   elems / (uint64_t)v.blockSize);
  if (v.blockSize > 0)
    return elems * 2ull                              // fp16 A
           + onnxElemBytes(v.dtype, (int64_t)elems)  // packed weights
           + elems / (uint64_t)v.blockSize * 2ull;   // fp16 scales
  if (v.qdq)
  {
    // The float-scaled form holds A as fp32 values rather than codes.
    const uint64_t aBytes = (shape == OnnxLiveShape::OperandScaled)
                                ? elems * 4ull
                                : onnxElemBytes(v.dtype, (int64_t)elems);
    return aBytes + onnxElemBytes(v.dtype, (int64_t)elems);
  }
  return 2ull * onnxElemBytes(v.dtype, (int64_t)elems);
}

size_t dtypeSize(int dtype)
{
  switch (dtype)
  {
  case ONNX_DT_FLOAT:
    return 4;
  case ONNX_DT_FLOAT16:
  case ONNX_DT_BFLOAT16:
    return 2;
  default:
    return 1; // int8 / uint8 / float8
  }
}

void fillTensor(std::string &raw, int dtype, int64_t count, uint32_t seed)
{
  uint32_t s = seed;
  // assign, not resize: the sub-byte types write one nibble at a time over
  // whatever is already there, so the buffer has to start at zero.
  raw.assign((size_t)onnxElemBytes(dtype, count), '\0');
  float *f = reinterpret_cast<float *>(&raw[0]);
  uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
  for (int64_t i = 0; i < count; i++)
  {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    float v = (float)(s >> 8) / 16777216.0f - 0.5f; // [-0.5, 0.5)
    switch (dtype)
    {
    case ONNX_DT_FLOAT:
      f[i] = v;
      break;
    case ONNX_DT_FLOAT16:
      h[i] = floatToHalf(v);
      break;
    case ONNX_DT_BFLOAT16:
      h[i] = floatToBf16(v);
      break;
    // Quantized types all store a value already spread over [-1, 1], so the
    // dequantized operands match whatever the format's rounding leaves of
    // them and nothing else differs between the rows.
    default:
      onnxStoreQuantElem(&raw[0], i, dtype, v * 2.0f);
      break;
    }
  }
}

float qdqOutputScale(int64_t K, int outDtype)
{
  // Each dequantized product is a pair of values in [-1, 1], so a K-deep dot
  // product has standard deviation sqrt(K)/3; four sigma keeps nearly every
  // output inside the widest code without compressing the useful range into
  // a handful of codes.  For the 8-bit types that code is 127 -- float8
  // reaches further but its precision is scale-invariant, so the choice does
  // not matter and the established figures stay comparable.  Float4 tops out
  // at 6, and using 127 there would saturate almost every value it was
  // handed.
  const double top = (outDtype == ONNX_DT_FLOAT4E2M1) ? 6.0 : 127.0;
  return (float)(4.0 * std::sqrt((double)K) / 3.0 / top);
}

void destroySetup(const OrtRuntime &rt, GemmSetup &g)
{
  if (g.inVal)
    rt.api->ReleaseValue(g.inVal);
  if (g.zaVal)
    rt.api->ReleaseValue(g.zaVal);
  if (g.outVal)
    rt.api->ReleaseValue(g.outVal);
  if (g.session)
    rt.api->ReleaseSession(g.session);
  g.inVal = nullptr;
  g.zaVal = nullptr;
  g.outVal = nullptr;
  g.session = nullptr;
  g.inBuf.clear();
  g.inBuf.shrink_to_fit();
  g.zaBuf.clear();
  g.zaBuf.shrink_to_fit();
  g.outBuf.clear();
  g.outBuf.shrink_to_fit();
  // `error` is deliberately left intact: callers tear a failed setup down
  // and then report its message.
}

// Create the session and bind the runtime scalar(s) and the reduced output.
// Shared by every model shape here: they differ in what they compute and
// agree entirely on how they are driven.  `sDtype` is the scalar's element
// type, `ioDtype` the output's, and `zaDtype` is nonzero only for the Add
// forms, whose activations take a stored-zero second input.
static void finishSetup(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                        GemmSetup &g, const std::string &modelBytes,
                        int sDtype, int ioDtype, int64_t D, bool profile,
                        bool keepQdqUnfused, int zaDtype)
{
  // Every model here holds its operands as constants and needs ORT's own
  // folding held off: the result-scaled form is otherwise evaluated once at
  // load time, and the live forms still carry a constant weight dequantize
  // that would be baked into full-width weights.
  auto ses = onnxCreateSession(rt, ep, modelBytes,
                               /*keepConstantsUnfolded=*/true, profile,
                               keepQdqUnfused);
  if (!ses.session)
  {
    g.error = ses.error;
    return;
  }
  g.session = ses.session;

  {
    // Exactly one, in the scalar's own width.  Where S scales the operand
    // this keeps every value the multiply sees identical to the constant
    // form's; where it scales the result it exists only so the graph
    // depends on something supplied at run time.  A compiler cannot know
    // the value, so it cannot fold on it.
    std::string one = onnxFloatScalar(1.0f, sDtype);
    g.inBuf.assign(one.begin(), one.end());
  }
  g.outBuf.assign((size_t)D * dtypeSize(ioDtype), 0);

  OrtMemoryInfo *mi = nullptr;
  OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault, &mi);
  if (st)
  {
    g.error = onnxStatusText(rt, st);
    destroySetup(rt, g);
    return;
  }

  if (zaDtype)
  {
    // Literal zero bytes: adding them is bit-identical in every dtype here
    // (0 for int, +0.0 for float), so the values -- and the fusion pattern
    // -- are unchanged, and only the load-time folding becomes impossible.
    g.zaBuf.assign((size_t)onnxElemBytes(zaDtype, 1), 0);
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, g.zaBuf.data(), g.zaBuf.size(), nullptr, 0,
        (ONNXTensorElementDataType)zaDtype, &g.zaVal);
  }
  const int64_t outShape[1] = {D};
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, g.inBuf.data(), g.inBuf.size(), nullptr, 0,
        (ONNXTensorElementDataType)sDtype, &g.inVal);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, g.outBuf.data(), g.outBuf.size(), outShape, 1,
        (ONNXTensorElementDataType)ioDtype, &g.outVal);
  rt.api->ReleaseMemoryInfo(mi);
  if (st)
  {
    g.error = onnxStatusText(rt, st);
    destroySetup(rt, g);
  }
}

GemmSetup makeSetup(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                    const Variant &v, int64_t D, bool profile,
                    int actDtype, bool reduceInFloat, int wgtDtype,
                    OnnxLiveShape shape)
{
  GemmSetup g;
  std::string modelBytes;

  if (v.nvfp4)
  {
    std::string aPacked, aScales, bPacked, bScales;
    onnxFillNvfp4(aPacked, aScales, D, D, /*blockAxis=*/1, v.blockSize,
                  kNvfp4GlobalScale, 0x9e3779b9u);
    onnxFillNvfp4(bPacked, bScales, D, D, /*blockAxis=*/0, v.blockSize,
                  kNvfp4GlobalScale, 0x243f6a88u);
    modelBytes = onnxResidentNvfp4MatMulModel(D, D, D, v.blockSize, aPacked,
                                              aScales, bPacked, bScales,
                                              kNvfp4GlobalScale);
    // Free the raw operands before the session copies the model.
    std::string().swap(aPacked);
    std::string().swap(aScales);
    std::string().swap(bPacked);
    std::string().swap(bScales);
    finishSetup(rt, ep, g, modelBytes, ONNX_DT_FLOAT, ONNX_DT_FLOAT, D,
                profile, /*keepQdqUnfused=*/true, /*zaDtype=*/0);
    return g;
  }

  if (v.blockSize > 0)
  {
    // Weight-only: fp16 activations, blocked-quantized weights, no quantized
    // tensor anywhere near the graph boundary.
    std::string aRaw, wPacked, wScales;
    fillTensor(aRaw, ONNX_DT_FLOAT16, D * D, 0x9e3779b9u);
    onnxFillBlockedWeights(wPacked, wScales, D, D, v.blockSize, 0x243f6a88u,
                           v.dtype);
    modelBytes = onnxResidentWeightOnlyMatMulModel(D, D, D, v.dtype,
                                                   v.blockSize, aRaw, wPacked,
                                                   wScales, shape);
    std::string().swap(aRaw);
    std::string().swap(wPacked);
    std::string().swap(wScales);
    finishSetup(rt, ep, g, modelBytes, ONNX_DT_FLOAT16, ONNX_DT_FLOAT16, D,
                profile, /*keepQdqUnfused=*/false, /*zaDtype=*/0);
    return g;
  }

  std::string aRaw, bRaw;
  const int wDtype = v.qdq ? wgtDtype : v.dtype;
  fillTensor(aRaw, v.qdq ? actDtype : v.dtype, D * D, 0x9e3779b9u);
  fillTensor(bRaw, wDtype, D * D, 0x243f6a88u);

  if (v.qdq)
  {
    modelBytes = onnxResidentQdqMatMulModel(
        D, D, D, aRaw, bRaw, onnxQuantScaleFor(actDtype),
        onnxQuantScaleFor(wDtype), qdqOutputScale(D, actDtype),
        actDtype, wDtype, shape);
  }
  else
  {
    modelBytes = onnxResidentMatMulModel(D, D, D, v.dtype, aRaw, bRaw, shape,
                                         reduceInFloat);
  }
  std::string().swap(aRaw);
  std::string().swap(bRaw);

  // Anything QLinearMatMul cannot carry must not be fused into it.
  const bool unfusable = v.qdq && (!onnxQdqFusionIsLegal(actDtype) ||
                                   !onnxQdqFusionIsLegal(wDtype));
  // The QDQ graph reduces in float, and so does a plain one whose reduction
  // had to be cast; otherwise the tail keeps the matmul's dtype.  The scalar
  // follows the operand it scales: the matmul's own dtype when it scales
  // the operand, the tail's when it scales the result.
  const int ioDtype = (v.qdq || reduceInFloat) ? ONNX_DT_FLOAT : v.dtype;
  const int sDtype = v.qdq ? ONNX_DT_FLOAT
                     : (shape == OnnxLiveShape::OperandScaled) ? v.dtype
                                                                : ioDtype;
  // Second input exists exactly when the QDQ model builds an Add form.
  const bool addForm = v.qdq && (shape == OnnxLiveShape::Add0 ||
                                 shape == OnnxLiveShape::QdqAdd0);
  finishSetup(rt, ep, g, modelBytes, sDtype, ioDtype, D, profile,
              /*keepQdqUnfused=*/unfusable, addForm ? actDtype : 0);
  return g;
}

double timeRuns(const OrtRuntime &rt, GemmSetup &g, unsigned int n)
{
  static const char *inNames[] = {"S", "ZA"};
  static const char *outNames[] = {"Y"};

  auto t0 = std::chrono::steady_clock::now();
  for (unsigned int i = 0; i < n; i++)
  {
    const OrtValue *ins[] = {g.inVal, g.zaVal};
    OrtStatus *st = rt.api->Run(g.session, nullptr,
                                inNames, ins, g.zaVal ? 2 : 1,
                                outNames, 1, &g.outVal);
    if (st)
    {
      g.error = onnxStatusText(rt, st);
      return -1.0;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::micro>(t1 - t0).count() / n;
}

} // namespace onnxgemm

#endif // ENABLE_ONNX
