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
// onnx_gemm refuses to publish a rate without.  Both tests now gate on the
// same fusion check and try the same signed→unsigned schemes, so their
// supported sets stay symmetric.  They also stay in step on folding: when
// gemm's resident ladder proves the provider folded the operands, the rate is
// refused and this row is suppressed with it (see onnxGemmFolded), even
// though this test's non-resident graph would still run.
//
// The fp32 row does a second job.  clpeak's CPU-fallback guard works at
// ORT's partitioning level, so it cannot see an EP that accepts a node and
// then quietly computes it at lower precision internally -- Core ML running
// an fp32 graph on the fp16 Neural Engine is the standard example.  An fp32
// row reading far above the low single digits of ppm is that downgrade
// showing up as a measurement instead of a footnote.
//
// Every measurement here is also the answer check the rate rows ask before
// they publish (OnnxPeak::answerCheck, include/common/answer_check.h): one
// per gemm row -- nvfp4's with no accuracy row of its own -- and one per
// convolution precision, the product written as a 1x1 convolution.  A wrong
// answer withholds the rate rows and files this row as an Error carrying its
// figure.

#include <onnx/onnx_peak.h>
#include "gemm_setup.h"
#include "onnx_model.h"
#include "onnx_probe.h"
#include "onnx_session.h"

#include <cmath>
#include <cstdint>
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

// How this backend's answer checks name things (include/common/answer_check.h).
const clpeak::AnswerWords kWords = {"this provider's", "onnx_numeric_error", "the fp32 reference",
                                    "the fp32 reference"};

// What the accuracy row of a wrong answer says became of its rates.
constexpr const char *kRowConsequence = "so its rate rows are withheld";

// How a format's graph quantizes, which decides how it is built and checked.
enum class Kind
{
  Plain,      // the datatype throughout
  Qdq,        // quantized in and out, the answer kept in the format
  WeightOnly, // fp16 activations against blocked weights
  Nvfp4,      // both operands NVFP4, the answer fp32
};

struct Variant
{
  const char *label;
  int         dtype;
  Kind        kind;
  bool        conv1x1; // the product spelled as a 1x1 convolution
  // The gemm probe entry that says whether the provider runs the format:
  // the gemm row's own, or the convolution checks' precision.
  const char *probe;
  // The row's description; null for a check with no row of its own.
  const char *note;
};

const Variant kVariants[] = {
  {"fp32", ONNX_DT_FLOAT, Kind::Plain, false, "fp32",
   "fp32 against fp32, so near zero; well above single digits means the "
   "provider quietly computes it narrower."},
  {"fp16", ONNX_DT_FLOAT16, Kind::Plain, false, "fp16",
   "fp16 operands and answer.  A provider that also keeps the running total "
   "at 16 bits adds error of its own."},
  {"bf16", ONNX_DT_BFLOAT16, Kind::Plain, false, "bf16",
   "bfloat16 operands and answer: three fewer mantissa bits than fp16, so a "
   "larger error on in-range values like these."},
  {"fp8_e4m3", ONNX_DT_FLOAT8E4M3FN, Kind::Qdq, false, "fp8_e4m3",
   "fp8 e4m3, quantized in and out.  Uniform values like these favour int8, "
   "since they never use the range float8 spends its bits on."},
  {"fp8_e5m2", ONNX_DT_FLOAT8E5M2, Kind::Qdq, false, "fp8_e5m2",
   "fp8 e5m2, quantized in and out: one fewer mantissa bit than e4m3, so "
   "about twice its error."},
  {"fp4_e2m1", ONNX_DT_FLOAT4E2M1, Kind::Qdq, false, "fp4_e2m1",
   "fp4 e2m1, quantized in and out: two fewer mantissa bits than e4m3, so "
   "roughly four times its error."},
  // No row: keeping NVFP4's answer in four bits needs a blocked
  // QuantizeLinear, which segfaults TensorRT (src/onnx/AGENTS.md), and
  // without it the figure would read as though four bits cost nothing.  The
  // check reads the product whole, which is all a check needs.
  {"nvfp4", ONNX_DT_FLOAT4E2M1, Kind::Nvfp4, false, "nvfp4", nullptr},
  // Weight-only: the reference takes the same codes at their exact scaled
  // values, so what is left is the kernel's cost, which no rate row shows.
  {"fp4_weight", ONNX_DT_FLOAT4E2M1, Kind::WeightOnly, false, "fp4_weight",
   "4-bit float weights in blocks of 32 against 16-bit activations.  Near "
   "the fp16 row means a 16-bit multiply; far above it, activations "
   "quantized to integers."},
  {"int4_weight", ONNX_DT_INT4, Kind::WeightOnly, false, "int4_weight",
   "4-bit integer weights in blocks of 32 against 16-bit activations.  Near "
   "the fp16 row means a 16-bit multiply; far above it, activations "
   "quantized to integers."},
  {"int8_weight", ONNX_DT_INT8, Kind::WeightOnly, false, "int8_weight",
   "8-bit integer weights in blocks of 32 against 16-bit activations.  Near "
   "the fp16 row means a 16-bit multiply; far above it, activations "
   "quantized to integers."},
  {"int8_qdq", ONNX_DT_INT8, Kind::Qdq, false, "int8_qdq",
   "Full-integer int8: 8-bit activations and weights, the answer kept in 8 "
   "bits."},
  {"int8_qdq_conv1x1", ONNX_DT_INT8, Kind::Qdq, true, "int8_qdq_conv1x1",
   "The int8 row's product as a quantized 1x1 convolution, the answer kept "
   "in 8 bits."},
  // The convolution rows' checks (conv.cpp): each precision's product as a
  // 1x1 convolution, the conv1x1 row's operator and the nearest one to the
  // other shapes'.
  {"fp32_conv1x1", ONNX_DT_FLOAT, Kind::Plain, true, "fp32", nullptr},
  {"fp16_conv1x1", ONNX_DT_FLOAT16, Kind::Plain, true, "fp16", nullptr},
};

const Variant *findVariant(const std::string &label)
{
  for (const Variant &v : kVariants)
    if (label == v.label)
      return &v;
  return nullptr;
}

// What a rate row's reason calls the format a check stands for.
std::string whatOf(const Variant &v)
{
  if (!v.note && v.conv1x1)
    return std::string(v.probe) + " written as a 1x1 convolution";
  return v.label;
}

// The gemm row's block size (gemm_setup.cpp): a check quantizes as the rate
// row it stands for does.
int64_t gemmBlockSize(const char *label)
{
  for (size_t i = 0; i < onnxgemm::kFpVariantCount; i++)
    if (std::strcmp(onnxgemm::kFpVariants[i].label, label) == 0)
      return onnxgemm::kFpVariants[i].blockSize;
  return 0;
}

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

// The values a blocked weight-only matrix multiplies, [K, N]: each stored
// code times its block's fp16 scale, exactly -- what a fused kernel, which
// scales after it multiplies, computes with.  Core ML's and LiteRT's
// references take the same exact products.
std::vector<float> dequantizeBlocked(const std::string &packed, const std::string &scales, int64_t K,
                                     int64_t N, int64_t block, int wDtype)
{
  std::vector<float> out((size_t)(K * N));
  const uint16_t *sc = reinterpret_cast<const uint16_t *>(scales.data());
  for (int64_t k = 0; k < K; k++)
    for (int64_t n = 0; n < N; n++)
    {
      const int64_t i = k * N + n;
      float code;
      switch (wDtype)
      {
      case ONNX_DT_INT4:       code = (float)onnxLoadInt4(packed.data(), i); break;
      case ONNX_DT_FLOAT4E2M1: code = fp4E2M1ToFloat(onnxLoadNibble(packed.data(), i)); break;
      default:                 code = (float)(int8_t)packed[(size_t)i]; break;
      }
      out[(size_t)i] = code * halfToFloat(sc[(k / block) * N + n]);
    }
  return out;
}

// The values an NVFP4 matrix multiplies (onnxFillNvfp4): each code times its
// block's E4M3 scale times the global one, all exact in fp32.
std::vector<float> dequantizeNvfp4(const std::string &packed, const std::string &blockScales,
                                   int64_t rows, int64_t cols, int blockAxis, int64_t block,
                                   float globalScale)
{
  std::vector<float> out((size_t)(rows * cols));
  const int64_t sCols = (blockAxis == 0) ? cols : cols / block;
  const uint8_t *bs = reinterpret_cast<const uint8_t *>(blockScales.data());
  for (int64_t i = 0; i < rows; i++)
    for (int64_t j = 0; j < cols; j++)
    {
      const int64_t s = (blockAxis == 0) ? (i / block) * sCols + j : i * sCols + j / block;
      out[(size_t)(i * cols + j)] =
          fp4E2M1ToFloat(onnxLoadNibble(packed.data(), i * cols + j)) * fp8E4M3ToFloat(bs[s]) * globalScale;
    }
  return out;
}

// One graph input or output: its name, element type and shape.
struct Io
{
  const char *name;
  int dtype;
  OnnxDims dims;
};

int64_t elemCount(const OnnxDims &dims)
{
  int64_t n = 1;
  for (int64_t d : dims)
    n *= d;
  return n;
}

// Run one already-built model on one EP, returning the raw output bytes.
// `error` is set (and the vector left empty) on any failure.  When `ops` is
// given the session is profiled and the kernels the provider actually ran are
// written to it.  `keepQdqUnfused` and `keepConstantsUnfolded` are
// onnxCreateSession's.
std::vector<uint8_t> runOnce(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                             const std::string &modelBytes,
                             const Io &in, const void *inData, size_t inBytes,
                             const Io &outIo, std::string &error,
                             std::vector<std::string> *ops = nullptr,
                             bool keepQdqUnfused = false,
                             bool keepConstantsUnfolded = false)
{
  std::vector<uint8_t> out;

  auto ses = onnxCreateSession(rt, ep, modelBytes, keepConstantsUnfolded,
                               /*profile=*/ops != nullptr, keepQdqUnfused);
  if (!ses.session)
  {
    error = ses.error;
    return out;
  }

  out.resize((size_t)elemCount(outIo.dims) * dtypeSize(outIo.dtype));

  OrtMemoryInfo *mi = nullptr;
  OrtValue *inVal = nullptr, *outVal = nullptr;
  OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault, &mi);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, const_cast<void *>(inData), inBytes,
        in.dims.empty() ? nullptr : in.dims.data(), in.dims.size(),
        (ONNXTensorElementDataType)in.dtype, &inVal);
  if (!st)
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, out.data(), out.size(), outIo.dims.data(), outIo.dims.size(),
        (ONNXTensorElementDataType)outIo.dtype, &outVal);
  if (!st)
  {
    const char *ins[]  = {in.name};
    const char *outs[] = {outIo.name};
    st = rt.api->Run(ses.session, nullptr, ins,
                     (const OrtValue *const *)&inVal, 1, outs, 1, &outVal);
  }
  if (st)
  {
    error = onnxStatusText(rt, st);
    out.clear();
  }
  // End profiling even after a failed Run: otherwise ORT leaves its profile
  // file behind when the benchmark turns that failure into an unsupported row.
  if (ops)
    *ops = onnxCollectExecutedOps(rt, ses.session);

  if (inVal)  rt.api->ReleaseValue(inVal);
  if (outVal) rt.api->ReleaseValue(outVal);
  if (mi)     rt.api->ReleaseMemoryInfo(mi);
  rt.api->ReleaseSession(ses.session);
  return out;
}

// The reference: aRef[M, K] times bRef[K, N] in fp32 on the CPU EP -- the one
// path present on every machine and the one whose arithmetic is not in
// question.
bool referenceProduct(const OrtRuntime &rt, const std::vector<float> &aRef,
                      const std::vector<float> &bRef, std::vector<float> &ref,
                      std::string &error)
{
  onnx_ep_info_t cpuEp;
  cpuEp.providerKey = "CPUExecutionProvider";
  cpuEp.displayName = "ONNX Runtime CPU";
  cpuEp.deviceType  = DeviceType::Cpu;

  std::string refWeights(reinterpret_cast<const char *>(bRef.data()),
                         bRef.size() * sizeof(float));
  std::string refModel = onnxMatMulModel(kDim, kDim, kDim, ONNX_DT_FLOAT,
                                         refWeights);
  auto refRaw = runOnce(rt, cpuEp, refModel,
                        Io{"A", ONNX_DT_FLOAT, {kDim, kDim}}, aRef.data(),
                        aRef.size() * sizeof(float),
                        Io{"C", ONNX_DT_FLOAT, {kDim, kDim}}, error);
  if (refRaw.empty())
    return false;
  ref.resize((size_t)kDim * kDim);
  std::memcpy(ref.data(), refRaw.data(), refRaw.size());
  return true;
}

// Every kernel a profiled run executed, in order, repeats included.
std::string joinRan(const std::vector<std::string> &ops)
{
  std::string joined;
  for (const auto &op : ops)
    joined += (joined.empty() ? "" : ", ") + op;
  return joined;
}

// The reason a quantized run that did not fuse gives, for the row.
std::string unfusedReason(const Variant &v, const std::string &tried)
{
  return "provider did not fuse a quantized matmul -- it dequantized the " +
         std::string(v.kind == Kind::WeightOnly ? "weights" : "operands") +
         " and multiplied in floating point, so this is not a " +
         std::string(v.label) + " error (ran: " + tried + ")";
}

// Measure one format's answer on this provider into `c`: its figure against
// the fp32 reference, or the status and reason there is none.
void measureAnswer(const OrtRuntime &rt, const onnx_ep_info_t &ep, const Variant &v,
                   OnnxPeak::AnswerCheck &c)
{
  // Global probe fast-path: gemm's 64^3 already knows if this dtype is
  // emulated/slow on this EP (QNN 33s).  Skip before paying 1024^3.
  {
    const auto &probe = onnxProbeGemmCache(rt, ep);
    auto it = probe.find(v.probe);
    if (it != probe.end() && !it->second.ok)
    {
      c.status = onnxFailureStatus(it->second.reason);
      c.error = it->second.reason;
      return;
    }
  }
  // The same gate onnx-gemm applies.  Without it a datatype newer than the
  // runtime reports whatever ORT says about IR versions, which names neither
  // the datatype nor the fix.
  if (std::string why = onnxDtypeUnsupportedReason(rt, v.dtype); !why.empty())
  {
    c.status = ResultStatus::Unsupported;
    c.error = why;
    return;
  }
  // And the fence, for the same reason: this graph is the probe's in
  // another shape, and a provider that crashes on it must not be handed
  // it whatever the cache says.
  const int64_t blockSize = (v.kind == Kind::WeightOnly || v.kind == Kind::Nvfp4)
                                ? gemmBlockSize(v.label) : 0;
  if (std::string why = onnxProviderFenceReason(ep, v.dtype, v.kind == Kind::Qdq, blockSize);
      !why.empty())
  {
    c.status = ResultStatus::Unsupported;
    c.error = why;
    return;
  }

  const float cScale = qdqOutputScale(kDim, v.dtype);
  // A 1x1 convolution's activations are [1, K, H, W] and its answer
  // [1, N, H, W]: the same values as the matmul's, channel-major.
  const OnnxDims actDims = v.conv1x1 ? OnnxDims{1, kDim, 32, 32} : OnnxDims{kDim, kDim};

  // Quantization schemes, tried in order until one fuses - aligned with
  // gemm.cpp.  There is no single choice that works everywhere: TensorRT
  // rejects unsigned activations and demands a zero point of zero, while
  // x86 MLAS without VNNI implements only the unsigned form and quietly
  // declines to fuse the signed one.  Trying is the only way to know, and
  // the fusion check is what decides.  Both tests now gate on the same
  // decision so their supported sets stay symmetric.
  struct QuantScheme
  {
    int actDtype;
    int wDtype;
    const char *name;
  };
  static const QuantScheme kInt8Schemes[] = {
      {ONNX_DT_INT8, ONNX_DT_INT8, "signed activations"},
      {ONNX_DT_UINT8, ONNX_DT_INT8, "unsigned activations"},
  };

  std::string err;
  std::vector<uint8_t> raw;
  std::vector<float> got, aRef, bRef;
  const char *what = v.label;

  if (v.kind == Kind::Qdq)
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
    QuantScheme single[1] = {{v.dtype, v.dtype, "matching activations and weights"}};
    const QuantScheme *schemes = (v.dtype == ONNX_DT_INT8) ? kInt8Schemes : single;
    const size_t nSchemes = (v.dtype == ONNX_DT_INT8) ? 2u : 1u;
    std::string firstErr;
    std::string tried; // what an unfused attempt actually ran
    std::string aRaw, bRaw;
    std::vector<float> aDeq;
    bool isFused = false;
    int aDtype = v.dtype;
    for (size_t si = 0; si < nSchemes; si++)
    {
      const QuantScheme &qs = schemes[si];
      aDtype = qs.actDtype;
      fillTensor(aRaw, qs.actDtype, kDim * kDim, 0x9e3779b9u);
      fillTensor(bRaw, qs.wDtype, kDim * kDim, 0x243f6a88u);
      std::string model = onnxQdqMatMulModel(
          kDim, kDim, kDim, bRaw, onnxQuantScaleFor(qs.actDtype),
          onnxQuantScaleFor(qs.wDtype), cScale, qs.actDtype, qs.wDtype,
          /*floatIo=*/true, v.conv1x1);
      err.clear();
      // Anything QLinearMatMul cannot carry must not be fused into it.
      const bool unfusable = !onnxQdqFusionIsLegal(qs.actDtype) ||
                             !onnxQdqFusionIsLegal(qs.wDtype);
      // What the device is given is the dequantized form of what was
      // quantized here, so the reference below and the run see one set of
      // values.
      aDeq = widen(aRaw, qs.actDtype, kDim * kDim, onnxQuantScaleFor(qs.actDtype));
      std::vector<std::string> ops;
      raw = runOnce(rt, ep, model, Io{"A", ONNX_DT_FLOAT, actDims}, aDeq.data(),
                    aDeq.size() * sizeof(float), Io{"C", ONNX_DT_FLOAT, actDims},
                    err, &ops, unfusable);
      if (raw.empty())
      {
        if (firstErr.empty())
          firstErr = err;
        CLPEAK_VLOG("onnx-numeric-error[%s/%s]: %s rejected: %s\n",
                    ep.providerKey.c_str(), what, qs.name, err.c_str());
        continue;
      }
      // Fusion check is the selector, exactly as in gemm.cpp.
      if (onnxOpsRanQuantizedMatMul(ops))
      {
        isFused = true;
        break;
      }
      const std::string joined = joinRan(ops);
      if (tried.empty() && !joined.empty())
        tried = joined;
      CLPEAK_VLOG("onnx-numeric-error[%s/%s]: %s executed %s (no fused quantized matmul)\n",
                  ep.providerKey.c_str(), what, qs.name, joined.c_str());
      // Symmetry requires a fused kernel, so an unfused success is not kept.
      raw.clear();
      if (firstErr.empty())
        firstErr = err;
    }
    if (!isFused)
    {
      // Aligned with gemm.cpp: a quantized row is only published when the
      // provider fused a quantized matmul. Reporting the dequantized float
      // error as a quantized error would be symmetric with gemm's suppression
      // but would attribute the quantization scheme's cost to the hardware.
      const std::string why = !tried.empty() ? unfusedReason(v, tried) : firstErr;
      c.status = onnxFailureStatus(why);
      c.error = why.empty() ? "run failed" : why;
      return;
    }
    if (v.dtype == ONNX_DT_INT8)
      // Which scheme fused, for the row description.
      c.note = (aDtype == ONNX_DT_UINT8) ? "  Measured with unsigned activations."
                                         : "  Measured with signed activations.";
    // The quantized path returns fp32 already dequantized.
    got.resize((size_t)kDim * kDim);
    std::memcpy(got.data(), raw.data(), got.size() * sizeof(float));
    aRef = aDeq;
    bRef = widen(bRaw, v.dtype, kDim * kDim, onnxQuantScaleFor(v.dtype));
  }
  else if (v.kind == Kind::WeightOnly || v.kind == Kind::Nvfp4)
  {
    // The rate row's own operands and session options (gemm_setup.cpp):
    // weight-only, fp16 activations handed in against the blocked weights;
    // NVFP4, both operands resident as in its single multiply, the whole
    // fp32 product read back -- a check must not quantize anything to get
    // it, since TensorRT dies on a blocked QuantizeLinear.
    std::string model;
    std::string aRaw;
    const float one = 1.0f;
    const void *inData;
    size_t inBytes;
    Io in{"A", ONNX_DT_FLOAT16, {kDim, kDim}}, outIo{"C", ONNX_DT_FLOAT16, {kDim, kDim}};
    if (v.kind == Kind::WeightOnly)
    {
      std::string wPacked, wScales;
      fillTensor(aRaw, ONNX_DT_FLOAT16, kDim * kDim, 0x9e3779b9u);
      onnxFillBlockedWeights(wPacked, wScales, kDim, kDim, blockSize, 0x243f6a88u, v.dtype);
      model = onnxWeightOnlyMatMulModel(kDim, kDim, kDim, v.dtype, blockSize, wPacked, wScales);
      aRef = widen(aRaw, ONNX_DT_FLOAT16, kDim * kDim, 1.0f);
      bRef = dequantizeBlocked(wPacked, wScales, kDim, kDim, blockSize, v.dtype);
      inData = aRaw.data();
      inBytes = aRaw.size();
    }
    else
    {
      std::string aPacked, aScales, bPacked, bScales;
      onnxFillNvfp4(aPacked, aScales, kDim, kDim, /*blockAxis=*/1, blockSize,
                    onnxgemm::kNvfp4GlobalScale, 0x9e3779b9u);
      onnxFillNvfp4(bPacked, bScales, kDim, kDim, /*blockAxis=*/0, blockSize,
                    onnxgemm::kNvfp4GlobalScale, 0x243f6a88u);
      model = onnxResidentNvfp4MatMulModel(kDim, kDim, kDim, blockSize, aPacked, aScales,
                                           bPacked, bScales, onnxgemm::kNvfp4GlobalScale,
                                           OnnxReduceView::Rows, /*wholeProduct=*/true);
      aRef = dequantizeNvfp4(aPacked, aScales, kDim, kDim, 1, blockSize, onnxgemm::kNvfp4GlobalScale);
      bRef = dequantizeNvfp4(bPacked, bScales, kDim, kDim, 0, blockSize, onnxgemm::kNvfp4GlobalScale);
      in = Io{"S", ONNX_DT_FLOAT, {}};
      outIo = Io{"Y", ONNX_DT_FLOAT, {kDim, kDim}};
      inData = &one;
      inBytes = sizeof one;
    }
    std::vector<std::string> ops;
    raw = runOnce(rt, ep, model, in, inData, inBytes, outIo, err, &ops,
                  /*keepQdqUnfused=*/v.kind == Kind::Nvfp4, /*keepConstantsUnfolded=*/true);
    if (raw.empty())
    {
      c.status = onnxFailureStatus(err);
      c.error = err.empty() ? "run failed" : err;
      return;
    }
    // The fusion check the gemm probe applies to these rows.
    if (!onnxOpsRanQuantizedMatMul(ops))
    {
      CLPEAK_VLOG("onnx-numeric-error[%s/%s]: executed %s (no fused quantized matmul)\n",
                  ep.providerKey.c_str(), what, joinRan(ops).c_str());
      c.status = ResultStatus::Unsupported;
      c.error = unfusedReason(v, joinRan(ops));
      return;
    }
    got = widen(std::string(reinterpret_cast<const char *>(raw.data()), raw.size()),
                outIo.dtype, kDim * kDim, 1.0f);
  }
  else
  {
    std::string aRaw, bRaw;
    fillTensor(aRaw, v.dtype, kDim * kDim, 0x9e3779b9u);
    fillTensor(bRaw, v.dtype, kDim * kDim, 0x243f6a88u);
    std::string model = onnxMatMulModel(kDim, kDim, kDim, v.dtype, bRaw, v.conv1x1);
    raw = runOnce(rt, ep, model, Io{"A", v.dtype, actDims}, aRaw.data(), aRaw.size(),
                  Io{"C", v.dtype, actDims}, err);
    if (raw.empty())
    {
      c.status = onnxFailureStatus(err);
      c.error = err.empty() ? "run failed" : err;
      return;
    }
    // The provider's answer, widened to fp32, and the reference's operands:
    // the same values the provider saw, widened to fp32.
    got = widen(std::string(reinterpret_cast<const char *>(raw.data()), raw.size()),
                v.dtype, kDim * kDim, cScale);
    aRef = widen(aRaw, v.dtype, kDim * kDim, onnxQuantScaleFor(v.dtype));
    bRef = widen(bRaw, v.dtype, kDim * kDim, onnxQuantScaleFor(v.dtype));
  }

  if (got.empty() || aRef.empty() || bRef.empty())
  {
    c.status = ResultStatus::Error;
    c.error = "this datatype has no widening to fp32 here, so no reference "
              "can be built for it";
    return;
  }

  // A 1x1 convolution multiplies its [N, K] weights by its [K, H * W]
  // activations, so its reference is the product the other way round.
  std::vector<float> ref;
  std::string refErr;
  const bool refOk = v.conv1x1 ? referenceProduct(rt, bRef, aRef, ref, refErr)
                               : referenceProduct(rt, aRef, bRef, ref, refErr);
  if (!refOk)
  {
    c.status = ResultStatus::Error;
    c.error = "fp32 reference failed: " + refErr;
    return;
  }

  // The figure.  The reference multiplies values in [-1, 1] in fp32, so a
  // NaN or an infinity in it is the provider's.
  clpeak::judgeAnswer(c, got.data(), ref.data(), got.size(), kWords);
  if (c.ppm >= 0.0)
    CLPEAK_VLOG("onnx-numeric-error[%s/%s]: %.2f ppm%s\n", ep.providerKey.c_str(), what, c.ppm,
                c.wrong() ? ", a wrong answer" : "");
}

} // namespace

const OnnxPeak::AnswerCheck &OnnxPeak::answerCheck(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                                                   const std::string &label)
{
  // OpenVINO's targets share a providerKey and a plugin's devices a library,
  // so the device each names is part of the provider.
  const std::string key = ep.providerKey + '\x1f' + ep.epDevice + '\x1f' +
                          std::to_string((uintptr_t)(const void *)ep.epDevicePtr) + '\x1f' + label;
  auto found = answerChecks_.find(key);
  if (found != answerChecks_.end())
    return found->second;
  AnswerCheck &c = answerChecks_[key];
  if (const Variant *v = findVariant(label))
    measureAnswer(rt, ep, *v, c);
  else
  {
    c.status = ResultStatus::Error;
    c.error = "no answer check for " + label;
  }
  return c;
}

std::string OnnxPeak::wrongAnswer(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                                  const std::string &label)
{
  const AnswerCheck &c = answerCheck(rt, ep, label);
  const Variant *v = findVariant(label);
  return clpeak::wrongAnswerReason(c, kWords, v ? whatOf(*v) : label);
}

int OnnxPeak::runNumericError(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                              benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"onnx_numeric_error", "ONNX MatMul numeric error", "ppm",
       Category::Compute,
       "How far each data type's answer on a 1024-cubed matmul drifts from "
       "an fp32 reference, in parts per million.  The reference multiplies "
       "the exact values this provider was given, so only the arithmetic and "
       "the width the answer was kept in remain.",
       // Lower is better, which the `ppm` unit already says.
       TestShape::Heterogeneous, "data type"});

  for (const Variant &v : kVariants)
  {
    if (!v.note)
      continue;   // a check the rate rows ask, with no row of its own
    if (clpeak::cancelRequested())
      break;

    logger::EmitOptions o;
    o.description = v.note;

    // Paired suppression: gemm's ladder proved this provider folded the
    // resident operands for this label, so its rate was refused as
    // meaningless.  This test's non-resident graph cannot fold and would
    // still produce a number -- but a rate without its accuracy is half a
    // number, and so is an accuracy without its rate.  Suppress rather than
    // publish one half of the pair alone.
    if (onnxGemmFolded(ep, v.label))
    {
      CLPEAK_VLOG("onnx-numeric-error[%s/%s]: suppressed (gemm folded)\n",
                  ep.providerKey.c_str(), v.label);
      test.skip(v.label, ResultStatus::Unsupported,
                "suppressed: onnx-gemm folded the operands for this datatype, "
                "so no rate was published for the pair",
                o.description);
      continue;
    }

    const AnswerCheck &c = answerCheck(rt, ep, v.label);
    o.description += c.note;
    // A wrong answer is no precision figure: the row is an Error carrying
    // it, beside the rate rows it withheld.
    if (c.wrong())
    {
      test.skip(v.label, ResultStatus::Error,
                clpeak::wrongAnswerRowReason(c, kWords, kRowConsequence), o.description);
      continue;
    }
    if (c.ppm < 0.0)
    {
      test.skip(v.label, c.status, c.error, o.description);
      continue;
    }
    test.emit(v.label, (float)c.ppm, o);
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
