#ifdef ENABLE_ONNX

// onnx-gemm: MatMul peak through an ONNX Runtime execution provider.
//
// Both operands are model constants and the result is summed down to a single
// row, so nothing large crosses the host boundary per run.  That shape is
// forced by discrete GPUs: with A as a graph input and C returned to the
// host, an RTX 5060 reported 15 TFLOPS for fp16 while a whole transformer
// block -- whose weights are resident -- reached 28 on the same device.  The
// peak was measuring PCIe.  On unified-memory devices the difference is
// small, but the graph is identical everywhere so the rows stay comparable.
//
// One test, `onnx_gemm`: the same single-operation model on whichever
// formats the provider accepts.  The int8 QDQ reading is measured in ops
// rather than flops and carries that unit itself.
// int8 is the dtype most NPUs are actually built for, so an NPU whose only
// measured reading is the int8 one is the expected shape, not a gap.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

namespace
{

  // The ladder doubles from 1024 until the rate stops improving, and the peak
  // is reported along with the size that produced it.
  //
  // Reporting a peak rather than "the rate at 4096" is what keeps this number
  // comparable over time.  A fixed size has to be raised as hardware grows --
  // today's largest rung will one day be too small to saturate anything -- and
  // the moment it is raised, every result recorded before becomes a different
  // measurement wearing the same name.  An extending search has no such
  // horizon: faster hardware simply climbs further, and "the best this device
  // can do at any size" means the same thing in ten years as it does now.
  //
  // It is also not a size chosen from a timing probe, which was the previous
  // design.  A probe is unstable -- the size comes out of a cube root and is
  // then bucketed, so a couple of percent of timing noise can push the estimate
  // across a bucket edge and change the answer.  On an M1 Pro that made the
  // fp16 row alternate between 5.8 and 6.2 TFLOPS depending on nothing else.
  // And no single size is right anyway: fp32 there peaks at 4096 while fp16
  // peaks at 2048, because different engines serve them.
  constexpr int64_t kMinDim = 1024;

  // NVFP4's second scale.  A power of two so factoring it out of the block scales
  // is exact and the row measures the format rather than an arithmetic accident.
  constexpr float kNvfp4GlobalScale = 0.125f;
  constexpr int64_t kMaxDim = 32768;

  // A size counts as an improvement only if it beats the best so far by this
  // much; two failures in a row end the search.  The grace of one lets a curve
  // dip at a single size and recover, which happens when one size lands badly
  // against a cache but the next tiles better.
  constexpr double kImproveFactor = 1.03;
  constexpr int kMaxStrikes = 2;

  // Ceilings that keep the search from running away on either axis.  The time
  // bound is predicted from the previous size's measured rate, so a slow
  // provider stops early instead of spending minutes on one matrix, and it
  // scales itself: hardware fast enough to make a bigger size cheap is exactly
  // the hardware that should try it.
  constexpr double kMaxIterUs = 2.0e6; // one iteration, predicted

  // Both operands together, capped at a quarter of physical memory.  A fixed
  // ceiling here would be a crash on a phone and a needless limit on a
  // workstation; see clpeak::memoryBudget.
  //
  // And capped again by protobuf.  An ONNX model is a protobuf message, whose
  // serialized size cannot exceed 2 GiB, and both operands live inside it as
  // initializers -- so a size the machine has memory for can still be
  // unbuildable.  fp32 at 16384 needs exactly 2 GiB of operands and ORT answers
  // "Model data size exceeds maximum supported size (2GB)", which is a property
  // of the format rather than of the device and does not belong in a memory
  // budget.  The slack leaves room for the graph around the weights.
  static uint64_t maxWeightBytes()
  {
    const uint64_t protobufCeiling = (2ull << 30) - (64ull << 20);
    return std::min(clpeak::memoryBudget(3ull << 30), protobufCeiling);
  }

  // Per-size budget for the timed phase.  Lower than the 5 s a single-size test
  // would use, since the ladder measures several.
  constexpr unsigned int kSizeBudgetUs = 2000000;

  struct Variant
  {
    int dtype; // element type of the graph's input/output
    bool qdq;  // build the quantized (DequantizeLinear/MatMul/Q) form
    const char *label;
    const char *note;
    int64_t blockSize; // >0: blocked, one scale per this many elements
    bool nvfp4;        // blocked on *both* operands, with a second scale
  };

  // Quantization schemes, tried in order until one fuses.  There is no single
  // choice that works everywhere: TensorRT rejects unsigned activations and
  // demands a zero point of zero, while x86 MLAS without VNNI implements only
  // the unsigned form and quietly declines to fuse the signed one.  Trying is
  // the only way to know, and the fusion check is what decides.
  struct QuantScheme
  {
    int actDtype;
    int wDtype;
    const char *name;
  };

  // int8 has two spellings and no provider takes both.  The float8 formats have
  // one: activations and weights share the type, and there is no signed/unsigned
  // question because they are signed floats.
  const QuantScheme kInt8Schemes[] = {
      {ONNX_DT_INT8, ONNX_DT_INT8, "signed activations"},    // TensorRT, ARM
      {ONNX_DT_UINT8, ONNX_DT_INT8, "unsigned activations"}, // x86 without VNNI
  };

  struct Variant;
  size_t schemesFor(const Variant &v, QuantScheme out[2]);

  struct GemmSetup
  {
    OrtSession *session = nullptr;
    OrtValue *inVal = nullptr;  // the scalar that keeps the graph live
    OrtValue *outVal = nullptr; // reduced row
    std::vector<uint8_t> inBuf, outBuf;
    std::string error;

    // Not copyable, and the compiler has to enforce it: inVal and outVal are
    // OrtValues built over inBuf and outBuf, so a copy leaves them pointing at
    // the original's buffers.  Moving is fine -- a moved vector keeps its
    // allocation -- which is why returning one of these by value works and
    // handing back a reference to one does not.
    GemmSetup() = default;
    GemmSetup(const GemmSetup &) = delete;
    GemmSetup &operator=(const GemmSetup &) = delete;
    GemmSetup(GemmSetup &&) = default;
    GemmSetup &operator=(GemmSetup &&) = default;
  };

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

  // Deterministic values, generated once and reused for inputs and weights.
  // Floats land in [-0.5, 0.5) and int8 in [-127, 127]: small magnitudes keep
  // fp16 accumulation over a 4096-deep dot product far from overflow, and
  // avoid the NaN/denormal slow paths raw random bit patterns would hit.
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

  // Output scale for the QDQ form.  Each dequantized product is a pair of
  // values in [-1, 1], so a K-deep dot product has standard deviation
  // sqrt(K)/3; four sigma keeps nearly every output inside int8 without
  // compressing the useful range into a handful of codes.
  float qdqOutputScale(int64_t K, int outDtype)
  {
    // Four sigma of a K-deep dot product mapped onto the widest code the output
    // type has.  For the 8-bit types that is 127 -- float8 reaches further but
    // its precision is scale-invariant, so the choice does not matter and the
    // established figures stay comparable.  Float4 tops out at 6, and using 127
    // there would saturate almost every value it was handed.
    const double top = (outDtype == ONNX_DT_FLOAT4E2M1) ? 6.0 : 127.0;
    return (float)(4.0 * std::sqrt((double)K) / 3.0 / top);
  }

  void destroySetup(const OrtRuntime &rt, GemmSetup &g)
  {
    if (g.inVal)
      rt.api->ReleaseValue(g.inVal);
    if (g.outVal)
      rt.api->ReleaseValue(g.outVal);
    if (g.session)
      rt.api->ReleaseSession(g.session);
    g.inVal = nullptr;
    g.outVal = nullptr;
    g.session = nullptr;
    g.inBuf.clear();
    g.inBuf.shrink_to_fit();
    g.outBuf.clear();
    g.outBuf.shrink_to_fit();
    // `error` is deliberately left intact: callers tear a failed setup down
    // and then report its message.
  }

  // Build model + session + bound input/output tensors for one (variant, D).
  // Create the session and bind the scalar input and reduced output.  Shared by
  // every model shape here: they differ in what they compute and agree entirely
  // on how they are driven.
  void finishSetup(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                   GemmSetup &g, const std::string &modelBytes,
                   int ioDtype, int64_t D, bool profile,
                   bool keepQdqUnfused)
  {
    // Every model here holds its operands as constants and needs folding held
    // off, or the whole multiply is evaluated once at load time.
    auto ses = onnxCreateSession(rt, ep, modelBytes,
                                 /*keepConstantsUnfolded=*/true, profile,
                                 keepQdqUnfused);
    if (!ses.session)
    {
      g.error = ses.error;
      return;
    }
    g.session = ses.session;

    const size_t es = dtypeSize(ioDtype);
    {
      // Scales the reduced result; exists only so the graph depends on
      // something supplied at run time.
      std::string one;
      fillTensor(one, ioDtype, 1, 0x12345678u);
      g.inBuf.assign(one.begin(), one.end());
    }
    g.outBuf.assign((size_t)D * es, 0);

    OrtMemoryInfo *mi = nullptr;
    OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                                OrtMemTypeDefault, &mi);
    if (st)
    {
      g.error = onnxStatusText(rt, st);
      destroySetup(rt, g);
      return;
    }

    const int64_t outShape[1] = {D};
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, g.inBuf.data(), g.inBuf.size(), nullptr, 0,
        (ONNXTensorElementDataType)ioDtype, &g.inVal);
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
                      const Variant &v, int64_t D, bool profile = false,
                      int actDtype = ONNX_DT_UINT8,
                      bool reduceInFloat = false,
                      int wgtDtype = ONNX_DT_INT8)
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
      aPacked.clear();
      aPacked.shrink_to_fit();
      aScales.clear();
      aScales.shrink_to_fit();
      bPacked.clear();
      bPacked.shrink_to_fit();
      bScales.clear();
      bScales.shrink_to_fit();
      finishSetup(rt, ep, g, modelBytes, ONNX_DT_FLOAT, D, profile,
                  /*keepQdqUnfused=*/true);
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
                                                     wScales);
      aRaw.clear();
      aRaw.shrink_to_fit();
      wPacked.clear();
      wPacked.shrink_to_fit();
      wScales.clear();
      wScales.shrink_to_fit();
      finishSetup(rt, ep, g, modelBytes, ONNX_DT_FLOAT16, D, profile,
                  /*keepQdqUnfused=*/false);
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
          actDtype, wDtype);
    }
    else
    {
      modelBytes = onnxResidentMatMulModel(D, D, D, v.dtype, aRaw, bRaw,
                                           reduceInFloat);
    }
    aRaw.clear();
    aRaw.shrink_to_fit();
    bRaw.clear();
    bRaw.shrink_to_fit();

    // Anything QLinearMatMul cannot carry must not be fused into it.
    const bool unfusable = !onnxQdqFusionIsLegal(actDtype) ||
                           !onnxQdqFusionIsLegal(wDtype);
    // The QDQ graph reduces in float, and so does a plain one whose reduction
    // had to be cast; otherwise the tail keeps the matmul's dtype.
    const int ioDtype = (v.qdq || reduceInFloat) ? ONNX_DT_FLOAT : v.dtype;
    finishSetup(rt, ep, g, modelBytes, ioDtype, D, profile,
                /*keepQdqUnfused=*/v.qdq && unfusable);
    return g;
  }

  // Mean microseconds per Run() over n runs; negative on failure.
  double timeRuns(const OrtRuntime &rt, GemmSetup &g, unsigned int n)
  {
    static const char *inNames[] = {"S"};
    static const char *outNames[] = {"Y"};

    auto t0 = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < n; i++)
    {
      OrtStatus *st = rt.api->Run(g.session, nullptr,
                                  inNames, (const OrtValue *const *)&g.inVal, 1,
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

} // namespace

int OnnxPeak::runGemm(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                      benchmark_config_t &cfg)
{
  (void)cfg;

  static const Variant kFpVariants[] = {
      {ONNX_DT_FLOAT, false, "fp32",
       "FP32 graph inputs and outputs.  A provider may use a narrower internal "
       "format; read the fp32 numeric-error row beside this one to see whether "
       "it did.  Many NPUs cannot run this at all, or route it away from the "
       "matrix hardware -- that is a finding, not a failure."},
      {ONNX_DT_FLOAT16, false, "fp16",
       "16-bit floats, the native currency of most NPU matrix hardware."},
      {ONNX_DT_BFLOAT16, false, "bf16",
       "The 16-bit float with fp32's exponent range and three fewer mantissa "
       "bits.  Modern matrix hardware usually runs it at the fp16 rate; a "
       "provider that falls well short of its own fp16 row is emulating it, "
       "and one that refuses it outright has no bf16 path at all."},
      {ONNX_DT_FLOAT8E4M3FN, true, "fp8_e4m3",
       "8-bit floats in QDQ form, in the variant that spends its bits on "
       "precision: four exponent bits and three of mantissa, reaching 448.  "
       "This is the format quantized inference actually uses when it moves below "
       "16 bits without going to integers."},
      {ONNX_DT_FLOAT8E5M2, true, "fp8_e5m2",
       "The other 8-bit float, trading a mantissa bit for an exponent one: it "
       "reaches 57344 and rounds more coarsely.  Hardware usually runs both at "
       "the same rate, so a difference between these two rows is the provider "
       "choosing different machinery, and the accuracy rows say what each costs."},
      {ONNX_DT_FLOAT4E2M1, true, "fp4_e2m1",
       "4-bit floating point on both operands: two exponent bits, one of "
       "mantissa, eight magnitudes in all and a largest value of 6.  This is the "
       "narrowest format current tensor cores implement, and unlike int4 there "
       "is a chance a provider fuses it into a real 4-bit multiply rather than "
       "unpacking it -- the row says which happened."},
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
       /*blockSize=*/32},
      {ONNX_DT_INT4, false, "int4_weight",
       "4-bit weights with one scale per 32 of them, against 16-bit activations "
       "-- the form quantized language models actually ship in.  The arithmetic "
       "is still 16-bit, because ONNX has no 4-bit multiply and the weights are "
       "unpacked on the way in, so this row is reported in TFLOPS and what four "
       "bits buys is a quarter of the weight traffic rather than a faster "
       "multiply.  On a square problem like this one that mostly shows up as "
       "matching the fp16 row; a provider well below it is unpacking badly.",
       /*blockSize=*/32},
  };
  static const Variant kIntVariants[] = {
      {ONNX_DT_INT8, true, "int8_qdq",
       "8-bit integers in QDQ form -- quantized in, quantized out, the shape "
       "quantized inference actually ships in.  This is what an NPU's headline "
       "TOPS figure is quoted for."},
  };

  auto test = currentDeviceScope->beginTest(
      {"onnx_gemm", "ONNX MatMul peak",
       "tflops",
       Category::Unknown,
       "Matrix-multiply speed through ONNX Runtime on this execution "
       "provider, using a single-operation model with constant weights.  "
       "The identical model runs on every provider, so NPU, GPU and CPU "
       "rows are directly comparable -- and the gap against a vendor's "
       "advertised TOPS is real, not an artifact of different test code.  "
       "Providers that cannot run an operation entirely on their device "
       "report it as unsupported instead of quietly measuring the CPU.  "
       "Each reading is a different input format.",
       TestShape::Heterogeneous, "data type"});

  // ---- Sweep every variant (all data types in one test) ---------------
  // Helper to run one variant's sweep
  auto runVariant = [&](const Variant &v) {
    const bool isInt = (v.dtype == ONNX_DT_INT8 && v.qdq);

    if (std::string why = onnxDtypeUnsupportedReason(rt, v.dtype); !why.empty())
    {
      logger::EmitOptions o;
      o.description = v.note;
      if (isInt)
        o.unit = "tops";
      test.skip(v.label, ResultStatus::Unsupported, why, o);
      return;
    }

    double best = 0.0;
    int64_t bestDim = 0;
    std::string firstErr;
    ResultStatus errStatus = ResultStatus::Unsupported;

    // First and last timings with their sizes, to confirm the work actually
    // happened (see the folding check after the loop).
    double firstUs = 0.0, lastUs = 0.0;
    int64_t firstDim = 0, lastDim = 0;
    double lastRate = 0.0;
    int strikes = 0;
    std::string ranAs; // kernel the provider actually used (quantized only)
    int actDtype = ONNX_DT_UINT8;
    int wgtDtype = ONNX_DT_INT8;
    const char *schemeName = "";
    bool castedActs = false; // provider converted the activations first
    // A provider can implement the matmul for a datatype and not the reduction
    // that keeps the result on the device: the CUDA EP multiplies bf16 and has
    // no bf16 ReduceMax.  Retried once, lazily, so a provider that needs
    // nothing pays nothing -- and never for the quantized form, whose reduction
    // already runs on a dequantized fp32 result.
    bool reduceInFloat = false, triedFloatReduce = false;

    // For the quantized variant, settle the scheme before sweeping anything.
    // ONNX Runtime rewrites graphs before running them, and a provider that
    // will not fuse dequantize/matmul/quantize into a quantized kernel
    // dequantizes the operands and multiplies in floating point instead --
    // a perfectly good number that is not an int8 number.  One profiled run
    // per scheme at the smallest size answers it, and doing it first means a
    // provider that fuses neither is not swept at all.
    // Both quantized shapes need the same question answered before they are
    // swept: did the provider fuse, or did it unpack the weights into floats
    // and run an ordinary multiply?  For weight-only the unfused form pays for
    // a full dequantize of the matrix on every single run, so it is not the
    // rate a 4-bit deployment would ever see.
    if (v.qdq || v.blockSize > 0)
    {
      std::string tried;
      QuantScheme schemes[2];
      const size_t nSchemes = schemesFor(v, schemes);
      for (size_t si = 0; si < nSchemes; si++)
      {
        const QuantScheme &qs = schemes[si];
        if (clpeak::cancelRequested())
          break;
        GemmSetup probe = makeSetup(rt, ep, v, kMinDim, /*profile=*/true,
                                    qs.actDtype, /*reduceInFloat=*/false,
                                    qs.wDtype);
        if (!probe.session)
        {
          if (firstErr.empty())
            firstErr = probe.error;
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %s rejected: %s\n",
                      ep.providerKey.c_str(), v.label, qs.name,
                      probe.error.c_str());
          continue;
        }
        timeRuns(rt, probe, 1);
        auto ops = onnxCollectExecutedOps(rt, probe.session);
        destroySetup(rt, probe);

        std::string joined;
        for (const auto &o : ops)
          joined += (joined.empty() ? "" : ", ") + o;
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %s executed %s\n",
                    ep.providerKey.c_str(), v.label, qs.name, joined.c_str());

        if (onnxOpsRanQuantizedMatMul(ops))
        {
          actDtype = qs.actDtype;
          wgtDtype = qs.wDtype;
          schemeName = qs.name;
          for (const auto &o : ops)
            if (o == "Cast")
              castedActs = true;
          // A provider that compiles the whole subgraph names its kernel
          // after itself, and that name carries a hash of the graph -- it
          // would differ between runs and has no place in a saved result.
          const std::string named = onnxQuantizedKernelName(ops);
          ranAs = named.empty() ? "a kernel it compiled itself" : named;
          break;
        }
        if (!ops.empty())
          tried = joined;
      }

      if (ranAs.empty())
      {
        logger::EmitOptions o;
        o.description = std::string("Peak over a doubling sweep of square "
                                    "sizes.  ") +
                        v.note;
        if (isInt)
          o.unit = "tops";
        test.skip(v.label,
                  tried.empty() ? errStatus : ResultStatus::Unsupported,
                  tried.empty()
                      ? (firstErr.empty()
                             ? std::string("this provider accepted no session "
                                           "for ") +
                                   v.label
                             : firstErr)
                      : std::string("provider did not fuse a quantized matmul "
                                    "-- it dequantized the operands and multiplied in "
                                    "floating point, so this is not a ") +
                            v.label +
                            " rate (ran: " + tried + ")",
                  o);
        return;
      }
    }

    for (int64_t D = kMinDim; D <= kMaxDim; D *= 2)
    {
      if (clpeak::cancelRequested())
        break;

      // Would this size fit, and would one iteration finish in reasonable
      // time at the rate the previous size managed?
      // Weight-only carries fp16 activations, packed nibbles and a scale per
      // block, not two equal operands -- and on a phone an under-estimate here
      // is an out-of-memory kill rather than a slow row.
      const uint64_t elems = (uint64_t)D * (uint64_t)D;
      const uint64_t weightBytes =
          v.nvfp4
              ? 2ull * (onnxElemBytes(ONNX_DT_FLOAT4E2M1, (int64_t)elems) + elems / (uint64_t)v.blockSize)
          : (v.blockSize > 0)
              ? (elems * 2ull                             // fp16 A
                 + onnxElemBytes(v.dtype, (int64_t)elems) // packed weights
                 + elems / (uint64_t)v.blockSize * 2ull)  // fp16 scales
              : 2ull * onnxElemBytes(v.dtype, (int64_t)elems);
      if (weightBytes > maxWeightBytes())
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 needs %llu MB of operands, "
                    "stopping\n",
                    ep.providerKey.c_str(), v.label,
                    (long long)D, (unsigned long long)(weightBytes >> 20));
        break;
      }
      if (lastRate > 0.0)
      {
        const double predictedUs =
            2.0 * (double)D * (double)D * (double)D / lastRate;
        if (predictedUs > kMaxIterUs)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 would take ~%.1f s per "
                      "iteration, stopping\n",
                      ep.providerKey.c_str(),
                      v.label, (long long)D, predictedUs / 1.0e6);
          break;
        }
      }

      GemmSetup g = makeSetup(rt, ep, v, D, /*profile=*/false, actDtype,
                              reduceInFloat, wgtDtype);
      if (!g.session && !v.qdq && !triedFloatReduce)
      {
        triedFloatReduce = true;
        // Keep the native refusal: when the cast form fails too, the cause is
        // almost always the matmul rather than the reduction, and the first
        // message is the one that says so.
        const std::string nativeErr = g.error;
        CLPEAK_VLOG("onnx-gemm[%s/%s]: native reduction refused (%s), "
                    "retrying with the product cast to fp32\n",
                    ep.providerKey.c_str(), v.label, nativeErr.c_str());
        destroySetup(rt, g);
        g = makeSetup(rt, ep, v, D, /*profile=*/false, actDtype,
                      /*reduceInFloat=*/true, wgtDtype);
        if (g.session)
          reduceInFloat = true;
        else
          g.error = nativeErr;
      }
      if (!g.session)
      {
        if (firstErr.empty())
          firstErr = g.error;
        // Larger sizes need strictly more of everything, so nothing above
        // this one can succeed either.
        break;
      }

      double per_iter_us = -1.0;
      if (timeRuns(rt, g, 1 + warmupCount) > 0.0) // compile + warmup
        per_iter_us = timeRuns(rt, g, 1);         // calibration probe
      if (per_iter_us <= 0.0)
      {
        if (firstErr.empty())
        {
          firstErr = g.error.empty() ? "run failed" : g.error;
          errStatus = ResultStatus::Error;
        }
        destroySetup(rt, g);
        break;
      }

      unsigned int iters = pickIters(per_iter_us, kSizeBudgetUs,
                                     forceIters ? specifiedIters : 0,
                                     kOnnxMaxIters);
      // The probe was one whole iteration, so when the budget affords only
      // one, it already is the measurement -- and at that end of the ladder
      // repeating it is the most expensive thing the sweep does.
      double mean_us = (iters > 1) ? timeRuns(rt, g, iters) : per_iter_us;
      if (mean_us <= 0.0 && firstErr.empty())
      {
        firstErr = g.error.empty() ? "run failed" : g.error;
        errStatus = ResultStatus::Error;
      }
      destroySetup(rt, g);
      if (mean_us <= 0.0)
        break;

      if (firstUs == 0.0)
      {
        firstUs = mean_us;
        firstDim = D;
      }
      lastUs = mean_us;
      lastDim = D;

      const double ops = 2.0 * (double)D * (double)D * (double)D;
      const double rate = ops * 1.0e6 / mean_us / 1.0e12;
      // Rate per FLOP-count is what the ladder is searching on; the raw rate
      // in ops/us drives the time prediction for the next rung.
      lastRate = ops / mean_us;
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 -> %.3f\n", ep.providerKey.c_str(),
                  v.label, (long long)D, rate);

      if (rate > best * kImproveFactor)
      {
        strikes = 0;
        best = rate;
        bestDim = D;
      }
      else
      {
        if (rate > best)
        {
          best = rate;
          bestDim = D;
        }
        if (++strikes >= kMaxStrikes)
        {
          CLPEAK_VLOG("onnx-gemm[%s/%s]: no further gain past %lld^3\n",
                      ep.providerKey.c_str(), v.label, (long long)bestDim);
          break;
        }
      }

      // A rung that measured slower than the ceiling ends the ladder here.
      //
      // The gate at the top of the loop asks the same question of the *next*
      // size, but it has to answer it by extrapolating from the rate of the
      // size before -- and an extrapolation cannot see a cliff, which is
      // exactly what it is being asked to look for.  A provider that falls
      // off one reads as fast right up to the rung that collapses: Core ML
      // runs fp16 at 6.1 TFLOPS at 4096 and 0.34 at 8192, so the prediction
      // for 8192 came out nineteen times short of the truth.
      //
      // This one is not a prediction.  The rung has been measured, it took
      // longer than a whole iteration is allowed to take, and the next size
      // is eight times the work -- so there is nothing above this worth the
      // wait, whatever the rate did.
      if (per_iter_us > kMaxIterUs)
      {
        CLPEAK_VLOG("onnx-gemm[%s/%s]: %lld^3 measured %.1f s per iteration, "
                    "stopping\n",
                    ep.providerKey.c_str(), v.label,
                    (long long)D, per_iter_us / 1.0e6);
        break;
      }
    }

    // Both operands are constants, so this test depends on ORT honouring the
    // request not to fold them; if it ever stopped, the matmul would be
    // evaluated once at load time and every timed run would measure an empty
    // graph.  Real work grows with the cube of the size -- 64x across this
    // ladder -- so anything close to flat means nothing was computed.  Better
    // an error than a spectacular number.
    // Folding collapses the graph to a constant, so what remains is dispatch
    // and the time stops tracking the size at all -- 143 us at 1024 against
    // 162 us at 8192.  Real work grows as the cube of the size divided by
    // whatever the rate gained along the way, and that gain can be large:
    // TensorRT's int8 rate improves 9.4x between 1024 and 16384, so its time
    // grows 433x where the work grew 4096x.  The tolerance has to clear that
    // comfortably or the guard discards the very measurements it exists to
    // protect -- a factor of 8 threw away a correct 124 TOPS reading.  A rate
    // improving 64x across one ladder has never been observed.
    double expectedGrowth = 1.0;
    for (int64_t d = firstDim; d > 0 && d < lastDim; d *= 2)
      expectedGrowth *= 8.0; // each doubling is 8x work
    if (best > 0.0 && firstUs > 0.0 && lastDim > firstDim &&
        lastUs < firstUs * expectedGrowth / 64.0)
    {
      CLPEAK_VLOG("onnx-gemm[%s/%s]: %.1f us at %lld vs %.1f us at %lld -- "
                  "work does not scale, constants were folded\n",
                  ep.providerKey.c_str(), v.label, firstUs,
                  (long long)firstDim, lastUs, (long long)lastDim);
      best = 0.0;
      firstErr = "this runtime folded the operands at load time: it accepted "
                 "the request to disable constant folding and ignored it, "
                 "which ONNX Runtime did before about 1.18, so the timings "
                 "do not scale with the problem size and mean nothing";
      errStatus = ResultStatus::Error;
    }

    logger::EmitOptions o;
    if (best > 0.0)
    {
      o.description = "Peak over a doubling sweep of square sizes; fastest at " + std::to_string(bestDim) + " cubed.  " + v.note;
      if (!ranAs.empty())
        o.description += "  The provider ran the multiply as " + ranAs +
                         " with " + schemeName +
                         ", confirming it really ran in " + v.label + ".";
      if (castedActs)
        o.description += "  This provider does not take the activations in the "
                         "width they were given, so it converts them first: "
                         "that is a full pass over them on every run and it is "
                         "inside this figure.";
      if (reduceInFloat)
        o.description += "  This provider has no reduction for the datatype, "
                         "so the product is cast to fp32 before being reduced; "
                         "the multiply itself is unaffected, but the cast is a "
                         "full pass over the result and costs a few percent.";
      if (isInt)
        o.unit = "tops";
      test.emit(v.label, (float)best, o);
    }
    else
    {
      o.description = std::string("Peak over a doubling sweep of square sizes.  ") + v.note;
      if (isInt)
        o.unit = "tops";
      test.skip(v.label, errStatus,
                firstErr.empty() ? "no supported datatype" : firstErr, o);
    }
  };
  for (size_t i = 0; i < sizeof(kFpVariants)/sizeof(kFpVariants[0]); i++)
  {
    if (clpeak::cancelRequested()) break;
    runVariant(kFpVariants[i]);
  }
  for (size_t i = 0; i < sizeof(kIntVariants)/sizeof(kIntVariants[0]); i++)
  {
    if (clpeak::cancelRequested()) break;
    runVariant(kIntVariants[i]);
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
