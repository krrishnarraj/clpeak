#ifdef ENABLE_ONNX

// onnx-block: one fixed transformer decoder block, run in the two regimes
// that bound all LLM inference, at each precision a language model actually
// ships in.  This is the rung above a raw GEMM peak and below tokens/second:
// an intermediate number that is meaningful, comparable across completely
// different hardware, and needs no model download.
//
//   prefill (64/512/2048 tokens at once)      compute-bound -> effective FLOPS
//   decode  (1 token, 512/2048/8192 context)  memory-bound  -> effective B/s
//
// Both numbers come out of the whole stack -- graph scheduling, layout
// conversions, softmax, the lot -- not just the matmuls, so they are what a
// real pipeline can actually reach rather than what the silicon could do in
// principle.  Multiply the latency rows by a model's layer count to sanity
// check any tokens/second claim made for this device.
//
// Six timings per precision, three scopes, one unit each, and nothing
// restated: the prompt ladder is onnx-block-prefill (flops, or ops where the
// arithmetic is integer), the 2048-context token is onnx-block-decode (bps)
// because that is the row onnx-tensor-bw compares against, and every timing
// that is worth reading as a duration lands in onnx-block-latency (s).
// Splitting a ladder's headline rung into a scope of its own is what to avoid
// here: the rung and the scope hold the same timing in the same unit, so one
// of the two rows is pure repetition.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_probe.h"
#include "onnx_session.h"

#include <chrono>
#include <map>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace
{

  // Fixed geometry: llama-style proportions (SwiGLU hidden = 2.6875x d_model,
  // 128-wide heads), sized so the weights -- 50.6M parameters, 101 MB at fp16
  // -- overflow every cache on every device while still compiling in seconds
  // on the NPU toolchains, which build graphs ahead of time.  A 7B block would
  // be 4x this and push AOT compile times into minutes for no extra insight.
  constexpr int64_t kDModel = 2048;
  constexpr int64_t kHeads = 16;
  constexpr int64_t kHeadDim = 128;
  constexpr int64_t kFfnHidden = 5504;
  constexpr int64_t kPrefillSeq = 512;
  constexpr int64_t kDecodeKv = 2048;

  // Context lengths for the decode rows of onnx-block-latency.  Each is a
  // separate graph with its own cache baked in -- 16 MB at 2048, growing with
  // the length -- so each costs a session.  It stops at 8192 not because longer
  // contexts are uninteresting but because providers that compile ahead of time
  // charge dearly for the bigger graphs: Core ML needs the better part of a
  // minute per session here.
  // Rows are named by length, so a longer rung can be appended later without
  // changing what any existing one means.
  const int64_t kKvLadder[] = {512, 2048, 8192};

  // Prompt lengths for onnx-block-prefill.  The 512 rung is the one to quote on
  // its own; the other two are what show where the device saturates.
  const int64_t kPromptLadder[] = {64, 512, 2048};

  // Block size for the weight-only rows: one scale per 32 weights along the
  // reduction axis, the same grouping onnx-gemm's int4_weight row uses and the
  // one AWQ, GPTQ and MatMulNBits all default to.  Sharing it is what makes a
  // block reading divisible by a GEMM reading.
  constexpr int64_t kWeightBlock = 32;

  // Budget for one point's timed phase.  It doubles as the affordability test:
  // a point whose single iteration costs more than the entire budget for
  // measuring it cannot be measured properly anyway, so it is skipped.
  constexpr unsigned int kBlockBudgetUs = 5000000;

  // The precisions the block is run in.
  //
  // A datatype is the wrong axis for a layer, and it is why the fp16-only
  // version of this test could not say what its TFLOPS figure was a figure
  // *of*.  A GEMM has one operand pair, so "dtype" is one word; a layer has
  // weights and an arithmetic width and real deployments vary them
  // independently.  What models ship as are pairs -- W16A16, W8A8, W4A16 -- and
  // those are the rows.
  //
  // Labels are onnx-gemm's, deliberately: `int4_weight` there and
  // `int4_weight` here are the same format under the same name, so the block
  // reading divides by the GEMM reading and the quotient is how much of the raw
  // matmul rate a complete layer retains in that format.
  struct Variant
  {
    const char *label;
    int actDtype;     // arithmetic width: attention, softmax, residuals
    int wDtype;       // how the seven projection weights are stored
    int64_t wBlock;   // >0: blocked weight-only, one scale per this many
    bool qdq;         // quantize the activations too
    bool sweep;       // walk the whole prompt and context ladders
    const char *unit; // nullptr: the scope's own unit
    const char *note;
    int kvDtype;     // 0: the arithmetic width; else a quantized cache
    bool decodeOnly; // prefill has no cache to store, so it has no row
  };

  // fp16 first: it is the reference every other row is read against, and
  // running it first is also what lets the affordability gate size the rest of
  // the table from a provider that has already been timed once.
  //
  // Which of these get the full ladders is a cost decision.  Every point is its
  // own session, and a session is where an ahead-of-time provider spends its
  // minute, so a full cross product would be 36 of them.  Two rows sweep --
  // fp16 because it is the reference, int4_weight because its prompt ladder is
  // a measurement rather than a repetition (see the note there) -- and the rest
  // take the headline rung of each regime, which is the number anyone quotes.
  const Variant kVariants[] = {
      {"fp16", ONNX_DT_FLOAT16, ONNX_DT_FLOAT16, 0, false, /*sweep=*/true, nullptr,
       "16-bit weights and 16-bit arithmetic, the form an unquantized model is "
       "served in and the reference the other rows are read against."},

      {"int4_weight", ONNX_DT_FLOAT16, ONNX_DT_INT4, kWeightBlock, false,
       /*sweep=*/true, nullptr,
       "4-bit weights, one scale per 32, against 16-bit activations -- what a "
       "quantized language model ships as.  The arithmetic stays 16-bit, so "
       "four bits buys weight traffic rather than rate."},

      {"fp4_weight", ONNX_DT_FLOAT16, ONNX_DT_FLOAT4E2M1, kWeightBlock, false,
       /*sweep=*/false, nullptr,
       "The int4 row's shape exactly, with the four bits spent on a float "
       "instead of an integer, so a gap between the two is the format alone."},

      {"int8_weight", ONNX_DT_FLOAT16, ONNX_DT_INT8, kWeightBlock, false,
       /*sweep=*/false, nullptr,
       "8-bit weights, blocked the same way, against 16-bit activations.  This "
       "narrows only the weights where int8_qdq narrows the arithmetic too, so "
       "the gap between them is what the integer units are worth."},

      {"int8_qdq", ONNX_DT_FLOAT16, ONNX_DT_INT8, 0, /*qdq=*/true,
       /*sweep=*/false, "ops",
       "8-bit weights and 8-bit arithmetic through the projections, quantized "
       "in and out -- what headline TOPS figures are quoted for, measured on a "
       "whole layer.  Attention and the softmax stay 16-bit, as they do in "
       "every real deployment."},

      {"fp32", ONNX_DT_FLOAT, ONNX_DT_FLOAT, 0, false, /*sweep=*/false, nullptr,
       "Full precision, which nobody serves a language model in, here as a "
       "control: a provider whose fp16 row fails to beat it is not running "
       "half-precision hardware."},

      {"bf16", ONNX_DT_BFLOAT16, ONNX_DT_BFLOAT16, 0, false, /*sweep=*/false,
       nullptr,
       "The 16-bit float with fp32's exponent range and three fewer mantissa "
       "bits.  A layer needs bf16 kernels for every operation in it, not just "
       "the matmul, so a refusal here beside a working MatMul bf16 row is that "
       "gap."},

      {"fp8_e4m3", ONNX_DT_FLOAT16, ONNX_DT_FLOAT8E4M3FN, 0, /*qdq=*/true,
       /*sweep=*/false, nullptr,
       "8-bit floats through the projections, quantized in and out, against "
       "16-bit attention.  Unlike int8 it keeps exponent range for activations "
       "that have some, and it reports in TFLOPS because it is a float."},

      {"int8_kv", ONNX_DT_FLOAT16, ONNX_DT_FLOAT16, 0, false, /*sweep=*/true,
       nullptr,
       "16-bit throughout with only the cached context stored as 8-bit "
       "integers -- the axis that decides how long a conversation a device can "
       "hold, which is why this row sweeps context.  It counts only if the "
       "provider folds the dequantize into its attention kernel; reading the "
       "cache back at full width every token is refused.",
       /*kvDtype=*/ONNX_DT_INT8, /*decodeOnly=*/true},
  };

  int64_t weightParams()
  {
    return 4 * kDModel * kDModel      // Wq, Wk, Wv, Wo
           + 2 * kDModel * kFfnHidden // Wg, Wu
           + kFfnHidden * kDModel;    // Wd
  }

  // Bytes of projection weights this variant declares, blocked scales
  // included.  This is the model's own size: what has to be built, uploaded
  // and held, which is what the memory gate asks about.
  uint64_t weightBytes(const Variant &v)
  {
    const int64_t n = weightParams();
    uint64_t bytes = onnxElemBytes(v.wDtype, n);
    if (v.wBlock > 0)
      bytes += (uint64_t)(n / v.wBlock) * 2ull; // one fp16 scale per block
    return bytes;
  }

  uint64_t kvBytes(const Variant &v, int64_t kv)
  {
    // K and V, every head, at whatever width the cache is stored in.
    const int dt = v.kvDtype ? v.kvDtype : v.actDtype;
    return 2ull * (uint64_t)kHeads * (uint64_t)kv * (uint64_t)kHeadDim * onnxElemBytes(dt, 1);
  }

  // The same two, as the provider actually streams them -- the numerator of
  // the decode row, which has to be what moved.
  //
  // They differ only where a provider stores an fp32 graph at 16 bits, which
  // QNN's HTP and OpenVINO's GPU and NPU targets do by default: the model
  // declares four bytes an element and the hardware reads two.  Counting the
  // declared width there doubles the row -- OpenVINO's GPU reported fp32 at
  // 110 GB/s against fp16's 54.8 off the identical 2.15 ms, which is one
  // measurement wearing two numbers.  Every other row declares what it
  // stores, so this is the fp32 rows and nothing else.
  uint64_t streamedWeightBytes(const Variant &v, bool fp32As16)
  {
    uint64_t bytes = weightBytes(v);
    if (fp32As16 && v.wDtype == ONNX_DT_FLOAT)
      bytes /= 2;
    return bytes;
  }

  uint64_t streamedKvBytes(const Variant &v, int64_t kv, bool fp32As16)
  {
    uint64_t bytes = kvBytes(v, kv);
    const int dt = v.kvDtype ? v.kvDtype : v.actDtype;
    if (fp32As16 && dt == ONNX_DT_FLOAT)
      bytes /= 2;
    return bytes;
  }

  std::vector<int64_t> promptsFor(const Variant &v)
  {
    // A cache-format row has no prefill reading: prefill builds K and V from the
    // pass it is measuring, so there is no stored cache for its format to apply
    // to.  An absent row beats one that silently measures the fp16 graph.
    if (v.decodeOnly)
      return {};
    if (v.sweep)
      return {kPromptLadder[0], kPromptLadder[1], kPromptLadder[2]};
    return {kPrefillSeq};
  }

  std::vector<int64_t> contextsFor(const Variant &v)
  {
    if (v.sweep)
      return {kKvLadder[0], kKvLadder[1], kKvLadder[2]};
    return {kDecodeKv};
  }

  // The opset the model for this variant will declare, mirroring onnxBlockModel.
  // Checking it here rather than letting the load fail turns "IR version 9 is
  // not supported" into a sentence naming the datatype that asked for it.
  int variantOpset(const Variant &v)
  {
    int opset = onnxOpsetForDtype(v.actDtype);
    const int w = onnxOpsetForDtype(v.wDtype);
    if (w > opset)
      opset = w;
    if (v.kvDtype)
    {
      const int k = onnxOpsetForDtype(v.kvDtype);
      if (k > opset)
        opset = k;
      // Half-precision dequantize scales are opset 19; see onnxBlockModel.
      if (v.actDtype != ONNX_DT_FLOAT && opset < 19)
        opset = 19;
    }
    if (v.wBlock > 0 && opset < 21)
      opset = 21;
    return opset;
  }

  // Multiply-accumulates x2, over every matmul in the block.
  double blockFlops(int64_t seq, int64_t ctx)
  {
    const double S = (double)seq, C = (double)ctx;
    const double d = (double)kDModel, ffn = (double)kFfnHidden;
    const double H = (double)kHeads, Dh = (double)kHeadDim;

    const double qkv = 3.0 * 2.0 * S * d * d;
    const double attn = 2.0 * 2.0 * H * S * C * Dh; // scores + context
    const double proj = 2.0 * S * d * d;
    const double ff = 2.0 * 2.0 * S * d * ffn + 2.0 * S * ffn * d;
    return qkv + attn + proj + ff;
  }

  // Did the projections run as quantized kernels, or did the provider unpack
  // the weights and multiply in floating point?
  //
  // onnx_session.cpp's onnxOpsRanQuantizedMatMul cannot answer this one.  It
  // reads a failed fusion as "a plain MatMul beside the dequantize nodes", and a
  // transformer block has two plain MatMuls that are *supposed* to be there --
  // attention is not quantized in any variant here -- so that test would reject
  // every block unconditionally.
  //
  // The same inverted reasoning still works on this graph.  A provider that
  // fused either names a quantized kernel (QLinearMatMul, MatMulNBits) or
  // swallowed the subgraph into one kernel of its own, and in that case no
  // DequantizeLinear kernel runs at all.  A provider that did not fuse has no
  // choice but to execute DequantizeLinear as a real kernel -- a full pass over
  // 101 MB of weights, on every single run -- and that is the reading worth
  // refusing, because it is not a rate any four-bit deployment would ever see.
  bool projectionsRanQuantized(const std::vector<std::string> &ops)
  {
    if (ops.empty())
      return true; // no profile to judge by; do not reject on silence

    for (const auto &op : ops)
      if (op == "DequantizeLinear" || op == "QuantizeLinear")
        return !onnxQuantizedKernelName(ops).empty();
    return true;
  }

  struct BlockRun
  {
    OrtSession *session = nullptr;
    OrtValue *inVal = nullptr;
    std::vector<OrtValue *> outVals;
    std::vector<const char *> outNames;
    std::vector<uint8_t> inBuf;
    std::string error;
  };

  void destroyRun(const OrtRuntime &rt, BlockRun &r)
  {
    if (r.inVal)
      rt.api->ReleaseValue(r.inVal);
    for (OrtValue *v : r.outVals)
      if (v)
        rt.api->ReleaseValue(v);
    if (r.session)
      rt.api->ReleaseSession(r.session);
    r.inVal = nullptr;
    r.outVals.clear();
    r.session = nullptr;
    r.inBuf.clear();
    r.inBuf.shrink_to_fit();
    // `error` survives: callers tear a failed run down and then report it.
  }

  BlockRun makeRun(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                   const Variant &v, bool decode, int64_t kvLen,
                   int64_t prefillSeq, int qActDtype, bool profile)
  {
    BlockRun r;

    OnnxBlockShape sh;
    sh.dModel = kDModel;
    sh.heads = kHeads;
    sh.headDim = kHeadDim;
    sh.ffnHidden = kFfnHidden;
    sh.seq = decode ? 1 : prefillSeq;
    sh.kvLen = decode ? kvLen : 0;
    sh.actDtype = v.actDtype;
    sh.wDtype = v.wDtype;
    sh.wBlock = v.wBlock;
    sh.qdq = v.qdq;
    sh.qActDtype = qActDtype;
    sh.kvDtype = v.kvDtype;

    {
      std::string model = onnxBlockModel(sh);
      // Constant folding stays off for every variant, and it is the quantized
      // ones that need it: a weight DequantizeLinear has nothing but constants
      // on its inputs, so folding it would bake fp16 weights into the model at
      // load time and every timed run would measure the fp16 row under another
      // name.  The floating-point variants have nothing foldable -- the runtime
      // scalar makes everything downstream of it non-constant -- so disabling it
      // uniformly costs them nothing and keeps all the rows under one optimizer
      // setting, which is what makes them comparable.
      // Hold the QDQ selector off for a float8 graph, and only for one.  It
      // rewrites DequantizeLinear/MatMul/QuantizeLinear into QLinearMatMul,
      // which is an 8-bit *integer* operator with no float8 type constraint, so
      // on a float8 graph the rewrite turns a valid model into one that fails
      // its own type check.  Hardware with real float8 matmul consumes the QDQ
      // nodes itself and never wanted the rewrite; int8 does want it, and still
      // gets it.
      const bool keepQdqUnfused = v.qdq && (!onnxQdqFusionIsLegal(qActDtype) ||
                                            !onnxQdqFusionIsLegal(v.wDtype));
      auto ses = onnxCreateSession(rt, ep, model, /*keepConstantsUnfolded=*/true,
                                   profile, keepQdqUnfused);
      // The model is the largest allocation in the process; drop it before
      // anything else is allocated on top of the session's own copy.
      model.clear();
      model.shrink_to_fit();
      if (!ses.session)
      {
        r.error = ses.error;
        return r;
      }
      r.session = ses.session;
    }

    // The only input is the scalar that scales the resident activations.
    const size_t es = (size_t)onnxElemBytes(v.actDtype, 1);
    r.inBuf.assign(es, 0);
    {
      const float one = 1.0009765625f;
      if (v.actDtype == ONNX_DT_FLOAT)
        std::memcpy(r.inBuf.data(), &one, 4);
      else
      {
        uint16_t h = (v.actDtype == ONNX_DT_BFLOAT16) ? floatToBf16(one)
                                                      : floatToHalf(one);
        std::memcpy(r.inBuf.data(), &h, 2);
      }
    }

    // Decode also returns the new K/V (the cache write); let ORT allocate
    // those, so only the input needs a bound buffer here.
    r.outNames.push_back("Yr");
    if (decode)
    {
      r.outNames.push_back("Knew");
      r.outNames.push_back("Vnew");
    }
    r.outVals.assign(r.outNames.size(), nullptr);

    OrtMemoryInfo *mi = nullptr;
    OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                                OrtMemTypeDefault, &mi);
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, r.inBuf.data(), r.inBuf.size(), nullptr, 0,
          (ONNXTensorElementDataType)v.actDtype, &r.inVal);
    if (mi)
      rt.api->ReleaseMemoryInfo(mi);
    if (st)
    {
      r.error = onnxStatusText(rt, st);
      destroyRun(rt, r);
    }
    return r;
  }

  // Mean microseconds per block; negative on failure.
  // (declared before measure(), which uses it)
  double timeRuns(const OrtRuntime &rt, BlockRun &r, unsigned int n)
  {
    static const char *inNames[] = {"S"};

    auto t0 = std::chrono::steady_clock::now();
    for (unsigned int i = 0; i < n; i++)
    {
      // ORT allocates the outputs each call; release the previous set first so
      // a long timed phase does not grow without bound.
      for (OrtValue *&v : r.outVals)
      {
        if (v)
          rt.api->ReleaseValue(v);
        v = nullptr;
      }
      OrtStatus *st = rt.api->Run(r.session, nullptr,
                                  inNames, (const OrtValue *const *)&r.inVal, 1,
                                  r.outNames.data(), r.outNames.size(),
                                  r.outVals.data());
      if (st)
      {
        r.error = onnxStatusText(rt, st);
        return -1.0;
      }
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / n;
  }

  // Time one regime end to end.  Returns mean us/block, or negative with
  // `error` set.
  double measure(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                 const Variant &v, bool decode, int qActDtype,
                 unsigned int warmup, bool forceIters, unsigned int forced,
                 std::string &error, ResultStatus &status,
                 int64_t kvLen, int64_t prefillSeq)
  {
    auto createStart = std::chrono::steady_clock::now();
    BlockRun r = makeRun(rt, ep, v, decode, kvLen, prefillSeq, qActDtype,
                         /*profile=*/false);
    auto createEnd = std::chrono::steady_clock::now();
    double createUs = std::chrono::duration<double, std::micro>(
                          createEnd - createStart).count();
    CLPEAK_VLOG("onnx-block[%s/%s]: %s create %.1f s\n",
                ep.providerKey.c_str(), v.label,
                decode ? ("decode_kv" + std::to_string(kvLen)).c_str()
                       : ("prefill_s" + std::to_string(prefillSeq)).c_str(),
                createUs / 1.0e6);
    if (r.session && createUs > kOnnxMaxBlockCreateUs)
    {
      CLPEAK_VLOG("onnx-block[%s/%s]: create %.1f s > %.1f s, skipping\n",
                  ep.providerKey.c_str(), v.label,
                  createUs / 1.0e6, kOnnxMaxBlockCreateUs / 1.0e6);
      error = "session creation took " +
              std::to_string((long long)(createUs / 1.0e6)) +
              " s, exceeds " +
              std::to_string((long long)(kOnnxMaxBlockCreateUs / 1.0e6)) +
              " s compilation budget";
      status = ResultStatus::Unsupported;
      destroyRun(rt, r);
      return -1.0;
    }
    if (!r.session)
    {
      error = r.error;
      status = ResultStatus::Unsupported;
      return -1.0;
    }

    double per_iter_us = -1.0;
    if (timeRuns(rt, r, 1 + warmup) > 0.0) // graph compile + warmup
      per_iter_us = timeRuns(rt, r, 1);    // calibration probe
    if (per_iter_us <= 0.0)
    {
      error = r.error.empty() ? "run failed" : r.error;
      status = ResultStatus::Error;
      destroyRun(rt, r);
      return -1.0;
    }

    unsigned int iters = pickIters(per_iter_us, kBlockBudgetUs,
                                   forceIters ? forced : 0, kOnnxMaxIters);
    // The probe was one whole block; when the budget affords only one, it
    // already is the measurement.
    double mean_us = (iters > 1) ? timeRuns(rt, r, iters) : per_iter_us;
    if (mean_us <= 0.0)
    {
      error = r.error.empty() ? "run failed" : r.error;
      status = ResultStatus::Error;
    }
    destroyRun(rt, r);
    return mean_us;
  }

  struct Point
  {
    double us = -1.0;
    std::string error;
    ResultStatus status = ResultStatus::Ok;
  };

  // Everything measured for one precision, plus why it was not.
  struct VariantResult
  {
    bool usable = false;
    std::string skipReason;
    ResultStatus skipStatus = ResultStatus::Unsupported;

    int qActDtype = ONNX_DT_INT8;
    const char *schemeName = "";
    std::string ranAs; // the fused kernel, quantized rows only
    bool castedActs = false;

    std::map<int64_t, Point> prefill, decode;
  };

  // Quantization schemes for the QDQ form, tried in order until one fuses.
  //
  // int8 has two spellings and no provider takes both: x86 MLAS without VNNI
  // implements unsigned activations against signed weights and declines to fuse
  // the signed form, while TensorRT rejects uint8 outright.  Trying is the only
  // way to know, exactly as in gemm.cpp -- the fusion check is the selector.
  // The float8 formats have one spelling: activations and weights share the
  // type, and there is no signed/unsigned question because they are signed
  // floats.
  struct Scheme
  {
    int actDtype;
    const char *name;
  };

  size_t qdqSchemesFor(const Variant &v, Scheme out[2])
  {
    if (v.wDtype == ONNX_DT_INT8)
    {
      out[0] = {ONNX_DT_INT8, "signed activations"};    // TensorRT, ARM
      out[1] = {ONNX_DT_UINT8, "unsigned activations"}; // x86 without VNNI
      return 2;
    }
    out[0] = {v.wDtype, "matching activations and weights"};
    return 1;
  }

} // namespace

int OnnxPeak::runBlock(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                       benchmark_config_t &cfg)
{
  (void)cfg;

  constexpr size_t kNVariants = sizeof(kVariants) / sizeof(kVariants[0]);
  std::vector<VariantResult> results(kNVariants);

  auto isIntVariant = [](const Variant &v) -> bool
  {
    return v.unit != nullptr;
  };

  // Providers that serve an fp32 graph at 16 bits by default (QNN HTP,
  // OpenVINO GPU/NPU).  That is their default mode and what an fp32 model
  // actually gets on that hardware, so it stays measured -- but the fp32 rows
  // then say so, and the decode row counts the bytes that moved rather than
  // the ones the model declared.
  const bool fp32As16 = onnxEpRunsFp32AsFp16(ep);
  auto fp32Note = [fp32As16](const Variant &v) -> const char * {
    if (!fp32As16 || v.actDtype != ONNX_DT_FLOAT)
      return "";
    return "  This provider runs fp32 graphs at 16-bit precision by default, "
           "which is how an fp32 model reaches its hardware at all, so this "
           "row is that conversion rather than full precision; the fp32 "
           "numeric-error row shows the cost.";
  };

  auto provenanceEarly = [](const Variant &v, const VariantResult &vr)
  {
    if (vr.ranAs.empty())
      return std::string();
    std::string s = "  Ran as " + vr.ranAs;
    if (vr.schemeName[0])
      s += " (" + std::string(vr.schemeName) + ")";
    s += ".";
    if (vr.castedActs)
      s += "  The provider casts the activations to the width its kernel "
           "wanted, a full pass inside this figure.";
    return s;
  };

  // The geometry is identical on every row of every variant, so it is stated
  // once in each test description rather than pasted onto each reading.
  const char *geometryEarly = "";

  // The affordability seed, carried across variants.  Each one's first point
  // has no measurement of its own to predict from, and letting all six pay
  // full price on a provider that has already proved itself slow is how this
  // test would come to take tens of minutes.  The most recent measured rate
  // is the estimate: precisions differ by a factor of a few, not orders.
  double refPrefillRate = 0.0, refDecodeRate = 0.0;

  auto emitPrefillTo = [&](logger::TestScope &test, const Variant &vv,
                           const VariantResult &vvr)
  {
    const std::string prov = provenanceEarly(vv, vvr);
    for (int64_t sseq : promptsFor(vv))
    {
      const std::string metric = std::string(vv.label) + "_s" + std::to_string(sseq);
      logger::EmitOptions o;
      o.description = std::string(geometryEarly) + vv.note + "  A prompt of " +
                      std::to_string(sseq) +
                      " tokens in one pass, counting every multiply in the "
                      "layer." +
                      prov + fp32Note(vv);
      if (vv.unit)
        o.unit = vv.unit;
      if (!vvr.usable)
      {
        test.skip(metric, vvr.skipStatus, vvr.skipReason, o);
        continue;
      }
      auto it = vvr.prefill.find(sseq);
      if (it == vvr.prefill.end())
        continue;
      if (it->second.us > 0.0)
        test.emit(metric,
                  (float)(blockFlops(sseq, sseq) * 1.0e6 / it->second.us), o);
      else
        test.skip(metric, it->second.status, it->second.error, o);
    }
  };


  // ---- Test specs (single header per test, streaming per variant) ----
  const logger::TestSpec prefillFlopsSpec = {
      "onnx_block_prefill", "Transformer block, prefill", "flops",
      Category::Ai,
      "One 2048-wide, 16-head decoder block with a SwiGLU feed-forward "
      "(50.6M parameters), working through a prompt -- the "
      "phase that decides how long you wait for the first word -- at each "
      "precision a model ships in.  Everything a real layer does is in here, "
      "and only the seven projection matmuls change precision, so whatever "
      "separates two rows is the projection format.",
      TestShape::Heterogeneous, "data type and prompt length"};
  const logger::TestSpec prefillOpsSpec = {
      "onnx_block_prefill", "Transformer block, prefill", "ops",
      Category::Ai,
      "One 2048-wide, 16-head decoder block with a SwiGLU feed-forward "
      "(50.6M parameters), working through a prompt -- the "
      "phase that decides how long you wait for the first word -- at each "
      "precision a model ships in.  Everything a real layer does is in here, "
      "and only the seven projection matmuls change precision, so whatever "
      "separates two rows is the projection format.",
      TestShape::Heterogeneous, "data type and prompt length"};
  const logger::TestSpec decodeSpec = {
      "onnx_block_decode", "Transformer block, decode", "bps",
      Category::Ai,
      "How fast one 2048-wide, 16-head decoder block (50.6M parameters) "
      "streams its weights while generating a token with 2048 of context, at "
      "each precision.  Each row counts the bytes "
      "that format actually moves, so it compares directly against the plain "
      "memory-bandwidth rows -- and a narrow-weight row far below the 16-bit "
      "one is a provider unpacking to full width before using them.",
      TestShape::Heterogeneous, "data type"};
  const logger::TestSpec latencySpec = {
      "onnx_block_latency", "Transformer block latency", "s",
      Category::Ai,
      "How long one 2048-wide, 16-head decoder block (50.6M parameters) takes "
      "at each precision: multiply by a model's "
      "layer count for a floor on its time-to-first-token and per-token time "
      "here, without downloading anything.  Everything but attention costs "
      "the same at every context length, so whatever the decode rows add as "
      "the context grows is attention.",
      TestShape::Heterogeneous, "data type, phase and context length"};

  auto provenance = [](const Variant &v, const VariantResult &vr)
  {
    if (vr.ranAs.empty())
      return std::string();
    std::string s = "  Ran as " + vr.ranAs;
    if (vr.schemeName[0])
      s += " (" + std::string(vr.schemeName) + ")";
    s += ".";
    if (vr.castedActs)
      s += "  The provider casts the activations to the width its kernel "
           "wanted, a full pass inside this figure.";
    return s;
  };

  const char *geometry = "";

  auto doVariant = [&](const Variant &v, VariantResult &vr,
                       logger::TestScope &test)
  {
    (void)v; (void)vr; (void)test;
  };
  (void)doVariant;

  // ---- Prefill: flops vs ops split, single header per unit, per-variant ----
  // Measures only the prompt ladder; decode/latency are measured in their
  // own tests below so each test has a single header and streams per
  // variant as that test's variants are timed.
  {
    auto testFlops = currentDeviceScope->beginTest(prefillFlopsSpec);
    for (size_t vi = 0; vi < kNVariants; vi++)
    {
      if (clpeak::cancelRequested()) break;
      const Variant &v = kVariants[vi];
      if (isIntVariant(v)) continue;
      VariantResult &vr = results[vi];

      // Global probe fast-path: if gemm's tiny probe already says this
      // dtype is unsupported/emulated on this EP, skip without paying
      // block's 50M-weight model compilation.
      {
        const auto &probe = onnxProbeGemmCache(rt, ep);
        auto it = probe.find(v.label);
        if (it != probe.end() && !it->second.ok)
        {
          vr.skipReason = it->second.reason;
          emitPrefillTo(testFlops, v, vr);
          continue;
        }
      }
      // dtype / opset
      {
        std::string why = onnxDtypeUnsupportedReason(rt, v.wDtype);
        if (why.empty()) why = onnxDtypeUnsupportedReason(rt, v.actDtype);
        if (why.empty())
        {
          const int opset = variantOpset(v);
          const uint32_t needApi = onnxMinOrtApiForOpset(opset);
          if (needApi && rt.apiVersion < needApi)
            why = "needs opset " + std::to_string(opset) +
                  ", which arrived in ONNX Runtime 1." + std::to_string(needApi) +
                  "; this runtime is " + rt.versionString;
        }
        if (!why.empty()) { vr.skipReason = why; emitPrefillTo(testFlops, v, vr); continue; }
      }
      {
        const uint64_t needed = weightBytes(v) + kvBytes(v, contextsFor(v).back()) + (64ull << 20) * onnxElemBytes(v.actDtype, 1) / 2;
        const uint64_t budget = clpeak::memoryBudget(~0ull, 8);
        if (budget && budget < needed)
        {
          vr.skipReason = "not enough memory for the canonical block; its geometry is fixed so the numbers stay comparable, and a smaller layer would not be the same test";
          CLPEAK_VLOG("onnx-block[%s/%s]: needs %llu MB, budget %llu MB\n", ep.providerKey.c_str(), v.label, (unsigned long long)(needed >> 20), (unsigned long long)(budget >> 20));
          emitPrefillTo(testFlops, v, vr); continue;
        }
      }
      if (v.qdq || v.wBlock > 0 || v.kvDtype)
      {
        std::string tried, firstErr; Scheme schemes[2];
        const size_t nSchemes = v.qdq ? qdqSchemesFor(v, schemes) : 1;
        const bool probeDecode = v.decodeOnly;
        for (size_t si = 0; si < nSchemes; si++)
        {
          if (clpeak::cancelRequested()) break;
          const int qAct = v.qdq ? schemes[si].actDtype : ONNX_DT_INT8;
          const char *what = v.qdq ? schemes[si].name : (v.kvDtype ? "quantized cache" : "weight-only");
          BlockRun probe = makeRun(rt, ep, v, probeDecode, kKvLadder[0], kPromptLadder[0], qAct, true);
          if (!probe.session) { if (firstErr.empty()) firstErr = probe.error; CLPEAK_VLOG("onnx-block[%s/%s]: %s rejected: %s\n", ep.providerKey.c_str(), v.label, what, probe.error.c_str()); continue; }
          timeRuns(rt, probe, 1);
          auto ops = onnxCollectExecutedOps(rt, probe.session); destroyRun(rt, probe);
          std::string joined; for (auto &o: ops) joined += (joined.empty()?"":", ")+o;
          CLPEAK_VLOG("onnx-block[%s/%s]: %s executed %s\n", ep.providerKey.c_str(), v.label, what, joined.c_str());
          if (projectionsRanQuantized(ops))
          {
            vr.qActDtype = qAct; vr.schemeName = v.qdq ? schemes[si].name : "";
            if (!v.qdq) for (auto &o: ops) if (o=="Cast") vr.castedActs = true;
            const std::string named = onnxQuantizedKernelName(ops);
            vr.ranAs = named.empty() ? "a kernel it compiled itself" : named; break;
          }
          if (!ops.empty()) tried = joined;
        }
        if (vr.ranAs.empty())
        {
          vr.skipReason = tried.empty() ? (firstErr.empty()? std::string("this provider accepted no session for ")+v.label : firstErr) : std::string("provider did not fuse a quantized matmul -- it dequantized the ")+(v.kvDtype?"cache":"weights")+" to full width and multiplied in floating point, a complete pass over them on every run, so this is not a "+v.label+" rate (ran: "+tried+")";
          emitPrefillTo(testFlops, v, vr); continue;
        }
      }
      vr.usable = true;
      // affordability uses refPrefillRate which is shared across all variants
      double prefillRate = refPrefillRate;
      auto affordable = [&](double flops, double rate){ return rate<=0.0 || (flops/rate) <= (double)kBlockBudgetUs; };
      for (int64_t seq : promptsFor(v))
      {
        if (clpeak::cancelRequested()) break;
        Point pt; const double flops = blockFlops(seq, seq);
        if (!affordable(flops, prefillRate))
        {
          pt.status = ResultStatus::Error; pt.error = "one pass would take about "+std::to_string((long long)(flops/prefillRate/1.0e6))+" s on this provider, too slow to measure";
          CLPEAK_VLOG("onnx-block[%s/%s]: skipping prefill %lld, %s\n", ep.providerKey.c_str(), v.label, (long long)seq, pt.error.c_str());
        }
        else
        {
          pt.us = measure(rt, ep, v, false, vr.qActDtype, warmupCount, forceIters, specifiedIters, pt.error, pt.status, kDecodeKv, seq);
          if (pt.us > 0.0) prefillRate = refPrefillRate = flops / pt.us;
        }
        vr.prefill[seq] = pt;
      }
      // keep decodeRate warm for later groups (measure decode in its own test below,
      // but seed it from prefill's last rate to avoid cold start)
      if (!vr.prefill.empty())
      {
        // seed decode rate from last prefill point if available
        auto it = vr.prefill.find(kPrefillSeq);
        if (it != vr.prefill.end() && it->second.us > 0.0) refDecodeRate = blockFlops(1, kDecodeKv) / it->second.us;
      }
      emitPrefillTo(testFlops, v, vr);
    }
    testFlops.end();
  }
  {
    auto testOps = currentDeviceScope->beginTest(prefillOpsSpec);
    for (size_t vi = 0; vi < kNVariants; vi++)
    {
      if (clpeak::cancelRequested()) break;
      const Variant &v = kVariants[vi];
      if (!isIntVariant(v)) continue;
      VariantResult &vr = results[vi];
      {
        const auto &probe = onnxProbeGemmCache(rt, ep);
        auto it = probe.find(v.label);
        if (it != probe.end() && !it->second.ok)
        {
          vr.skipReason = it->second.reason;
          emitPrefillTo(testOps, v, vr);
          continue;
        }
      }
      {
        std::string why = onnxDtypeUnsupportedReason(rt, v.wDtype);
        if (why.empty()) why = onnxDtypeUnsupportedReason(rt, v.actDtype);
        if (why.empty())
        {
          const int opset = variantOpset(v);
          const uint32_t needApi = onnxMinOrtApiForOpset(opset);
          if (needApi && rt.apiVersion < needApi)
            why = "needs opset " + std::to_string(opset) + ", which arrived in ONNX Runtime 1." + std::to_string(needApi) + "; this runtime is " + rt.versionString;
        }
        if (!why.empty()) { vr.skipReason = why; emitPrefillTo(testOps, v, vr); continue; }
      }
      {
        const uint64_t needed = weightBytes(v) + kvBytes(v, contextsFor(v).back()) + (64ull << 20) * onnxElemBytes(v.actDtype, 1) / 2;
        const uint64_t budget = clpeak::memoryBudget(~0ull, 8);
        if (budget && budget < needed)
        {
          vr.skipReason = "not enough memory for the canonical block; its geometry is fixed so the numbers stay comparable, and a smaller layer would not be the same test";
          CLPEAK_VLOG("onnx-block[%s/%s]: needs %llu MB, budget %llu MB\n", ep.providerKey.c_str(), v.label, (unsigned long long)(needed >> 20), (unsigned long long)(budget >> 20));
          emitPrefillTo(testOps, v, vr); continue;
        }
      }
      if (v.qdq || v.wBlock > 0 || v.kvDtype)
      {
        std::string tried, firstErr; Scheme schemes[2];
        const size_t nSchemes = v.qdq ? qdqSchemesFor(v, schemes) : 1;
        const bool probeDecode = v.decodeOnly;
        for (size_t si = 0; si < nSchemes; si++)
        {
          if (clpeak::cancelRequested()) break;
          const int qAct = v.qdq ? schemes[si].actDtype : ONNX_DT_INT8;
          const char *what = v.qdq ? schemes[si].name : (v.kvDtype ? "quantized cache" : "weight-only");
          BlockRun probe = makeRun(rt, ep, v, probeDecode, kKvLadder[0], kPromptLadder[0], qAct, true);
          if (!probe.session) { if (firstErr.empty()) firstErr = probe.error; CLPEAK_VLOG("onnx-block[%s/%s]: %s rejected: %s\n", ep.providerKey.c_str(), v.label, what, probe.error.c_str()); continue; }
          timeRuns(rt, probe, 1);
          auto ops = onnxCollectExecutedOps(rt, probe.session); destroyRun(rt, probe);
          std::string joined; for (auto &o: ops) joined += (joined.empty()?"":", ")+o;
          CLPEAK_VLOG("onnx-block[%s/%s]: %s executed %s\n", ep.providerKey.c_str(), v.label, what, joined.c_str());
          if (projectionsRanQuantized(ops))
          {
            vr.qActDtype = qAct; vr.schemeName = v.qdq ? schemes[si].name : "";
            if (!v.qdq) for (auto &o: ops) if (o=="Cast") vr.castedActs = true;
            const std::string named = onnxQuantizedKernelName(ops);
            vr.ranAs = named.empty() ? "a kernel it compiled itself" : named; break;
          }
          if (!ops.empty()) tried = joined;
        }
        if (vr.ranAs.empty())
        {
          vr.skipReason = tried.empty() ? (firstErr.empty()? std::string("this provider accepted no session for ")+v.label : firstErr) : std::string("provider did not fuse a quantized matmul -- it dequantized the ")+(v.kvDtype?"cache":"weights")+" to full width and multiplied in floating point, a complete pass over them on every run, so this is not a "+v.label+" rate (ran: "+tried+")";
          emitPrefillTo(testOps, v, vr); continue;
        }
      }
      vr.usable = true;
      double prefillRate = refPrefillRate;
      auto affordable = [&](double flops, double rate){ return rate<=0.0 || (flops/rate) <= (double)kBlockBudgetUs; };
      for (int64_t seq : promptsFor(v))
      {
        if (clpeak::cancelRequested()) break;
        Point pt; const double flops = blockFlops(seq, seq);
        if (!affordable(flops, prefillRate))
        {
          pt.status = ResultStatus::Error; pt.error = "one pass would take about "+std::to_string((long long)(flops/prefillRate/1.0e6))+" s on this provider, too slow to measure";
          CLPEAK_VLOG("onnx-block[%s/%s]: skipping prefill %lld, %s\n", ep.providerKey.c_str(), v.label, (long long)seq, pt.error.c_str());
        }
        else
        {
          pt.us = measure(rt, ep, v, false, vr.qActDtype, warmupCount, forceIters, specifiedIters, pt.error, pt.status, kDecodeKv, seq);
          if (pt.us > 0.0) prefillRate = refPrefillRate = flops / pt.us;
        }
        vr.prefill[seq] = pt;
      }
      emitPrefillTo(testOps, v, vr);
    }
    testOps.end();
  }

  // ---- Decode: single header, per-variant streaming ---------------------------
  {
    auto test = currentDeviceScope->beginTest(decodeSpec);
    for (size_t vi = 0; vi < kNVariants; vi++)
    {
      if (clpeak::cancelRequested()) break;
      const Variant &v = kVariants[vi];
      VariantResult &vr = results[vi];
      const std::string metric = std::string(v.label) + "_kv" + std::to_string(kDecodeKv);
      const uint64_t wStreamed = streamedWeightBytes(v, fp32As16);
      const uint64_t kvStreamed = streamedKvBytes(v, kDecodeKv, fp32As16);
      const double bytes = (double)(wStreamed + kvStreamed);
      logger::EmitOptions o;
      o.description = std::string(geometry) + v.note + "  One token with 2048 of context: " + std::to_string((unsigned long long)(wStreamed >> 20)) + " MB of weights plus " + std::to_string((unsigned long long)(kvStreamed >> 20)) + " MB of cached context, read in full for one token." + provenance(v, vr) + fp32Note(v);

      // If variant was already determined unusable in prefill phase, reuse skip
      if (!vr.usable && !vr.skipReason.empty())
      {
        test.skip(metric, vr.skipStatus, vr.skipReason, o);
        continue;
      }
      // For variants not yet evaluated (e.g. those skipped only for prefill decodeOnly), ensure fusion/memory checks are done
      // Reuse the same checks as prefill if vr still empty
      if (!vr.usable && vr.skipReason.empty())
      {
        // dtype / memory / fusion already handled in prefill loops for most variants;
        // if we reach here for a decodeOnly variant that was skipped in prefill's int check, handle it
        std::string why = onnxDtypeUnsupportedReason(rt, v.wDtype);
        if (why.empty()) why = onnxDtypeUnsupportedReason(rt, v.actDtype);
        if (why.empty())
        {
          const int opset = variantOpset(v);
          const uint32_t needApi = onnxMinOrtApiForOpset(opset);
          if (needApi && rt.apiVersion < needApi) why = "needs opset " + std::to_string(opset) + ", which arrived in ONNX Runtime 1." + std::to_string(needApi) + "; this runtime is " + rt.versionString;
        }
        if (!why.empty()) { vr.skipReason = why; test.skip(metric, vr.skipStatus, vr.skipReason, o); continue; }
        const uint64_t needed = weightBytes(v) + kvBytes(v, contextsFor(v).back()) + (64ull << 20) * onnxElemBytes(v.actDtype, 1) / 2;
        const uint64_t budget = clpeak::memoryBudget(~0ull, 8);
        if (budget && budget < needed) { vr.skipReason = "not enough memory for the canonical block; its geometry is fixed so the numbers stay comparable, and a smaller layer would not be the same test"; test.skip(metric, vr.skipStatus, vr.skipReason, o); continue; }
        // fusion probe if needed
        if (v.qdq || v.wBlock > 0 || v.kvDtype)
        {
          std::string tried, firstErr; Scheme schemes[2];
          const size_t nSchemes = v.qdq ? qdqSchemesFor(v, schemes) : 1;
          for (size_t si = 0; si < nSchemes; si++)
          {
            if (clpeak::cancelRequested()) break;
            const int qAct = v.qdq ? schemes[si].actDtype : ONNX_DT_INT8;
            const char *what = v.qdq ? schemes[si].name : (v.kvDtype ? "quantized cache" : "weight-only");
            BlockRun probe = makeRun(rt, ep, v, true, kKvLadder[0], kPromptLadder[0], qAct, true);
            if (!probe.session) { if (firstErr.empty()) firstErr = probe.error; continue; }
            timeRuns(rt, probe, 1);
            auto ops = onnxCollectExecutedOps(rt, probe.session); destroyRun(rt, probe);
            std::string joined; for (auto &o: ops) joined += (joined.empty()?"":", ")+o;
            if (projectionsRanQuantized(ops)) { vr.qActDtype = qAct; vr.schemeName = v.qdq ? schemes[si].name : ""; if (!v.qdq) for (auto &o: ops) if (o=="Cast") vr.castedActs = true; const std::string named = onnxQuantizedKernelName(ops); vr.ranAs = named.empty() ? "a kernel it compiled itself" : named; break; }
            if (!ops.empty()) tried = joined;
          }
          if (vr.ranAs.empty()) { vr.skipReason = tried.empty() ? (firstErr.empty()? std::string("this provider accepted no session for ")+v.label : firstErr) : std::string("provider did not fuse a quantized matmul -- it dequantized the ")+(v.kvDtype?"cache":"weights")+" to full width and multiplied in floating point, a complete pass over them on every run, so this is not a "+v.label+" rate (ran: "+tried+")"; test.skip(metric, vr.skipStatus, vr.skipReason, o); continue; }
        }
        vr.usable = true;
      }

      // If decode measurement already exists from prefill phase (when we measured both), reuse it;
      // otherwise measure it now inside decode's own scope so it streams per variant.
      auto it = vr.decode.find(kDecodeKv);
      if (it == vr.decode.end() || it->second.us <= 0.0)
      {
        // Need to measure decode for this variant within decode test so it streams
        double decodeRate = refDecodeRate;
        auto affordable = [&](double flops, double rate){ return rate<=0.0 || (flops/rate) <= (double)kBlockBudgetUs; };
        Point pt; const double flops = blockFlops(1, kDecodeKv);
        if (!affordable(flops, decodeRate))
        {
          pt.status = ResultStatus::Error; pt.error = "one token would take about "+std::to_string((long long)(flops/decodeRate/1.0e6))+" s on this provider, too slow to measure";
          CLPEAK_VLOG("onnx-block[%s/%s]: skipping decode kv%lld, %s\n", ep.providerKey.c_str(), v.label, (long long)kDecodeKv, pt.error.c_str());
        }
        else
        {
          // Ensure fusion etc. already validated; vr.qActDtype is set
          pt.us = measure(rt, ep, v, true, vr.qActDtype, warmupCount, forceIters, specifiedIters, pt.error, pt.status, kDecodeKv, kPrefillSeq);
          if (pt.us > 0.0) decodeRate = refDecodeRate = flops / pt.us;
        }
        vr.decode[kDecodeKv] = pt;
        it = vr.decode.find(kDecodeKv);
      }
      if (it != vr.decode.end())
      {
        if (it->second.us > 0.0) test.emit(metric, (float)(bytes / (it->second.us * 1.0e-6)), o);
        else test.skip(metric, it->second.status, it->second.error, o);
      }
    }
    test.end();
  }

  // ---- Latency: single header, per-variant streaming ------------------------
  {
    auto test = currentDeviceScope->beginTest(latencySpec);
    for (size_t vi = 0; vi < kNVariants; vi++)
    {
      if (clpeak::cancelRequested()) break;
      const Variant &v = kVariants[vi];
      VariantResult &vr = results[vi];
      const std::string prov = provenance(v, vr);

      // Prefill latency (s512)
      if (!v.decodeOnly)
      {
        const std::string metric = std::string(v.label) + "_prefill_s" + std::to_string(kPrefillSeq);
        const std::string note = std::string("One pass over a 512-token prompt.  ") + v.note + prov + fp32Note(v);
        if (!vr.usable && !vr.skipReason.empty()) { test.skip(metric, vr.skipStatus, vr.skipReason, note); }
        else
        {
          // Ensure prefill measurement exists; if not, measure inside latency so it streams
          auto it = vr.prefill.find(kPrefillSeq);
          if (it == vr.prefill.end() || it->second.us <= 0.0)
          {
            if (vr.usable) // already validated
            {
              Point pt; std::string err; ResultStatus st = ResultStatus::Ok;
              double flops = blockFlops(kPrefillSeq, kPrefillSeq);
              // affordability already seeded
              pt.us = measure(rt, ep, v, false, vr.qActDtype, warmupCount, forceIters, specifiedIters, err, st, kDecodeKv, kPrefillSeq);
              pt.error = err; pt.status = st;
              vr.prefill[kPrefillSeq] = pt; it = vr.prefill.find(kPrefillSeq);
            }
          }
          if (it != vr.prefill.end())
          {
            if (it->second.us > 0.0) test.emit(metric, (float)(it->second.us * 1e-6), note.c_str());
            else if (vr.usable) test.skip(metric, it->second.status, it->second.error, note);
            else test.skip(metric, vr.skipStatus, vr.skipReason, note);
          }
        }
      }
      // Decode ladder for latency
      for (int64_t kv : contextsFor(v))
      {
        const std::string metric = std::string(v.label) + "_decode_kv" + std::to_string(kv);
        const std::string note = "One generated token with " + std::to_string(kv) + " tokens of context behind it.  " + v.note + prov + fp32Note(v);
        if (!vr.usable && !vr.skipReason.empty()) { test.skip(metric, vr.skipStatus, vr.skipReason, note); continue; }
        auto it = vr.decode.find(kv);
        if (it == vr.decode.end() || it->second.us <= 0.0)
        {
          if (vr.usable)
          {
            Point pt; const double flops = blockFlops(1, kv);
            double decodeRate = refDecodeRate;
            auto affordable = [&](double f, double r){ return r<=0.0 || (f/r) <= (double)kBlockBudgetUs; };
            if (!affordable(flops, decodeRate))
            {
              pt.status = ResultStatus::Error; pt.error = "one token would take about "+std::to_string((long long)(flops/decodeRate/1.0e6))+" s on this provider, too slow to measure";
            }
            else
            {
              pt.us = measure(rt, ep, v, true, vr.qActDtype, warmupCount, forceIters, specifiedIters, pt.error, pt.status, kv, kPrefillSeq);
              if (pt.us > 0.0) decodeRate = refDecodeRate = flops / pt.us;
            }
            vr.decode[kv] = pt; it = vr.decode.find(kv);
          }
        }
        if (it == vr.decode.end()) continue;
        if (it->second.us > 0.0) test.emit(metric, (float)(it->second.us * 1e-6), note.c_str());
        else { test.skip(metric, it->second.status, it->second.error.empty() ? "run failed" : it->second.error, note); break; }
      }
    }
    test.end();
  }


  return 0;
}

#endif // ENABLE_ONNX
