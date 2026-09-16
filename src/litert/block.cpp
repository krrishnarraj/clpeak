#ifdef ENABLE_LITERT

// litert-block: one fixed transformer decoder block through LiteRT, run in
// the two regimes that bound all LLM inference, at each format a language
// model ships in -- the rung above a raw matmul peak and below tokens per
// second, on the accelerator this device row names.
//
//   prefill (64/512/2048 tokens at once)      compute-bound -> effective FLOPS
//   decode  (1 token, 512/2048/8192 context)  memory-bound  -> effective B/s
//
// Geometry, ladders, formats and reporting are the ONNX backend's
// (src/onnx/block.cpp), so a LiteRT row divides by an ONNX or Core ML one:
// the block is 2048 wide, 16 heads of 128, a 5504-wide SwiGLU feed-forward
// -- 50.6M parameters, 101 MB at fp16.  Attention is explicit batched
// matmul / softmax / matmul, the primitive spelling every runtime accepts;
// the composite-op spelling LiteRT-LM ships (odml.scaled_dot_product_
// attention and friends) is a later variant.
//
// The block is where a format's real answer lives.  A single matmul with
// resident activations can be packed once and run at its kernel's rate; a
// layer's activations are live and its seven projections interleave with
// attention, the softmax and the SwiGLU, and whether an accelerator keeps
// its matmul rate through all of that is what a model would meet.

#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace
{

constexpr int64_t kDModel = 2048;
constexpr int64_t kHeads = 16;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kFfnHidden = 5504;
constexpr int64_t kPrefillSeq = 512;
constexpr int64_t kDecodeKv = 2048;
const int64_t kKvLadder[] = {512, 2048, 8192};
const int64_t kPromptLadder[] = {64, 512, 2048};
constexpr unsigned int kBlockBudgetUs = 5000000;

struct Variant
{
  const char *label;
  LitertFormat f;
  bool sweep;         // walk the whole prompt and context ladders
  const char *unit;   // nullptr: the scope's own unit
  const char *note;
  bool int8Kv;        // the cache stored as int8
  bool decodeOnly;    // a cache format has no prefill row
};

const Variant kVariants[] = {
    {"fp16", LitertFormat::Fp16, true, nullptr,
     "16-bit weights and 16-bit arithmetic, the form an unquantized model is "
     "served in and the reference the other rows are read against.",
     false, false},
    {"int4_weight", LitertFormat::Int4Weight, true, nullptr,
     "4-bit weights, one scale per 32, against float activations -- what a "
     "quantized language model ships as.  XNNPACK runs the projections as int8 "
     "arithmetic on dynamically quantized activations; an NPU does whatever its "
     "compiler makes of the blocked form with live activations, which is the "
     "finding.",
     false, false},
    {"int8_weight", LitertFormat::Int8Weight, false, nullptr,
     "8-bit weights, one scale per output row, against float activations -- "
     "dynamic-range quantization, the format a post-training-quantized model "
     "ships in.",
     false, false},
    {"int8_qdq", LitertFormat::Int8Qdq, false, "ops",
     "8-bit weights and 8-bit arithmetic through the projections, quantized in "
     "and out -- what headline TOPS figures are quoted for, measured on a whole "
     "layer.  Attention and the softmax stay in float, as they do in every "
     "real deployment.",
     false, false},
    {"fp32", LitertFormat::Fp32, false, nullptr,
     "Full precision, which nobody serves a language model in, here as a "
     "control: an accelerator whose fp16 row fails to beat it is not running "
     "half-precision hardware.",
     false, false},
    {"bf16", LitertFormat::Bf16, false, nullptr,
     "bfloat16 tensors throughout; a layer needs bf16 kernels for every "
     "operation in it, and the row records what LiteRT says to that.",
     false, false},
    {"fp8_weight", LitertFormat::Fp8Weight, false, nullptr,
     "8-bit float weights with a scale per row, if any kernel takes them.",
     false, false},
    {"int8_kv", LitertFormat::Fp16, true, nullptr,
     "16-bit throughout with only the cached context stored as 8-bit integers "
     "and decompressed into attention -- the axis that decides how long a "
     "conversation a device can hold, which is why this row sweeps context.",
     true, true},
};

uint64_t weightBytes(const LitertPlan &p)
{
  return 4 * litertWeightBytes(p, kDModel, kDModel) + 2 * litertWeightBytes(p, kFfnHidden, kDModel) +
         litertWeightBytes(p, kDModel, kFfnHidden);
}

// The cache as the model stores it: int8 when asked, else the float
// constant type of the block's float parts (the quantized block keeps
// attention in float).
uint64_t kvBytes(const LitertPlan &p, bool int8Kv, int64_t kv)
{
  LitertPlan fp = p;
  if (fp.act == clpeak_tflite::TfType::I8)
    fp.act = clpeak_tflite::TfType::F32;
  const uint64_t elem = int8Kv ? 1 : litertElemBytes(litertConstantType(fp), 1);
  return 2ull * (uint64_t)kHeads * (uint64_t)kv * (uint64_t)kHeadDim * elem;
}

std::vector<int64_t> promptsFor(const Variant &v)
{
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

double blockFlops(int64_t seq, int64_t ctx)
{
  const double S = (double)seq, C = (double)ctx;
  const double d = (double)kDModel, ffn = (double)kFfnHidden;
  const double H = (double)kHeads, Dh = (double)kHeadDim;
  const double qkv = 3.0 * 2.0 * S * d * d;
  const double attn = 2.0 * 2.0 * H * S * C * Dh;
  const double proj = 2.0 * S * d * d;
  const double ff = 2.0 * 2.0 * S * d * ffn + 2.0 * S * ffn * d;
  return qkv + attn + proj + ff;
}

struct Point
{
  double us = -1.0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

struct VariantResult
{
  bool usable = false;
  std::string skipReason;
  ResultStatus skipStatus = ResultStatus::Unsupported;
  std::map<int64_t, Point> prefill, decode;
};

LitertBlockShape shapeFor(const Variant &v, bool decode, int64_t kvLen, int64_t prefillSeq)
{
  LitertBlockShape sh;
  sh.dModel = kDModel;
  sh.heads = kHeads;
  sh.headDim = kHeadDim;
  sh.ffnHidden = kFfnHidden;
  sh.seq = decode ? 1 : prefillSeq;
  sh.kvLen = decode ? kvLen : 0;
  sh.int8Kv = v.int8Kv;
  return sh;
}

} // namespace

int LitertPeak::runBlock(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;
  constexpr size_t kNVariants = sizeof(kVariants) / sizeof(kVariants[0]);
  std::vector<VariantResult> results(kNVariants);
  std::vector<LitertPlan> plans;
  for (const Variant &v : kVariants)
    plans.push_back(litertPlanFor(v.f, dev.accel));

  // Time one regime end to end: mean us per block, or negative with the
  // point's status and error set.
  auto measure = [&](const Variant &v, const LitertPlan &plan, bool decode, int64_t kvLen,
                     int64_t prefillSeq, Point &pt)
  {
    std::string err;
    auto s = LitertSession::create(rt, dev, litertBlockModel(plan, shapeFor(v, decode, kvLen, prefillSeq)),
                                   litertConfigFor(plan), err);
    const std::string what = decode ? "decode_kv" + std::to_string(kvLen)
                                    : "prefill_s" + std::to_string(prefillSeq);
    if (!s)
    {
      CLPEAK_VLOG("litert-block[%s/%s]: %s create failed: %s\n", dev.displayName.c_str(), v.label,
                  what.c_str(), err.c_str());
      pt.error = err;
      pt.status = ResultStatus::Unsupported;
      return;
    }
    CLPEAK_VLOG("litert-block[%s/%s]: %s create %.2f s\n", dev.displayName.c_str(), v.label, what.c_str(),
                s->createUs / 1.0e6);
    if (!s->onDevice())
    {
      pt.error = s->offDevice();
      pt.status = ResultStatus::Unsupported;
      return;
    }
    if (s->createUs > kLitertMaxBlockCreateUs)
    {
      pt.error = "model creation took " + std::to_string((long long)(s->createUs / 1.0e6)) + " s, exceeds " +
                 std::to_string((long long)(kLitertMaxBlockCreateUs / 1.0e6)) + " s compilation budget";
      pt.status = ResultStatus::Unsupported;
      return;
    }
    LitertPlan sp = plan;
    if (sp.act == clpeak_tflite::TfType::I8)
      sp.act = clpeak_tflite::TfType::F32;   // the block's float parts carry the scalar
    if (!litertBindScalar(*s, sp, err))
    {
      pt.error = err;
      pt.status = ResultStatus::Error;
      return;
    }
    auto m = litertMeasure(*s, warmupCount, kBlockBudgetUs, forceIters, specifiedIters);
    if (m.meanUs <= 0.0)
    {
      pt.error = m.error.empty() ? "run failed" : m.error;
      pt.status = m.status;
      return;
    }
    pt.us = m.meanUs;
    pt.status = ResultStatus::Ok;
  };

  // Everything that has to be true before a variant is worth timing: the
  // format applies here, the fixed geometry fits, and one small session
  // compiles and lands on this accelerator.
  auto validateVariant = [&](size_t vi) -> bool
  {
    const Variant &v = kVariants[vi];
    const LitertPlan &plan = plans[vi];
    VariantResult &vr = results[vi];
    if (vr.usable || !vr.skipReason.empty())
      return vr.usable;
    if (!plan.applies)
    {
      vr.skipReason = plan.whyNot;
      return false;
    }
    if (v.int8Kv && dev.accel != LitertAccel::Npu)
    {
      // XNNPACK dequantizes a constant int8 tensor once, when the model
      // loads, and the GPU accelerator converts constant weights at load
      // the same way: by the time attention reads the cache it is full
      // width, and the row would time a float cache under an int8 label.
      // Only a compiler that consumes the int8 cache inside attention can
      // make the row real, so it is only asked of the NPU.
      vr.skipReason = std::string("the ") + litertAccelName(dev.accel) +
                      " decompresses a constant int8 tensor when the model loads, so the cache would "
                      "be full width by the time attention read it -- a float cache under an int8 "
                      "label; only an NPU compiler can consume it as int8";
      return false;
    }
    {
      const uint64_t acts = (64ull << 20) * litertElemBytes(plan.act, 1) / 2;
      const uint64_t needed = 2 * weightBytes(plan) + kvBytes(plan, v.int8Kv, contextsFor(v).back()) + acts;
      const uint64_t budget = clpeak::memoryBudget(~0ull, 8);
      if (budget && budget < needed)
      {
        vr.skipReason = "not enough memory for the canonical block; its geometry is fixed so the "
                        "numbers stay comparable, and a smaller layer would not be the same test";
        return false;
      }
    }
    Point pt;
    const bool decode = v.decodeOnly;
    measure(v, plan, decode, kKvLadder[0], kPromptLadder[0], pt);
    if (pt.us <= 0.0)
    {
      vr.skipReason = pt.error.empty() ? std::string("the block could not be built for ") + v.label : pt.error;
      vr.skipStatus = pt.status;
      return false;
    }
    if (decode)
      vr.decode[kKvLadder[0]] = pt;
    else
      vr.prefill[kPromptLadder[0]] = pt;
    vr.usable = true;
    return true;
  };

  double refPrefillRate = 0.0, refDecodeRate = 0.0;
  auto affordable = [&](double flops, double rate) { return rate <= 0.0 || (flops / rate) <= (double)kBlockBudgetUs; };

  auto measurePrefill = [&](size_t vi, int64_t seq)
  {
    VariantResult &vr = results[vi];
    if (vr.prefill.count(seq))
      return;
    Point pt;
    const double flops = blockFlops(seq, seq);
    if (!affordable(flops, refPrefillRate))
    {
      pt.status = ResultStatus::Error;
      pt.error = "one pass would take about " + std::to_string((long long)(flops / refPrefillRate / 1.0e6)) +
                 " s on this accelerator, too slow to measure";
    }
    else
    {
      measure(kVariants[vi], plans[vi], false, kDecodeKv, seq, pt);
      if (pt.us > 0.0)
        refPrefillRate = flops / pt.us;
    }
    vr.prefill[seq] = pt;
  };

  auto measureDecode = [&](size_t vi, int64_t kv)
  {
    VariantResult &vr = results[vi];
    if (vr.decode.count(kv))
      return;
    Point pt;
    const double flops = blockFlops(1, kv);
    if (!affordable(flops, refDecodeRate))
    {
      pt.status = ResultStatus::Error;
      pt.error = "one token would take about " + std::to_string((long long)(flops / refDecodeRate / 1.0e6)) +
                 " s on this accelerator, too slow to measure";
    }
    else
    {
      measure(kVariants[vi], plans[vi], true, kv, kPrefillSeq, pt);
      if (pt.us > 0.0)
        refDecodeRate = flops / pt.us;
    }
    vr.decode[kv] = pt;
  };

  auto emitPrefillTo = [&](logger::TestScope &test, size_t vi)
  {
    const Variant &v = kVariants[vi];
    const VariantResult &vr = results[vi];
    for (int64_t seq : promptsFor(v))
    {
      const std::string metric = std::string(v.label) + "_s" + std::to_string(seq);
      logger::EmitOptions o;
      o.description = std::string(v.note) + "  A prompt of " + std::to_string(seq) +
                      " tokens in one pass, counting every multiply in the layer.";
      if (v.unit)
        o.unit = v.unit;
      if (!vr.usable)
      {
        test.skip(metric, vr.skipStatus, vr.skipReason, o);
        continue;
      }
      auto it = vr.prefill.find(seq);
      if (it == vr.prefill.end())
        continue;
      if (it->second.us > 0.0)
        test.emit(metric, (float)(blockFlops(seq, seq) * 1.0e6 / it->second.us), o);
      else
        test.skip(metric, it->second.status, it->second.error, o);
    }
  };

  const std::string geometry =
      "One 2048-wide, 16-head decoder block with a SwiGLU feed-forward (50.6M parameters)";

  const logger::TestSpec prefillSpec = {
      "litert_block_prefill", "Transformer block, prefill", "flops", Category::Ai,
      geometry + ", working through a prompt on this accelerator -- the phase that decides "
      "how long you wait for the first word -- at each format a model ships in, with attention "
      "as explicit batched matmul, softmax and matmul.  Only the seven projection matmuls "
      "change format, so whatever separates two rows is the projection format; and LiteRT's "
      "own answer proves every operation ran here.",
      TestShape::Heterogeneous, "format and prompt length"};
  const logger::TestSpec prefillOpsSpec = {
      "litert_block_prefill", "Transformer block, prefill", "ops", Category::Ai,
      prefillSpec.description, TestShape::Heterogeneous, "format and prompt length"};
  const logger::TestSpec decodeSpec = {
      "litert_block_decode", "Transformer block, decode", "bps", Category::Ai,
      "How fast " + geometry + " streams its weights on this accelerator while generating a "
      "token with 2048 of context, at each format.  Each row counts the bytes that format "
      "actually moves, so it compares directly against the resident-weight bandwidth rows -- "
      "and a narrow-weight row far below the 16-bit one is an accelerator unpacking to full "
      "width before using them, or one whose unpack costs more per byte than the byte saved.",
      TestShape::Heterogeneous, "format"};
  const logger::TestSpec latencySpec = {
      "litert_block_latency", "Transformer block latency", "s", Category::Ai,
      "How long " + geometry + " takes on this accelerator at each format: multiply by a "
      "model's layer count for a floor on its time-to-first-token and per-token time here.  "
      "Everything but attention costs the same at every context length, so whatever the decode "
      "rows add as the context grows is attention.",
      TestShape::Heterogeneous, "format, phase and context length"};

  // ---- Prefill, flops ------------------------------------------------------
  {
    auto test = currentDeviceScope->beginTest(prefillSpec);
    for (size_t vi = 0; vi < kNVariants; vi++)
    {
      if (clpeak::cancelRequested())
        break;
      const Variant &v = kVariants[vi];
      if (v.unit || v.decodeOnly)
        continue;
      if (!validateVariant(vi))
      {
        emitPrefillTo(test, vi);
        continue;
      }
      for (int64_t seq : promptsFor(v))
      {
        if (clpeak::cancelRequested())
          break;
        measurePrefill(vi, seq);
      }
      VariantResult &vr = results[vi];
      if (auto it = vr.prefill.find(kPrefillSeq); it != vr.prefill.end() && it->second.us > 0.0)
        refDecodeRate = blockFlops(kPrefillSeq, kPrefillSeq) / it->second.us;
      emitPrefillTo(test, vi);
    }
    test.end();
  }
  // ---- Prefill, ops --------------------------------------------------------
  {
    auto test = currentDeviceScope->beginTest(prefillOpsSpec);
    for (size_t vi = 0; vi < kNVariants; vi++)
    {
      if (clpeak::cancelRequested())
        break;
      const Variant &v = kVariants[vi];
      if (!v.unit || v.decodeOnly)
        continue;
      if (!validateVariant(vi))
      {
        emitPrefillTo(test, vi);
        continue;
      }
      for (int64_t seq : promptsFor(v))
      {
        if (clpeak::cancelRequested())
          break;
        measurePrefill(vi, seq);
      }
      emitPrefillTo(test, vi);
    }
    test.end();
  }

  // ---- Decode --------------------------------------------------------------
  {
    auto test = currentDeviceScope->beginTest(decodeSpec);
    for (size_t vi = 0; vi < kNVariants; vi++)
    {
      if (clpeak::cancelRequested())
        break;
      const Variant &v = kVariants[vi];
      const LitertPlan &plan = plans[vi];
      VariantResult &vr = results[vi];
      const std::string metric = std::string(v.label) + "_kv" + std::to_string(kDecodeKv);
      const uint64_t wBytes = weightBytes(plan);
      const uint64_t kvB = kvBytes(plan, v.int8Kv, kDecodeKv);
      logger::EmitOptions o;
      o.description = std::string(v.note) + "  One token with 2048 of context: " +
                      std::to_string((unsigned long long)(wBytes >> 20)) + " MB of weights plus " +
                      std::to_string((unsigned long long)(kvB >> 20)) +
                      " MB of cached context, read in full for one token.";
      if (!validateVariant(vi))
      {
        test.skip(metric, vr.skipStatus, vr.skipReason, o);
        continue;
      }
      measureDecode(vi, kDecodeKv);
      const Point &pt = vr.decode[kDecodeKv];
      if (pt.us > 0.0)
        test.emit(metric, (float)((double)(wBytes + kvB) / (pt.us * 1.0e-6)), o);
      else
        test.skip(metric, pt.status, pt.error, o);
    }
    test.end();
  }

  // ---- Latency -------------------------------------------------------------
  {
    auto test = currentDeviceScope->beginTest(latencySpec);
    for (size_t vi = 0; vi < kNVariants; vi++)
    {
      if (clpeak::cancelRequested())
        break;
      const Variant &v = kVariants[vi];
      VariantResult &vr = results[vi];
      if (!v.decodeOnly)
      {
        const std::string metric = std::string(v.label) + "_prefill_s" + std::to_string(kPrefillSeq);
        const std::string note = std::string("One pass over a 512-token prompt.  ") + v.note;
        if (!vr.usable)
          test.skip(metric, vr.skipStatus, vr.skipReason, note);
        else
        {
          measurePrefill(vi, kPrefillSeq);
          const Point &pt = vr.prefill[kPrefillSeq];
          if (pt.us > 0.0)
            test.emit(metric, (float)(pt.us * 1e-6), note.c_str());
          else
            test.skip(metric, pt.status, pt.error, note);
        }
      }
      for (int64_t kv : contextsFor(v))
      {
        if (clpeak::cancelRequested())
          break;
        const std::string metric = std::string(v.label) + "_decode_kv" + std::to_string(kv);
        const std::string note = "One generated token with " + std::to_string(kv) +
                                 " tokens of context behind it.  " + v.note;
        if (!vr.usable)
        {
          test.skip(metric, vr.skipStatus, vr.skipReason, note);
          continue;
        }
        measureDecode(vi, kv);
        const Point &pt = vr.decode[kv];
        if (pt.us > 0.0)
          test.emit(metric, (float)(pt.us * 1e-6), note.c_str());
        else
          test.skip(metric, pt.status, pt.error, note);
      }
    }
    test.end();
  }

  return 0;
}

#endif // ENABLE_LITERT
