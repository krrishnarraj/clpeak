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

#include <common/form_race.h>
#include <common/units.h>
#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <cmath>
#include <cstdio>
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

// The width of the seed a prompt longer than it enters through
// (LitertBlockShape::seedWidth), the ONNX block's kSeedWidth.  A prompt no
// longer than the seed, and decode's one row, gain nothing from one: the
// seed's widening weights would be as large as the input they replace.  The
// 64-token prompt keeps the plain input for its magnitudes too -- seeded, its
// largest value would grow from 2073 to 3389 (litertBlockModel).
constexpr int64_t kSeedWidth = 64;

struct Variant
{
  const char *label;
  LitertFormat f;
  bool sweep;         // walk the whole prompt and context ladders
  const char *unit;   // "ops" for full-integer, the scope it runs in; litertRateUnit has the row's
  const char *note;
  bool int8Kv;        // the cache stored as int8
  bool decodeOnly;    // a cache format has no prefill row
  bool composite;     // attention as the odml.scaled_dot_product_attention composite
};

const Variant kVariants[] = {
    {"fp16", LitertFormat::Fp16, true, nullptr,
     "16-bit weights and arithmetic, the form an unquantized model is served "
     "in and the reference for the other rows.",
     false, false, false},
    {"fp16_composite", LitertFormat::Fp16, false, nullptr,
     "fp16 with attention spelt as the odml.scaled_dot_product_attention "
     "composite that AI Edge Torch and LiteRT-LM ship; an accelerator either "
     "fuses it into one kernel or hands it back, and the guard says which.",
     false, false, true},
    {"int4_weight", LitertFormat::Int4Weight, true, nullptr,
     "4-bit blockwise weights against float activations, what a quantized "
     "language model ships as.",
     false, false, false},
    {"int8_weight", LitertFormat::Int8Weight, false, nullptr,
     "8-bit per-row weights against float activations -- dynamic-range "
     "quantization, the post-training-quantized format.",
     false, false, false},
    {"int8_qdq", LitertFormat::Int8Qdq, false, "ops",
     "8-bit weights and activations through the seven projections, quantized "
     "in and out, with attention and the softmax in float as in any real "
     "deployment.",
     false, false, false},
    {"fp32", LitertFormat::Fp32, false, nullptr,
     "Full precision as a control: an accelerator whose fp16 row fails to beat "
     "it is not running half-precision hardware.",
     false, false, false},
    {"bf16", LitertFormat::Bf16, false, nullptr,
     "bfloat16 throughout, which needs a bf16 kernel for every operation in "
     "the layer; the row records LiteRT's answer.",
     false, false, false},
    {"fp8_weight", LitertFormat::Fp8Weight, false, nullptr,
     "8-bit float weights with a scale per row, if any kernel takes them.",
     false, false, false},
    {"int8_kv", LitertFormat::Fp16, true, nullptr,
     "16-bit throughout with only the cached context stored as int8 and "
     "decompressed into attention -- the axis that decides how long a "
     "conversation a device can hold.",
     true, true, false},
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
  double createUs = 0.0;
  // Where the variant races forms (VariantResult::forms): the one this
  // point's time is, and the other's time at the same point when it ran.
  size_t form = 0;
  double otherUs = -1.0;
  bool nonFinite = false;   // the timed graph returned NaN or infinity
  // The kernel the projections ran as in each of those forms, for a format
  // whose weights are integer (litertRateUnit, litertKernelNote): empty
  // until profiled, and where a profile names none.
  bool profiled = false;
  std::string kernel, otherKernel;
};

struct VariantResult
{
  bool usable = false;
  std::string skipReason;
  ResultStatus skipStatus = ResultStatus::Unsupported;
  std::map<int64_t, Point> prefill, decode;
  // The forms the variant runs in (LitertForm): on a GPU with 8-bit kernels
  // for the format, with them disallowed and allowed, each regime racing the
  // two as gemm.cpp does; one form everywhere else.
  std::vector<LitertForm> forms;
  clpeak::MultiFormRace prefillRace{1}, decodeRace{1};
};

// Why a variant's decode form cannot be sent to this accelerator, or empty.
// The GPU accelerator's graph reader CHECK-fails -- an abort, not a refusal
// (object_reader.cc, "CanReadValue(node_input_index)", LiteRT 2.2.0) -- on a
// composite whose input is a constant, and decode's cache is one; prefill,
// whose K and V are computed, is fine.
std::string decodeFence(const Variant &v, LitertAccel accel)
{
  if (v.composite && accel == LitertAccel::Gpu)
    return "the GPU accelerator aborts the process on a composite with a constant input "
           "(LiteRT 2.2.0), and decode's cache is a constant, so only prefill is sent to it";
  return std::string();
}

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
  sh.composite = v.composite;
  sh.seedWidth = sh.seq > kSeedWidth ? kSeedWidth : 0;
  return sh;
}

// What a prompt row says about its way in, when it has a seed (kSeedWidth).
std::string seedNote(int64_t seq)
{
  if (seq <= kSeedWidth)
    return std::string();
  return "  The prompt starts from a " + std::to_string(kSeedWidth) +
         "-wide seed that takes a runtime value, and one more multiply, not "
         "counted, widens it into the layer's input -- inside the figure, "
         "a quarter of a percent of its arithmetic.";
}

} // namespace

int LitertPeak::runBlock(const LitertRuntime &rt, const litert_device_info_t &dev, benchmark_config_t &cfg)
{
  // The longest one pass may be predicted to take: on a GPU, --max-time-gpu
  // (gpuRunCapUs); elsewhere 0, unbounded.
  const double runCapUs = gpuRunCapUs(dev.deviceType, cfg);
  constexpr size_t kNVariants = sizeof(kVariants) / sizeof(kVariants[0]);
  std::vector<VariantResult> results(kNVariants);
  std::vector<LitertPlan> plans;
  for (const Variant &v : kVariants)
    plans.push_back(litertPlanFor(v.f, dev.accel));

  // Time one regime end to end: mean us per block, or negative with the
  // point's status and error set.
  auto measure = [&](const Variant &v, const LitertPlan &plan, const LitertForm &form, bool decode,
                     int64_t kvLen, int64_t prefillSeq, Point &pt)
  {
    std::string err;
    auto s = LitertSession::create(rt, dev, litertBlockModel(plan, shapeFor(v, decode, kvLen, prefillSeq)),
                                   litertConfigFor(plan, form), err);
    const std::string what = (decode ? "decode_kv" + std::to_string(kvLen) : "prefill_s" + std::to_string(prefillSeq)) +
                             (plan.gpuInt8KernelChoice
                                  ? std::string(form.gpuInt8Kernels ? " with" : " without") + " 8-bit kernels"
                                  : std::string());
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
    pt.createUs = s->createUs;
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
    pt.createUs += s->firstRunUs;   // the race's build cost, as in gemm.cpp
    if (m.meanUs <= 0.0)
    {
      pt.error = m.error.empty() ? "run failed" : m.error;
      pt.status = m.status;
      return;
    }
    // What the timed runs computed, in the float parts' type the block
    // returns its row in.  A wrong answer withholds this point alone: each
    // point is its own graph and its own row.
    const std::string wrong = litertNonFiniteReason(
        *s, sp.act,
        decode ? "for one token against " + std::to_string(kvLen) + " of context"
               : "for a " + std::to_string(prefillSeq) + "-token prompt");
    if (!wrong.empty())
    {
      CLPEAK_VLOG("litert-block[%s/%s]: %s\n", dev.displayName.c_str(), v.label, wrong.c_str());
      pt.error = wrong;
      pt.status = ResultStatus::Error;
      pt.nonFinite = true;
      return;
    }
    pt.us = m.meanUs;
    pt.status = ResultStatus::Ok;
  };

  // One point in every form the regime's race still runs: the faster one's
  // time -- a tie keeps the first unless the other was ready to run twice
  // as fast, FormRace's rule -- and the other's beside it.  A form that
  // cannot run a point leaves the race; the point fails only when no form
  // runs it.  No point settles the race for the next: the GPU uses its 8-bit
  // kernels on large layers only, so the 64-token point that validates a
  // variant says nothing of the 512-token one, and an int8 variant has one
  // prefill and one decode point to report anyway.
  auto measureRaced = [&](size_t vi, bool decode, int64_t kvLen, int64_t prefillSeq, Point &pt)
  {
    VariantResult &vr = results[vi];
    clpeak::MultiFormRace &race = decode ? vr.decodeRace : vr.prefillRace;
    const size_t nf = vr.forms.size();
    std::vector<Point> pts(nf);
    std::vector<double> rate(nf, 0.0), build(nf, 0.0);
    for (size_t f = 0; f < nf; f++)
    {
      if (!race.runs(f) || clpeak::cancelRequested())
        continue;
      measure(kVariants[vi], plans[vi], vr.forms[f], decode, kvLen, prefillSeq, pts[f]);
      if (pts[f].us > 0.0)
      {
        rate[f] = 1.0 / pts[f].us;
        build[f] = pts[f].createUs;
      }
      else
        race.drop(f);
    }
    // A NaN from either form withholds the point, as in gemm.cpp: the other
    // form is no alibi for the accelerator.
    for (size_t f = 0; f < nf; f++)
      if (pts[f].nonFinite)
      {
        pt = pts[f];
        return;
      }
    size_t w = nf, failed = nf;
    for (size_t f = 0; f < nf; f++)
    {
      if (rate[f] > 0.0 && (w == nf || rate[f] > rate[w] * clpeak::kFormRaceTie ||
                            (rate[f] * clpeak::kFormRaceTie >= rate[w] && build[w] >= build[f] * clpeak::kFormRaceBuildGap)))
        w = f;
      if (failed == nf && !pts[f].error.empty())
        failed = f;
    }
    if (w == nf)
    {
      if (failed != nf)
        pt = pts[failed];
      else
      {
        pt.status = ResultStatus::Error;
        pt.error = "no form of the block ran";
      }
      return;
    }
    pt = pts[w];
    pt.form = w;
    for (size_t f = 0; f < nf; f++)
      if (f != w && rate[f] > 0.0)
        pt.otherUs = pts[f].us;
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
    // The forms it runs in, each gated on the format's answer in that form
    // with float inputs and outputs -- the block's own boundary: its int8
    // projections quantize and dequantize inside the graph, so an
    // accelerator that cannot return an int8 tensor still runs it.
    vr.forms.clear();
    std::string firstWrong;
    for (int k = 0; k <= (plan.gpuInt8KernelChoice ? 1 : 0); k++)
    {
      const LitertForm form{false, k == 1, true};
      const std::string wrong = wrongAnswer(rt, dev, v.f, form);
      if (wrong.empty())
        vr.forms.push_back(form);
      else if (firstWrong.empty())
        firstWrong = wrong;
    }
    if (vr.forms.empty())
    {
      vr.skipReason = firstWrong;
      vr.skipStatus = ResultStatus::Error;
      return false;
    }
    vr.prefillRace = clpeak::MultiFormRace(vr.forms.size());
    vr.decodeRace = clpeak::MultiFormRace(vr.forms.size());
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
    measureRaced(vi, decode, kKvLadder[0], kPromptLadder[0], pt);
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
  // Why a point whose one pass is `flops` at `rate` (flops per microsecond,
  // from the last point timed; 0 before any) is not worth timing, or empty
  // when it is: past the whole budget for measuring it, or on a GPU past what
  // one run may keep the device busy.  `what` is "one pass" or "one token".
  auto tooSlow = [&](double flops, double rate, const char *what) -> std::string
  {
    if (rate <= 0.0)
      return std::string();
    const double us = flops / rate;
    if (us > (double)kBlockBudgetUs)
      return std::string(what) + " would take about " + std::to_string((long long)(us / 1.0e6)) +
             " s on this accelerator, too slow to measure";
    if (runCapUs > 0.0 && us > runCapUs)
    {
      char buf[32];
      std::snprintf(buf, sizeof buf, "%.1f", us / 1.0e6);
      return std::string(what) + " would keep this GPU busy for about " + buf +
             " s, longer than --max-time-gpu lets one run hold it -- a driver may reset a GPU held longer";
    }
    return std::string();
  };

  auto measurePrefill = [&](size_t vi, int64_t seq)
  {
    VariantResult &vr = results[vi];
    if (vr.prefill.count(seq))
      return;
    Point pt;
    const double flops = blockFlops(seq, seq);
    if (std::string slow = tooSlow(flops, refPrefillRate, "one pass"); !slow.empty())
    {
      pt.status = ResultStatus::Error;
      pt.error = slow;
    }
    else
    {
      measureRaced(vi, false, kDecodeKv, seq, pt);
      if (pt.us > 0.0)
        refPrefillRate = flops / pt.us;
    }
    vr.prefill[seq] = pt;
  };

  // The kernel the projections ran as at a prompt length, in each form the
  // point timed, from a profiled session of its own (gemm.cpp: profiling
  // never touches a timed one), for a format whose weights are integer: it
  // says whether they were multiplied as integers -- the unit of a
  // weight-only row and of the other form's reading beside it, and a
  // full-integer row's word on a float kernel (litertRateUnit,
  // litertKernelNote).  A profile that names no kernel whose arithmetic can
  // be read ends the profiling on this accelerator: the next would not
  // either, and on an NPU each is another compile.
  bool profileReadable = true;
  auto profilePrefill = [&](size_t vi, int64_t seq)
  {
    const Variant &v = kVariants[vi];
    const LitertPlan &plan = plans[vi];
    VariantResult &vr = results[vi];
    auto it = vr.prefill.find(seq);
    if (!litertIntegerWeights(plan) || it == vr.prefill.end() || it->second.us <= 0.0 || it->second.profiled)
      return;
    Point &pt = it->second;
    pt.profiled = true;
    auto kernelOf = [&](size_t f) -> std::string
    {
      if (!profileReadable || clpeak::cancelRequested())
        return std::string();
      std::string err;
      auto ps = LitertSession::create(rt, dev, litertBlockModel(plan, shapeFor(v, false, kDecodeKv, seq)),
                                      litertConfigFor(plan, vr.forms[f], true), err);
      LitertPlan sp = plan;
      if (sp.act == clpeak_tflite::TfType::I8)
        sp.act = clpeak_tflite::TfType::F32;
      std::string kernel;
      if (ps && ps->onDevice() && litertBindScalar(*ps, sp, err))
        kernel = litertMatMulKernel(ps->profileOps(err));
      CLPEAK_VLOG("litert-block[%s/%s]: prefill_s%lld%s projections ran as '%s'%s%s\n", dev.displayName.c_str(),
                  v.label, (long long)seq,
                  plan.gpuInt8KernelChoice
                      ? (vr.forms[f].gpuInt8Kernels ? " with 8-bit kernels" : " without 8-bit kernels")
                      : "",
                  kernel.c_str(), err.empty() ? "" : ": ", err.c_str());
      if (litertKernelArithmetic(kernel, dev.accel) == LitertArithmetic::Unknown)
      {
        CLPEAK_VLOG("litert-block[%s]: no kernel to read in the profile, so no more are taken\n",
                    dev.displayName.c_str());
        profileReadable = false;
      }
      return kernel;
    };
    pt.kernel = kernelOf(pt.form);
    if (pt.otherUs > 0.0 && vr.forms.size() == 2)
      pt.otherKernel = kernelOf(1 - pt.form);
  };

  auto measureDecode = [&](size_t vi, int64_t kv)
  {
    VariantResult &vr = results[vi];
    if (vr.decode.count(kv))
      return;
    Point pt;
    const double flops = blockFlops(1, kv);
    if (const std::string fence = decodeFence(kVariants[vi], dev.accel); !fence.empty())
    {
      pt.status = ResultStatus::Unsupported;
      pt.error = fence;
    }
    else if (std::string slow = tooSlow(flops, refDecodeRate, "one token"); !slow.empty())
    {
      pt.status = ResultStatus::Error;
      pt.error = slow;
    }
    else
    {
      measureRaced(vi, true, kv, kPrefillSeq, pt);
      if (pt.us > 0.0)
        refDecodeRate = flops / pt.us;
    }
    vr.decode[kv] = pt;
  };

  // A raced variant's word on its form, inside one of a row's sentences:
  // the form the point ran as and, when the other ran the same point, what
  // it read there (`reading` turns a time into the row's unit).
  auto formClause = [&](size_t vi, const Point &pt, auto &&reading) -> std::string
  {
    const VariantResult &vr = results[vi];
    if (vr.forms.size() < 2 || pt.us <= 0.0)
      return std::string();
    const bool k = vr.forms[pt.form].gpuInt8Kernels;
    std::string s = k ? ", with the GPU's 8-bit kernels allowed" : ", with the GPU's 8-bit kernels disallowed";
    if (pt.otherUs > 0.0)
      s += " (" + reading(pt.otherUs) + (k ? " without them" : " with them") + ")";
    return s;
  };

  auto emitPrefillTo = [&](logger::TestScope &test, size_t vi)
  {
    const Variant &v = kVariants[vi];
    const LitertPlan &plan = plans[vi];
    const VariantResult &vr = results[vi];
    for (int64_t seq : promptsFor(v))
    {
      const std::string metric = std::string(v.label) + "_s" + std::to_string(seq);
      auto it = vr.prefill.find(seq);
      const Point *pt = it == vr.prefill.end() ? nullptr : &it->second;
      // In the unit litertRateUnit gives the format and the projections'
      // kernel, and the other form's reading in its own.
      const char *unit = litertRateUnit(plan, pt ? pt->kernel : std::string(), dev.accel);
      const char *otherUnit = litertRateUnit(plan, pt ? pt->otherKernel : std::string(), dev.accel);
      const std::string form = !pt ? std::string() : formClause(vi, *pt, [&](double us) {
        return formatReading(blockFlops(seq, seq) * 1.0e6 / us, otherUnit);
      });
      const std::string kernel = !pt || pt->kernel.empty()
                                     ? std::string()
                                     : ", its projections as `" + pt->kernel + "`" +
                                           litertKernelNote(plan, pt->kernel, dev.accel);
      logger::EmitOptions o;
      o.description = std::string(v.note) + "  A prompt of " + std::to_string(seq) +
                      " tokens in one pass, counting every multiply in the layer" + form + kernel + "." +
                      seedNote(seq);
      if (v.unit || std::strcmp(unit, "flops") != 0)
        o.unit = unit;
      if (!vr.usable)
      {
        test.skip(metric, vr.skipStatus, vr.skipReason, o);
        continue;
      }
      if (it == vr.prefill.end())
        continue;
      if (it->second.us > 0.0)
        test.emit(metric, (float)(blockFlops(seq, seq) * 1.0e6 / it->second.us), o);
      else
        test.skip(metric, it->second.status, it->second.error, o);
    }
  };

  const std::string geometry =
      "one 2048-wide, 16-head decoder block with a SwiGLU feed-forward (50.6M parameters)";

  const logger::TestSpec prefillSpec = {
      "litert_block_prefill", "Transformer block, prefill", "flops", Category::Ai,
      "Prefill through " + geometry + " on this accelerator, at each format a model ships "
      "in: the phase that decides time to the first word.  Only the seven projections change "
      "format between rows, and LiteRT's own answer proves every operation ran here.",
      TestShape::Heterogeneous, "format and prompt length"};
  const logger::TestSpec prefillOpsSpec = {
      "litert_block_prefill", "Transformer block, prefill", "ops", Category::Ai,
      prefillSpec.description, TestShape::Heterogeneous, "format and prompt length"};
  const logger::TestSpec decodeSpec = {
      "litert_block_decode", "Transformer block, decode", "bps", Category::Ai,
      "How fast " + geometry + " streams its weights while generating one token with 2048 "
      "of context, counting the bytes each format actually moves.  A narrow-weight row far "
      "below the 16-bit one is an accelerator unpacking to full width before using them.",
      TestShape::Heterogeneous, "format"};
  const logger::TestSpec latencySpec = {
      "litert_block_latency", "Transformer block latency", "s", Category::Ai,
      "How long " + geometry + " takes at each format: multiply by a model's layer count "
      "for a floor on its time-to-first-token and per-token time.  Everything but attention "
      "costs the same at every context length, so what the decode rows add with context is "
      "attention.",
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
        profilePrefill(vi, seq);
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
        profilePrefill(vi, seq);
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
      const std::string head = std::string(v.note) + "  One token with 2048 of context: " +
                               std::to_string((unsigned long long)(wBytes >> 20)) + " MB of weights plus " +
                               std::to_string((unsigned long long)(kvB >> 20)) + " MB of cache, read once";
      o.description = head + ".";
      if (!validateVariant(vi))
      {
        test.skip(metric, vr.skipStatus, vr.skipReason, o);
        continue;
      }
      measureDecode(vi, kDecodeKv);
      const Point &pt = vr.decode[kDecodeKv];
      o.description = head + formClause(vi, pt, [&](double us) {
                        return formatReading((double)(wBytes + kvB) / (us * 1.0e-6), "bps");
                      }) + ".";
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
        if (!vr.usable)
          test.skip(metric, vr.skipStatus, vr.skipReason,
                    std::string("One pass over a 512-token prompt.  ") + v.note + seedNote(kPrefillSeq));
        else
        {
          measurePrefill(vi, kPrefillSeq);
          const Point &pt = vr.prefill[kPrefillSeq];
          const std::string note =
              "One pass over a 512-token prompt" +
              formClause(vi, pt, [&](double us) { return formatReading(us * 1.0e-6, "s"); }) + ".  " + v.note +
              seedNote(kPrefillSeq);
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
        if (!vr.usable)
        {
          test.skip(metric, vr.skipStatus, vr.skipReason,
                    "One token with " + std::to_string(kv) + " of context.  " + v.note);
          continue;
        }
        measureDecode(vi, kv);
        const Point &pt = vr.decode[kv];
        const std::string note =
            "One token with " + std::to_string(kv) + " of context" +
            formClause(vi, pt, [&](double us) { return formatReading(us * 1.0e-6, "s"); }) + ".  " + v.note;
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
