#ifdef ENABLE_COREML

// coreml-block: one fixed transformer decoder block through Core ML, run in
// the two regimes that bound all LLM inference, at each precision a language
// model ships in -- the rung above a raw matmul peak and below tokens per
// second, on the compute unit this device row names.
//
//   prefill (64/512/2048 tokens at once)      compute-bound -> effective FLOPS
//   decode  (1 token, 512/2048/8192 context)  memory-bound  -> effective B/s
//
// Geometry, ladders, precisions and reporting are the ONNX backend's
// (src/onnx/block.cpp), so a Core ML row divides by an ONNX one: the block
// is 2048 wide, 16 heads of 128, a 5504-wide SwiGLU feed-forward -- 50.6M
// parameters, 101 MB at fp16.  Attention is Core ML's own fused op where
// the OS has it (macOS 15 / iOS 18), which is what a converted model
// carries; older systems get explicit matmul / softmax / matmul.
//
// Every point is timed with the projection weights stored both ways, [in,
// out] and [out, in], and reports the faster, as the gemm rows do
// (CoremlLayoutRace): the prompt and the cache race separately, since a
// unit can take the wide multiply and the one-row one in opposite layouts.
//
// The block is where the Neural Engine's real answer on compressed weights
// lives.  A single matmul with resident activations can be decompressed at
// load time and run as fp16; a layer's activations are live, and whether
// the Neural Engine takes a 4-bit projection *there* is what a model would
// meet.  The compute plan settles it per session, and a row it declines
// reports so instead of measuring the CPU.

#include <coreml/coreml_peak.h>
#include "coreml_bench.h"
#include "coreml_model.h"
#include "coreml_session.h"

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
// (CoremlBlockShape::seedWidth), the ONNX block's kSeedWidth.  A prompt no
// longer than the seed, and decode's one row, gain nothing from one: the
// seed's widening weights would be as large as the input they replace.  The
// 64-token prompt keeps the plain input for its magnitudes too -- seeded, its
// largest value would grow from 2059 to 2716 (coremlBlockModel).
constexpr int64_t kSeedWidth = 64;

struct Variant
{
  const char *label;
  CoremlWeight w;
  bool sweep;         // walk the whole prompt and context ladders
  const char *unit;   // nullptr: the scope's own unit
  const char *note;
  bool int8Kv;        // the cache stored as int8
  bool decodeOnly;    // a cache format has no prefill row
  // Attention spelled as matmul, softmax and matmul instead of Core ML's
  // fused scaled_dot_product_attention op -- the form the ONNX backend's
  // block carries, and the one to read the fused op's own cost against.
  bool explicitAttention = false;
  // Walk the context ladder but not the prompt one: the row exists for how
  // attention scales with the cache, and prefill at one prompt is enough
  // to show whether the fused op wins there.
  bool sweepContext = false;
};

const Variant kVariants[] = {
    {"fp16", CoremlWeight::Fp16, true, nullptr,
     "16-bit weights and 16-bit arithmetic, the form an unquantized Core ML "
     "model is served in and the reference the other rows are read against.",
     false, false},
    {"fp16_explicit", CoremlWeight::Fp16, false, nullptr,
     "The fp16 row with attention spelled out as matmul, softmax and matmul "
     "over a key cache stored already transposed, instead of Core ML's fused "
     "scaled-dot-product op over the [heads, context, head_dim] cache it "
     "takes -- the form the ONNX backend's block carries, so this row divides "
     "directly against it.  Whatever separates it from the fp16 row is the "
     "fused op and its cache layout on this compute unit: on an M1 Pro's "
     "Neural Engine the fused op prefills slightly faster and decodes slower, "
     "half again the cost per cached token, the transpose of the cache it "
     "makes every token.",
     false, false, /*explicitAttention=*/true, /*sweepContext=*/true},
    {"int4_weight", CoremlWeight::Int4Block, true, nullptr,
     "4-bit weights, one scale per 32, against 16-bit activations -- what a "
     "quantized language model ships as.  The arithmetic stays 16-bit, so "
     "four bits buys weight traffic rather than rate; whether this compute "
     "unit takes the blocked form with live activations is the finding.",
     false, false},
    {"int4_lut", CoremlWeight::Int4Lut, false, nullptr,
     "4-bit palettized weights -- a 16-entry table per matrix, the compression "
     "Apple's Neural Engine decodes natively -- against 16-bit activations.",
     false, false},
    {"int8_weight", CoremlWeight::Int8Channel, false, nullptr,
     "8-bit weights, one scale per output column, against 16-bit activations.  "
     "This narrows only the weights where int8_qdq narrows the arithmetic too, "
     "so the gap between them is what integer multipliers are worth.",
     false, false},
    {"int8_qdq", CoremlWeight::Int8Qdq, false, "ops",
     "8-bit weights and 8-bit arithmetic through the projections, quantized in "
     "and out -- what headline TOPS figures are quoted for, measured on a whole "
     "layer.  Attention and the softmax stay 16-bit, as they do in every real "
     "deployment.",
     false, false},
    {"fp32", CoremlWeight::Fp32, false, nullptr,
     "Full precision, which nobody serves a language model in, here as a "
     "control -- and on the Neural Engine row, a layer Core ML has to send "
     "elsewhere, which the plan reports.",
     false, false},
    {"bf16", CoremlWeight::Bf16, false, nullptr,
     "bfloat16 has no arithmetic in Core ML; the row records the refusal.",
     false, false},
    {"fp8_weight", CoremlWeight::Fp8Block, false, nullptr,
     "8-bit float weights with block scales, if this OS takes them.",
     false, false},
    {"int8_kv", CoremlWeight::Fp16, true, nullptr,
     "16-bit throughout with only the cached context stored as 8-bit integers "
     "and decompressed into attention -- the axis that decides how long a "
     "conversation a device can hold, which is why this row sweeps context.",
     true, true},
};

int64_t weightParams()
{
  return 4 * kDModel * kDModel + 2 * kDModel * kFfnHidden + kFfnHidden * kDModel;
}

uint64_t weightBytes(const Variant &v)
{
  return 4 * coremlWeightBytes(v.w, kDModel, kDModel) +
         2 * coremlWeightBytes(v.w, kDModel, kFfnHidden) +
         coremlWeightBytes(v.w, kFfnHidden, kDModel);
}

uint64_t kvBytes(const Variant &v, int64_t kv)
{
  const uint64_t elem = v.int8Kv ? 1 : coremlElemBytes(coremlActDtype(v.w), 1);
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
  if (v.sweep || v.sweepContext)
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
  std::string glue;   // negligible operations the plan placed elsewhere
  // The plan sent the work elsewhere though this unit could run it: the
  // planner's size decision, which says nothing about the variant's other
  // points.
  bool plannerDeclined = false;
  bool wrong = false;         // the answer came back NaN or infinite
  bool transposed = false;    // the weight layout `us` was measured in
  double otherUs = -1.0;      // the other layout at this point, when both ran
  double createUs = 0.0;
};

struct VariantResult
{
  bool usable = false;
  std::string skipReason;
  ResultStatus skipStatus = ResultStatus::Unsupported;
  std::map<int64_t, Point> prefill, decode;
  // One weight-layout race per shape (CoremlLayoutRace): decode is an M=1
  // multiply where prefill is a wide one, and a unit can take the two in
  // opposite layouts -- the M1 Pro's GPU prefills blockwise int4 faster
  // stored [in, out] and decodes it faster stored [out, in].
  CoremlLayoutRace prefillRace, decodeRace;
  bool decodeRaceSet = false;
};

// A row's word on the layout its point's weights were stored in
// (CoremlLayoutRace), and what the other read at the same point when both
// ran: `work` / us * `scale` is the row's figure in `unit`, or with no work
// the row is a time.
std::string layoutNote(const Point &pt, double work, double scale, const char *unit)
{
  std::string s = "  Weights stored " + std::string(coremlLayoutName(pt.transposed));
  if (pt.otherUs > 0.0)
  {
    char buf[128];
    if (work > 0.0)
      std::snprintf(buf, sizeof buf, ": %#.3g %s against %#.3g stored %s", work / pt.us * scale, unit,
                    work / pt.otherUs * scale, coremlLayoutName(!pt.transposed));
    else
      std::snprintf(buf, sizeof buf, ": %#.3g ms against %#.3g stored %s", pt.us / 1.0e3,
                    pt.otherUs / 1.0e3, coremlLayoutName(!pt.transposed));
    s += buf;
  }
  return s + ".";
}

CoremlBlockShape shapeFor(const Variant &v, bool decode, int64_t kvLen, int64_t prefillSeq, int spec)
{
  CoremlBlockShape sh;
  sh.dModel = kDModel;
  sh.heads = kHeads;
  sh.headDim = kHeadDim;
  sh.ffnHidden = kFfnHidden;
  sh.seq = decode ? 1 : prefillSeq;
  sh.kvLen = decode ? kvLen : 0;
  sh.weights = v.w;
  sh.int8Kv = v.int8Kv;
  sh.fusedAttention = spec >= 9 && !v.explicitAttention;
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

int CoreMLPeak::runBlock(const coreml_device_info_t &dev, benchmark_config_t &cfg)
{
  (void)cfg;
  const int spec = coremlSpecVersion();
  constexpr size_t kNVariants = sizeof(kVariants) / sizeof(kVariants[0]);
  std::vector<VariantResult> results(kNVariants);

  // Time one regime end to end with the weights stored one way: mean us
  // per block, or negative with the point's status and error set.
  auto measureOne = [&](const Variant &v, bool decode, int64_t kvLen, int64_t prefillSeq,
                        bool transposed, Point &pt)
  {
    std::string err;
    CoremlBlockShape sh = shapeFor(v, decode, kvLen, prefillSeq, spec);
    sh.transposedWeights = transposed;
    pt.transposed = transposed;
    auto s = CoremlSession::create(dev, coremlBlockModel(spec, sh), err);
    const std::string what = (decode ? "decode_kv" + std::to_string(kvLen)
                                     : "prefill_s" + std::to_string(prefillSeq)) +
                             " " + coremlLayoutName(transposed);
    if (!s)
    {
      CLPEAK_VLOG("coreml-block[%s/%s]: %s create failed: %s\n", dev.displayName.c_str(), v.label,
                  what.c_str(), err.c_str());
      pt.error = err;
      pt.status = ResultStatus::Unsupported;
      return;
    }
    const double createUs = coremlCreateUs(*s);
    pt.createUs = createUs;
    CLPEAK_VLOG("coreml-block[%s/%s]: %s create %.1f s\n", dev.displayName.c_str(), v.label, what.c_str(),
                createUs / 1.0e6);
    if (!s->onDevice())
    {
      pt.error = coremlOffDeviceReason(dev, *s);
      pt.status = ResultStatus::Unsupported;
      pt.plannerDeclined = s->offDeviceCapable();
      CLPEAK_VLOG("coreml-block[%s/%s]: %s %s\n", dev.displayName.c_str(), v.label, what.c_str(),
                  pt.error.c_str());
      return;
    }
    if (createUs > kCoremlMaxBlockCreateUs)
    {
      pt.error = "model creation took " + std::to_string((long long)(createUs / 1.0e6)) + " s, exceeds " +
                 std::to_string((long long)(kCoremlMaxBlockCreateUs / 1.0e6)) + " s compilation budget";
      pt.status = ResultStatus::Unsupported;
      return;
    }
    const int act = coremlActDtype(v.w);
    if (!coremlBindScalar(*s, "s", act == CML_BF16 ? CML_FP16 : act, err))
    {
      pt.error = err;
      pt.status = ResultStatus::Error;
      return;
    }
    auto m = coremlMeasure(*s, warmupCount, kBlockBudgetUs, forceIters, specifiedIters);
    if (m.meanUs <= 0.0)
    {
      pt.error = m.error.empty() ? "run failed" : m.error;
      pt.status = m.status;
      return;
    }
    // What the timed runs computed.  A wrong answer withholds this point
    // alone: each point is its own graph and its own row.
    const std::string wrong = coremlNonFiniteReason(
        *s, "Yr", act,
        (decode ? "for one token against " + std::to_string(kvLen) + " of context"
                : "for a " + std::to_string(prefillSeq) + "-token prompt") +
            ", weights stored " + coremlLayoutName(transposed));
    if (!wrong.empty())
    {
      CLPEAK_VLOG("coreml-block[%s/%s]: %s\n", dev.displayName.c_str(), v.label, wrong.c_str());
      pt.error = wrong;
      pt.status = ResultStatus::Error;
      pt.wrong = true;
      return;
    }
    pt.us = m.meanUs;
    pt.status = ResultStatus::Ok;
    pt.glue = coremlGlueNote(*s);
    CLPEAK_VLOG("coreml-block[%s/%s]: %s %.1f us\n", dev.displayName.c_str(), v.label, what.c_str(),
                m.meanUs);
  };

  // One point, raced (CoremlLayoutRace): each weight layout the shape's race
  // still runs is built and timed, and the faster stands for the point.  A
  // wrong answer from either withholds it.  A layout that fails where the
  // other runs leaves the race; where both fail, the point reports it and
  // the next point tries again, as a point of one layout would.
  auto measure = [&](const Variant &v, bool decode, int64_t kvLen, int64_t prefillSeq,
                     CoremlLayoutRace &race, Point &pt)
  {
    Point one[2];
    bool tried[2] = {false, false};
    for (int t = 0; t < 2; t++)
    {
      if (!race.runs(t))
        continue;
      tried[t] = true;
      measureOne(v, decode, kvLen, prefillSeq, t, one[t]);
      if (one[t].wrong)
      {
        pt = one[t];
        return;
      }
    }
    const bool ok[2] = {one[0].us > 0.0, one[1].us > 0.0};
    if (ok[0] && ok[1])
    {
      const double rate[2] = {1.0 / one[0].us, 1.0 / one[1].us};
      const double createUs[2] = {one[0].createUs, one[1].createUs};
      race.settle(rate, createUs);
      const bool t = one[1].us < one[0].us;
      pt = one[t];
      pt.otherUs = one[!t].us;
      return;
    }
    if (ok[0] || ok[1])
    {
      const bool t = ok[1];
      if (tried[!t] && !one[!t].plannerDeclined)
        race.drop(!t);
      pt = one[t];
      return;
    }
    // Neither ran.  The planner's size decision first, since it settles
    // nothing about the variant's other points; then [in, out]'s reason.
    if (tried[1] && (one[1].plannerDeclined || !tried[0]))
      pt = one[1];
    else
      pt = one[0];
  };

  // The decode race starts where the prefill one ended when that was the
  // same program either way (CoremlLayoutRace::nextShape), and fresh when not.
  auto decodeRace = [&](VariantResult &vr) -> CoremlLayoutRace &
  {
    if (!vr.decodeRaceSet)
    {
      vr.decodeRace = vr.prefillRace.nextShape();
      vr.decodeRaceSet = true;
    }
    return vr.decodeRace;
  };

  // Core ML's compiler crashes -- a segfault inside BNNS's graph compiler,
  // reached through Espresso's CPU-backend lowering pass -- on the block's
  // decode form on macOS 27.0 (26A428): on the CPU compute unit for every
  // weight format (int4_weight's kv2048 point compiled, its next did not),
  // on the GPU unit for fp16 weights, whose M=1 projections the planner
  // hands to BNNS.  Prefill of the same block compiles everywhere, the
  // Neural Engine (which takes the whole graph, so BNNS never compiles it)
  // runs every form, and the same build ran clean on 26.6.  A crash in a
  // system library cannot be caught, so on that OS those rows are not
  // asked for; lift the fence when a release fixes it.
  auto decodeFence = [&](const Variant &v) -> std::string {
    if (coremlOsMajorVersion() < 27 || dev.kind == CoremlDeviceKind::NeuralEngine)
      return std::string();
    if (dev.kind == CoremlDeviceKind::Cpu || v.w == CoremlWeight::Fp16)
      return "Core ML's compiler crashes (a segfault in BNNS graph compilation, macOS 27) on the "
             "block's decode form for this compute unit, so only prefill is sent to it";
    return std::string();
  };

  // Everything that has to be true before a variant is worth timing: the OS
  // accepts its format, the fixed geometry fits, and one small session
  // compiles and lands on this compute unit.
  auto validateVariant = [&](const Variant &v, VariantResult &vr) -> bool
  {
    if (vr.usable || !vr.skipReason.empty())
      return vr.usable;
    if (v.decodeOnly)
      if (const std::string fence = decodeFence(v); !fence.empty())
      {
        vr.skipReason = fence;
        return false;
      }
    if (coremlSpecForWeight(v.w) > spec)
    {
      vr.skipReason = "needs " + coremlOsForSpec(coremlSpecForWeight(v.w)) + " (model specification " +
                      std::to_string(coremlSpecForWeight(v.w)) + "); this OS accepts " + std::to_string(spec);
      return false;
    }
    if (v.explicitAttention && spec < 9)
    {
      // Below spec 9 there is no fused op, so the fp16 row already spells
      // attention out and this one would repeat it.
      vr.skipReason = "this OS has no fused attention op (model specification 9, " +
                      coremlOsForSpec(9) + "), so the fp16 row already spells attention out";
      return false;
    }
    {
      const uint64_t needed = weightBytes(v) + kvBytes(v, contextsFor(v).back()) +
                              (64ull << 20) * coremlElemBytes(coremlActDtype(v.w), 1) / 2;
      const uint64_t budget = clpeak::memoryBudget(~0ull, 8);
      if (budget && budget < needed)
      {
        vr.skipReason = "not enough memory for the canonical block; its geometry is fixed so the "
                        "numbers stay comparable, and a smaller layer would not be the same test";
        return false;
      }
    }
    // The probe is the smallest point the variant reports -- its first
    // prompt, or its first context for a cache format -- and its timing is
    // kept: a point measured is a point measured.  A point the planner
    // declined for its size does not settle the variant: the explicit
    // attention block's 64-token prompt stays on the CPU where its 512-token
    // one goes to the Neural Engine, so that point reports the planner's
    // decision and the others are measured.
    Point pt;
    const bool decode = v.decodeOnly;
    const int64_t probeSeq = decode ? kPrefillSeq : promptsFor(v).front();
    measure(v, decode, kKvLadder[0], probeSeq, decode ? decodeRace(vr) : vr.prefillRace, pt);
    if (pt.us <= 0.0 && !pt.plannerDeclined)
    {
      vr.skipReason = pt.error.empty() ? std::string("the block could not be built for ") + v.label : pt.error;
      vr.skipStatus = pt.status;
      return false;
    }
    if (decode)
      vr.decode[kKvLadder[0]] = pt;
    else
      vr.prefill[probeSeq] = pt;
    vr.usable = true;
    return true;
  };

  double refPrefillRate = 0.0, refDecodeRate = 0.0;
  auto affordable = [&](double flops, double rate) { return rate <= 0.0 || (flops / rate) <= (double)kBlockBudgetUs; };

  auto measurePrefill = [&](const Variant &v, VariantResult &vr, int64_t seq)
  {
    auto it = vr.prefill.find(seq);
    if (it != vr.prefill.end())
      return;
    Point pt;
    const double flops = blockFlops(seq, seq);
    if (!affordable(flops, refPrefillRate))
    {
      pt.status = ResultStatus::Error;
      pt.error = "one pass would take about " + std::to_string((long long)(flops / refPrefillRate / 1.0e6)) +
                 " s on this compute unit, too slow to measure";
    }
    else
    {
      measure(v, false, kDecodeKv, seq, vr.prefillRace, pt);
      if (pt.us > 0.0)
        refPrefillRate = flops / pt.us;
    }
    vr.prefill[seq] = pt;
  };

  auto measureDecode = [&](const Variant &v, VariantResult &vr, int64_t kv)
  {
    auto it = vr.decode.find(kv);
    if (it != vr.decode.end())
      return;
    Point pt;
    const double flops = blockFlops(1, kv);
    if (const std::string fence = decodeFence(v); !fence.empty())
    {
      pt.status = ResultStatus::Unsupported;
      pt.error = fence;
    }
    else if (!affordable(flops, refDecodeRate))
    {
      pt.status = ResultStatus::Error;
      pt.error = "one token would take about " + std::to_string((long long)(flops / refDecodeRate / 1.0e6)) +
                 " s on this compute unit, too slow to measure";
    }
    else
    {
      measure(v, true, kv, kPrefillSeq, decodeRace(vr), pt);
      if (pt.us > 0.0)
        refDecodeRate = flops / pt.us;
    }
    vr.decode[kv] = pt;
  };

  auto emitPrefillTo = [&](logger::TestScope &test, const Variant &v, const VariantResult &vr)
  {
    for (int64_t seq : promptsFor(v))
    {
      const std::string metric = std::string(v.label) + "_s" + std::to_string(seq);
      logger::EmitOptions o;
      o.description = std::string(v.note) + "  A prompt of " + std::to_string(seq) +
                      " tokens in one pass, counting every multiply in the layer." + seedNote(seq);
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
      {
        o.description += layoutNote(it->second, blockFlops(seq, seq), 1.0e-6,
                                    v.unit ? "TOPS" : "TFLOPS") +
                         it->second.glue;
        test.emit(metric, (float)(blockFlops(seq, seq) * 1.0e6 / it->second.us), o);
      }
      else
        test.skip(metric, it->second.status, it->second.error, o);
    }
  };

  const std::string geometry =
      "One 2048-wide, 16-head decoder block with a SwiGLU feed-forward (50.6M parameters)";
  const std::string attention = spec >= 9 ? "attention as Core ML's fused scaled-dot-product op "
                                            "(the fp16_explicit row spells it out instead)"
                                          : "attention as explicit matmul, softmax and matmul";

  const logger::TestSpec prefillSpec = {
      "coreml_block_prefill", "Transformer block, prefill", "flops", Category::Ai,
      geometry + ", working through a prompt on this compute unit -- the phase that decides "
      "how long you wait for the first word -- at each precision a model ships in, with " +
      attention + ".  Only the seven projection matmuls change precision, so whatever separates "
      "two rows is the projection format, its weights stored in whichever of their two orders, "
      "[in, out] or [out, in], this unit runs faster; and the compute plan proves every "
      "operation ran here.",
      TestShape::Heterogeneous, "data type and prompt length"};
  const logger::TestSpec prefillOpsSpec = {
      "coreml_block_prefill", "Transformer block, prefill", "ops", Category::Ai,
      prefillSpec.description, TestShape::Heterogeneous, "data type and prompt length"};
  const logger::TestSpec decodeSpec = {
      "coreml_block_decode", "Transformer block, decode", "bps", Category::Ai,
      "How fast " + geometry + " streams its weights on this compute unit while generating a "
      "token with 2048 of context, at each precision.  Each row counts the bytes that format "
      "actually moves, so it compares directly against the resident-weight bandwidth rows -- "
      "and a narrow-weight row far below the 16-bit one is a unit unpacking to full width "
      "before using them.  The weights are stored in whichever of their two orders, [in, out] "
      "or [out, in], this unit streams faster.",
      TestShape::Heterogeneous, "data type"};
  const logger::TestSpec latencySpec = {
      "coreml_block_latency", "Transformer block latency", "s", Category::Ai,
      "How long " + geometry + " takes on this compute unit at each precision: multiply by a "
      "model's layer count for a floor on its time-to-first-token and per-token time here.  "
      "Everything but attention costs the same at every context length, so whatever the decode "
      "rows add as the context grows is attention.",
      TestShape::Heterogeneous, "data type, phase and context length"};

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
      VariantResult &vr = results[vi];
      if (!validateVariant(v, vr))
      {
        emitPrefillTo(test, v, vr);
        continue;
      }
      for (int64_t seq : promptsFor(v))
      {
        if (clpeak::cancelRequested())
          break;
        measurePrefill(v, vr, seq);
      }
      if (auto it = vr.prefill.find(kPrefillSeq); it != vr.prefill.end() && it->second.us > 0.0)
        refDecodeRate = blockFlops(kPrefillSeq, kPrefillSeq) / it->second.us;
      emitPrefillTo(test, v, vr);
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
      VariantResult &vr = results[vi];
      if (!validateVariant(v, vr))
      {
        emitPrefillTo(test, v, vr);
        continue;
      }
      for (int64_t seq : promptsFor(v))
      {
        if (clpeak::cancelRequested())
          break;
        measurePrefill(v, vr, seq);
      }
      emitPrefillTo(test, v, vr);
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
      VariantResult &vr = results[vi];
      const std::string metric = std::string(v.label) + "_kv" + std::to_string(kDecodeKv);
      const uint64_t wBytes = weightBytes(v);
      const uint64_t kvB = kvBytes(v, kDecodeKv);
      logger::EmitOptions o;
      o.description = std::string(v.note) + "  One token with 2048 of context: " +
                      std::to_string((unsigned long long)(wBytes >> 20)) + " MB of weights plus " +
                      std::to_string((unsigned long long)(kvB >> 20)) +
                      " MB of cached context, read in full for one token.";
      if (!validateVariant(v, vr))
      {
        test.skip(metric, vr.skipStatus, vr.skipReason, o);
        continue;
      }
      measureDecode(v, vr, kDecodeKv);
      const Point &pt = vr.decode[kDecodeKv];
      if (pt.us > 0.0)
      {
        o.description += layoutNote(pt, (double)(wBytes + kvB), 1.0e-3, "GB/s") + pt.glue;
        test.emit(metric, (float)((double)(wBytes + kvB) / (pt.us * 1.0e-6)), o);
      }
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
        const std::string note = std::string("One pass over a 512-token prompt.  ") + v.note +
                                 seedNote(kPrefillSeq);
        if (!vr.usable)
          test.skip(metric, vr.skipStatus, vr.skipReason, note);
        else
        {
          measurePrefill(v, vr, kPrefillSeq);
          const Point &pt = vr.prefill[kPrefillSeq];
          if (pt.us > 0.0)
            test.emit(metric, (float)(pt.us * 1e-6), (note + layoutNote(pt, 0.0, 0.0, "")).c_str());
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
        measureDecode(v, vr, kv);
        const Point &pt = vr.decode[kv];
        if (pt.us > 0.0)
          test.emit(metric, (float)(pt.us * 1e-6), (note + layoutNote(pt, 0.0, 0.0, "")).c_str());
        else
        {
          test.skip(metric, pt.status, pt.error.empty() ? "run failed" : pt.error, note);
          // A longer context is more work, so a point the unit could not
          // run ends the ladder -- but one the planner kept back for its
          // size says nothing about the next.
          if (!pt.plannerDeclined)
            break;
        }
      }
    }
    test.end();
  }

  return 0;
}

#endif // ENABLE_COREML
