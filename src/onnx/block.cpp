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

#include <algorithm>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdio>
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

  // The width of the seed a prompt longer than it enters through
  // (OnnxBlockShape::seedWidth), the GEMM chains' kSeedWidth.  A prompt no
  // longer than the seed, and decode's one row, gain nothing from one: the
  // seed's widening weights would be as large as the input they replace.  The
  // 64-token prompt keeps the plain input for its magnitudes too -- seeded, its
  // largest value would grow from 1954 to 2962 (blockWeights).  The fusion
  // probes run there, without the seed, and the projections they judge are
  // the same nodes either way.
  constexpr int64_t kSeedWidth = 64;

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
       "whole layer.  Attention and the softmax stay in floating point, as "
       "they do in every real deployment: 16-bit unless the row says "
       "otherwise."},

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

  // Why a variant cannot be sent to this provider at all, or empty.
  //
  // Not a statement about the format -- `onnxProviderFenceReason` is where
  // those live, and it is consulted through the gemm probe by every test
  // here.  This is for a fault only this test's graph provokes, where
  // fencing the format everywhere would withhold rows that demonstrably
  // run.
  //
  // DirectML on this block's blocked int8 weights: session creation ends the
  // process with an integer divide by zero (exit 0xC0000094), after ORT's
  // own transformers have finished with the graph -- the last line logged is
  // the attention scale being folded away -- and before the allocation
  // planner speaks, which is where the DML provider compiles its fused
  // partitions.  Seven DequantizeLinear(int8, one fp16 scale per 32 rows:
  // opset 21, block_size=32, axis=0) feeding 2048-wide MatMuls.  ONNX
  // Runtime 1.24.4, an Intel Arc A380, 2026-09-21, after the same block's
  // fp16 and int4_weight forms had built and run; nothing after it ran.
  //
  // int4 escapes because ORT rewrites it first: DQMatMulToMatMulNBits takes
  // 4-bit weights only (Is4BitIntType, qdq_selectors.cc), so the int4 block
  // reaches DirectML as MatMulNBits and the int8 one as a raw opset-21
  // DequantizeLinear, which the DML provider registers with no support query
  // and hands to DML_DEQUANTIZE_OPERATOR_DESC (DmlOperatorQuantization21).
  //
  // The same format at matmul sizes is fine on the same card, and other
  // DirectML devices (an RTX 4060, a UHD 630, an Adreno X1-45, the DirectML
  // CPU) run this block, so the fault wants that provider *and* this graph
  // *and* -- as far as anyone has seen -- that GPU.  Only the first two can
  // be keyed on: clpeak registers DirectML with no device id and ORT picks
  // the adapter, so which GPU is behind this device is not something the
  // backend can ask.  Naming one would be a guess, and a guess that is
  // wrong on a two-GPU box takes the run down.  So the row is withheld on
  // every DirectML device and says so; the gemm, conv and accuracy rows,
  // which is where the format is otherwise measured, are untouched.
  std::string variantFence(const Variant &v, const onnx_ep_info_t &ep)
  {
    if (ep.providerKey == "DmlExecutionProvider" && v.wDtype == ONNX_DT_INT8 &&
        !v.qdq && v.wBlock > 0)
      return "DirectML takes the process down (an integer divide by zero "
             "while the session is created, ONNX Runtime 1.24.4) on this "
             "block's blocked int8 weights, which ORT hands it unfused "
             "because only 4-bit weights become MatMulNBits.  Seen on an "
             "Intel Arc A380 and not on other DirectML devices, but a "
             "session picks its adapter inside the runtime, so no DirectML "
             "device is sent this graph; the same format is still measured "
             "by the matmul rows, where it runs";
    return std::string();
  }

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

  // The matmuls a quantized block quantizes: the query, key, value and output
  // projections and the feed-forward's gate, up and down.  A quantized-cache
  // row quantizes attention's two instead.
  constexpr size_t kProjections = 7;
  constexpr size_t kAttentionMatmuls = 2;

  // How many of the graph's `expected` quantized matmuls ran as quantized
  // kernels, rather than as floating-point multiplies over weights the
  // provider unpacked -- all of them when it compiled the graph into kernels
  // of its own, where there is nothing to count.
  //
  // onnx_session.cpp's onnxOpsRanQuantizedMatMul cannot answer this one.  It
  // reads a failed fusion as "a plain MatMul beside the dequantize nodes", and a
  // transformer block has two plain MatMuls that are *supposed* to be there --
  // attention is not quantized in any variant here -- so that test would reject
  // every block unconditionally.
  //
  // So this counts.  A provider that fused names a quantized kernel for every
  // quantized matmul (QLinearMatMul, MatMulNBits), or swallowed the subgraph
  // into one kernel of its own, in which case no DequantizeLinear kernel runs
  // at all.  A matmul it did not fuse has no choice but to execute its
  // DequantizeLinear as a real kernel -- a full pass over its weights on every
  // run -- and multiply in floating point, which is not a rate any quantized
  // deployment would ever see.  Counting is what catches the partial case: on
  // a Zen 2 without VNNI, signed int8 activations fused two of the seven
  // projections and multiplied the other five in floating point, and asking
  // only whether any quantized kernel ran passed that row as int8.
  size_t quantizedMatmulsFused(const std::vector<std::string> &ops,
                               size_t expected)
  {
    if (ops.empty())
      return expected; // no profile to judge by; do not reject on silence

    for (const auto &op : ops)
      if (op == "DequantizeLinear" || op == "QuantizeLinear")
        return std::min(expected, onnxCountQuantizedKernels(ops));
    return expected;
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

  // `nativeProfile` and `legend` are logPrefillProfile's: the provider's own
  // profiler's file, and where the graph's node legend goes.
  BlockRun makeRun(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                   const Variant &v, bool decode, int64_t kvLen,
                   int64_t prefillSeq, int qActDtype, bool profile,
                   const std::string &nativeProfile = std::string(),
                   std::vector<std::string> *legend = nullptr)
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
    sh.seedWidth = sh.seq > kSeedWidth ? kSeedWidth : 0;
    // The view of the output the provider is known to take (OnnxReduceView).
    sh.reduceView = onnxPrefersRank4Reduce(rt, ep) ? OnnxReduceView::Rank4
                                                   : OnnxReduceView::Rows;

    auto create = [&]() -> OnnxSessionResult {
      std::string model = onnxBlockModel(sh, legend);
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
      // A profiled run is the fusion probe at the smallest point, which asks
      // what the provider fused and not where it ran: Core ML keeps the
      // 64-token layer on the CPU and sends the 512-token one to the Neural
      // Engine, and verifying the probe would refuse every point above it
      // (onnx_session.h).  The timed points verify.
      // QDQ propagation stays off for every variant, and it is the int8 prompt
      // with fp32 float parts that needs it.  There, nothing but a Reshape and
      // a Transpose separates a projection's closing dequantize from
      // attention, the propagation copies it across both, and ONNX Runtime's
      // CPU provider then fused the scores matmul into MatMulIntegerToFloat --
      // int8 attention, in a row that says attention stays floating point.
      // The fp16 form's casts stop it on their own, and no other variant has a
      // quantize or dequantize beside a node it crosses.
      auto ses = onnxCreateSession(rt, ep, model, /*keepConstantsUnfolded=*/true,
                                   profile, keepQdqUnfused,
                                   /*verifyPlacement=*/!profile,
                                   /*keepQdqInPlace=*/true, nativeProfile);
      // The model is the largest allocation in the process; drop it before
      // anything else is allocated on top of the session's own copy.
      model.clear();
      model.shrink_to_fit();
      return ses;
    };
    OnnxSessionResult ses = create();
    // A provider can refuse the reduction that brings the block's row back
    // while taking the block: the QNN Adreno backend took the one-token
    // decode and refused every prompt length.  The fp16 reference row -- the
    // first built -- tries the rank-4 view once; a provider that takes it
    // keeps it for every row after (onnxPrefersRank4Reduce), and a row that
    // fails for any other reason pays no second compile.
    if (!ses.session && !ses.offDevice && &v == &kVariants[0] &&
        sh.reduceView == OnnxReduceView::Rows &&
        onnxFailureStatus(ses.error) == ResultStatus::Unsupported &&
        !onnxReasonIsOutOfMemory(ses.error) && !clpeak::cancelRequested())
    {
      CLPEAK_VLOG("onnx-block[%s/%s]: refused (%s); retrying the reduction "
                  "through a rank-4 view\n",
                  ep.providerKey.c_str(), v.label, ses.error.c_str());
      sh.reduceView = OnnxReduceView::Rank4;
      OnnxSessionResult r4 = create();
      if (r4.session)
      {
        onnxNoteRank4Reduce(rt, ep);
        ses = r4;
      }
      else
        CLPEAK_VLOG("onnx-block[%s/%s]: refused with the rank-4 view too (%s)\n",
                    ep.providerKey.c_str(), v.label, r4.error.c_str());
    }
    if (!ses.session)
    {
      r.error = ses.error;
      return r;
    }
    r.session = ses.session;

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

  // onnxNonFiniteReason over the row a run returned.  `yr` is ORT's own
  // allocation, in host memory, so its size is read from it rather than
  // assumed; empty when it cannot be read.
  std::string nonFiniteRow(const OrtRuntime &rt, OrtValue *yr, int dtype,
                           const std::string &where)
  {
    if (!yr)
      return std::string();
    OrtTensorTypeAndShapeInfo *info = nullptr;
    size_t count = 0;
    void *data = nullptr;
    OrtStatus *st = rt.api->GetTensorTypeAndShape(yr, &info);
    if (!st)
      st = rt.api->GetTensorShapeElementCount(info, &count);
    if (info)
      rt.api->ReleaseTensorTypeAndShapeInfo(info);
    if (!st)
      st = rt.api->GetTensorMutableData(yr, &data);
    if (st)
    {
      rt.api->ReleaseStatus(st);
      return std::string();
    }
    return onnxNonFiniteReason(data, (int64_t)count, dtype, where);
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
      status = onnxFailureStatus(r.error);
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
    else
    {
      // What the timed runs computed, read once they are over.  A wrong
      // answer withholds this point alone: each point is its own graph and
      // its own row.
      const std::string wrong = nonFiniteRow(
          rt, r.outVals[0], v.actDtype,
          decode ? "for one token against " + std::to_string(kvLen) +
                       " of context"
                 : "for a " + std::to_string(prefillSeq) + "-token prompt");
      if (!wrong.empty())
      {
        CLPEAK_VLOG("onnx-block[%s/%s]: %s\n", ep.providerKey.c_str(),
                    v.label, wrong.c_str());
        error = wrong;
        status = ResultStatus::Error;
        mean_us = -1.0;
      }
    }
    destroyRun(rt, r);
    return mean_us;
  }

  // What the provider's own profiler (onnxNativeProfilePath) says the
  // 512-token prompt's time went to, for the log: a build of its own, never
  // one that was timed, and only when --verbose asked for the detail.  QNN
  // runs the whole layer as one kernel, so ORT's profile cannot split it, and
  // its own profile names nodes only -- the legend logged with it turns those
  // names back into the graph's operations.  `form` is appended to the tag.
  void logPrefillProfile(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                         const Variant &v, int qActDtype, unsigned int warmup,
                         const char *form)
  {
    const std::string path =
        clpeak::verboseEnabled() ? onnxNativeProfilePath(ep) : std::string();
    if (path.empty() || clpeak::cancelRequested())
      return;
    const std::string tag = ep.providerKey + "/" + v.label + form + " s" +
                            std::to_string(kPrefillSeq);
    std::vector<std::string> legend;
    BlockRun r = makeRun(rt, ep, v, /*decode=*/false, kDecodeKv, kPrefillSeq,
                         qActDtype, /*profile=*/false, path, &legend);
    if (r.session && timeRuns(rt, r, 1 + warmup) > 0.0)
      timeRuns(rt, r, 3);
    else
      CLPEAK_VLOG("onnx-block[%s]: profiled build failed: %s\n", tag.c_str(),
                  r.error.c_str());
    destroyRun(rt, r);
    for (const auto &line : legend)
      CLPEAK_VLOG("onnx-native-profile[%s]: legend %s\n", tag.c_str(),
                  line.c_str());
    onnxLogNativeProfile(path, tag);
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

    // int8 QDQ only: the prompt ran with the layer's floating-point parts in
    // fp32, which this provider took faster than fp16 -- what the fp16 form
    // took, and the fp32 form's provenance (its fused kernel can differ).
    bool prefillFp32 = false;
    double prefillFp16Us = 0.0;
    std::string prefillProv;
  };

  // Quantization schemes for the QDQ form, tried in order until one fuses.
  //
  // int8 has two spellings and no provider takes both: x86 MLAS without VNNI
  // implements unsigned activations against signed weights and fuses the
  // signed form only in part -- two of the seven projections on a Zen 2 --
  // while TensorRT rejects uint8 outright.  Trying is the only way to know,
  // exactly as in gemm.cpp -- the fusion check is the selector, and it asks
  // for every projection.
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

  // The fp16 variant is kVariants[0] and is measured first, which makes it
  // the reference every other decode row is compared against: same layer,
  // same context, differing only in the width each declares.
  constexpr size_t kRefVariant = 0;
  double refDecodeUs = 0.0, refDecodeBytes = 0.0;

  // What this provider demonstrably streams, measured (see onnxStreamBps).
  // The decode rows count the bytes the model *declares*, which is all clpeak
  // can know: a provider is free to store them narrower than asked, and no
  // vendor publishes which do.  A row implying more traffic than the device
  // has been seen to move is that happening, and it is caught by measurement
  // rather than by a list of provider names -- the point being that a
  // provider nobody here has run gets the same treatment.
  const double streamBps = onnxStreamBps(rt, ep);
  // The one narrowing the probe can prove rather than suspect: fp32 held at
  // half width (onnxFp32Narrowed).  Where it is proven the fp32 row is
  // credited the bytes that move, instead of carrying a note beside a figure
  // twice what the device streamed.
  const bool fp32Narrowed = onnxFp32Narrowed(rt, ep);

  // What only a run can say: the kernel the provider fused to, the scheme it
  // settled on, and whether it converted the activations first.
  auto provenance = [](const Variant &v, const VariantResult &vr)
  {
    (void)v;
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

  // The affordability seed, carried across variants.  Each one's first point
  // has no measurement of its own to predict from, and letting all six pay
  // full price on a provider that has already proved itself slow is how this
  // test would come to take tens of minutes.  The most recent measured rate
  // is the estimate: precisions differ by a factor of a few, not orders.
  double refPrefillRate = 0.0, refDecodeRate = 0.0;

  // A prompt row's provenance: the fp16 form's, or -- where the layer's
  // floating-point parts ran in fp32 because that was faster -- the fp32
  // form's, with a sentence saying so and what each form took.
  auto prefillProvenance = [&](const Variant &v, const VariantResult &vr,
                               int64_t seq) -> std::string
  {
    if (!vr.prefillFp32 || seq != kPrefillSeq)
      return provenance(v, vr);
    const auto it = vr.prefill.find(kPrefillSeq);
    const double flops = blockFlops(kPrefillSeq, kPrefillSeq);
    char rates[96];
    std::snprintf(rates, sizeof rates, "%.3g TOPS against %.3g",
                  flops / it->second.us / 1.0e6,
                  flops / vr.prefillFp16Us / 1.0e6);
    std::string s = vr.prefillProv +
                    "  Attention, the softmax and the residuals ran in fp32 "
                    "here rather than fp16, because this provider takes the "
                    "layer faster that way: " +
                    std::string(rates) + ".";
    if (fp32Narrowed)
      s += "  It holds fp32 at 16 bits anyway, so the two forms do the same "
           "arithmetic and differ in the conversions around each quantized "
           "projection.";
    return s;
  };

  // What a prompt row says about its way in, when it has a seed
  // (kSeedWidth): the gemm rows say the same of theirs.
  auto seedNote = [](int64_t seq) -> std::string
  {
    if (seq <= kSeedWidth)
      return std::string();
    return "  The prompt starts from a " + std::to_string(kSeedWidth) +
           "-wide seed that takes a runtime value, and one more multiply, not "
           "counted, widens it into the layer's input -- inside the figure, "
           "a quarter of a percent of its arithmetic.";
  };

  auto emitPrefillTo = [&](logger::TestScope &test, const Variant &vv,
                           const VariantResult &vvr)
  {
    for (int64_t sseq : promptsFor(vv))
    {
      const std::string prov = prefillProvenance(vv, vvr, sseq);
      const std::string metric = std::string(vv.label) + "_s" + std::to_string(sseq);
      logger::EmitOptions o;
      o.description = std::string(vv.note) + "  A prompt of " +
                      std::to_string(sseq) +
                      " tokens in one pass, counting every multiply in the "
                      "layer." +
                      seedNote(sseq) + prov;
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


  // Everything that has to be true before a variant is worth timing, in one
  // place: the tiny gemm probe's verdict for the datatype, the opset the
  // runtime can parse, the memory the fixed geometry needs, and -- for the
  // quantized forms -- one profiled session proving the provider fused
  // rather than dequantizing 101 MB of weights on every run.  Sets
  // vr.usable, or vr.skipReason saying which of them failed.  Three call
  // sites (both prefill units and decode) asked the same four questions.
  auto validateVariant = [&](const Variant &v, VariantResult &vr) -> bool
  {
    if (vr.usable || !vr.skipReason.empty())
      return vr.usable; // already decided by an earlier scope

    // A graph this provider crashes on rather than declines: asked first,
    // because the fusion probe below is the session that dies.
    if (std::string why = variantFence(v, ep); !why.empty())
    {
      CLPEAK_VLOG("onnx-block[%s/%s]: not built: %s\n", ep.providerKey.c_str(),
                  v.label, why.c_str());
      vr.skipReason = why;
      return false;
    }

    // Gemm's 32^3 probe already knows if this dtype is unsupported here, and
    // answering from it costs nothing where the block's 50M-weight model
    // would cost a compile.
    {
      const auto &probe = onnxProbeGemmCache(rt, ep);
      auto it = probe.find(v.label);
      if (it != probe.end() && !it->second.ok)
      {
        vr.skipReason = it->second.reason;
        return false;
      }
    }

    // Naming the datatype that raised the opset turns "IR version 9 is not
    // supported" into a sentence about the row.
    {
      std::string why = onnxDtypeUnsupportedReason(rt, v.wDtype);
      if (why.empty())
        why = onnxDtypeUnsupportedReason(rt, v.actDtype);
      if (why.empty())
      {
        const int opset = variantOpset(v);
        const uint32_t needApi = onnxMinOrtApiForOpset(opset);
        if (needApi && rt.apiVersion < needApi)
          why = "needs opset " + std::to_string(opset) +
                ", which arrived in ONNX Runtime 1." + std::to_string(needApi) +
                "; this runtime is " + rt.versionString;
      }
      if (!why.empty())
      {
        vr.skipReason = why;
        return false;
      }
    }

    // The geometry is fixed, so this test cannot shrink to fit: a device too
    // small reports unsupported rather than measuring a smaller layer under
    // the same name.
    {
      const uint64_t needed = weightBytes(v) + kvBytes(v, contextsFor(v).back()) +
                              (64ull << 20) * onnxElemBytes(v.actDtype, 1) / 2;
      const uint64_t budget = clpeak::memoryBudget(~0ull, 8);
      if (budget && budget < needed)
      {
        CLPEAK_VLOG("onnx-block[%s/%s]: needs %llu MB, budget %llu MB\n",
                    ep.providerKey.c_str(), v.label,
                    (unsigned long long)(needed >> 20),
                    (unsigned long long)(budget >> 20));
        vr.skipReason = "not enough memory for the canonical block; its "
                        "geometry is fixed so the numbers stay comparable, and "
                        "a smaller layer would not be the same test";
        return false;
      }
    }

    if (v.qdq || v.wBlock > 0 || v.kvDtype)
    {
      std::string tried, firstErr;
      const size_t expected = v.kvDtype ? kAttentionMatmuls : kProjections;
      size_t triedFused = 0;
      Scheme schemes[2];
      const size_t nSchemes = v.qdq ? qdqSchemesFor(v, schemes) : 1;
      for (size_t si = 0; si < nSchemes; si++)
      {
        if (clpeak::cancelRequested())
          break;
        const int qAct = v.qdq ? schemes[si].actDtype : ONNX_DT_INT8;
        const char *what = v.qdq ? schemes[si].name
                                 : (v.kvDtype ? "quantized cache" : "weight-only");
        // A cache-format row has to be probed in the regime that has a cache.
        BlockRun probe = makeRun(rt, ep, v, v.decodeOnly, kKvLadder[0],
                                 kPromptLadder[0], qAct, true);
        if (!probe.session)
        {
          if (firstErr.empty())
            firstErr = probe.error;
          CLPEAK_VLOG("onnx-block[%s/%s]: %s rejected: %s\n",
                      ep.providerKey.c_str(), v.label, what, probe.error.c_str());
          continue;
        }
        timeRuns(rt, probe, 1);
        auto ops = onnxCollectExecutedOps(rt, probe.session);
        destroyRun(rt, probe);
        const std::string joined = onnxJoinOps(ops);
        CLPEAK_VLOG("onnx-block[%s/%s]: %s executed %s\n",
                    ep.providerKey.c_str(), v.label, what, joined.c_str());
        const size_t fused = quantizedMatmulsFused(ops, expected);
        if (fused < expected)
          CLPEAK_VLOG("onnx-block[%s/%s]: %s fused %zu of %zu quantized "
                      "matmuls\n", ep.providerKey.c_str(), v.label, what,
                      fused, expected);
        if (fused == expected)
        {
          vr.qActDtype = qAct;
          vr.schemeName = v.qdq ? schemes[si].name : "";
          if (!v.qdq)
            vr.castedActs = onnxCountOp(ops, "Cast") > 0;
          const std::string named = onnxQuantizedKernelName(ops);
          vr.ranAs = named.empty() ? "a kernel it compiled itself" : named;
          break;
        }
        if (!ops.empty())
        {
          tried = joined;
          triedFused = fused;
        }
      }
      if (vr.ranAs.empty())
      {
        const std::string what = v.kvDtype ? "cache" : "weights";
        const std::string how =
            triedFused == 0
                ? "provider did not fuse a quantized matmul -- it dequantized "
                  "the " + what
                : "provider fused only " + std::to_string(triedFused) +
                      " of the " + std::to_string(expected) +
                      " quantized matmuls -- it dequantized the rest of the " +
                      what;
        vr.skipReason =
            tried.empty()
                ? (firstErr.empty()
                       ? std::string("this provider accepted no session for ") + v.label
                       : firstErr)
                : how + " to full width and multiplied in floating point, a "
                        "complete pass over them on every run, so this is not "
                        "a " + v.label + " rate (ran: " + tried + ")";
        return false;
      }
    }

    vr.usable = true;
    return true;
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

      if (!validateVariant(v, vr))
      {
        emitPrefillTo(testFlops, v, vr);
        continue;
      }
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
      // Seed the decode affordability gate from the prefill rate, so the
      // first decode point of the run is not measured blind.  It has to be a
      // *rate* -- flops per microsecond -- and both regimes report one, which
      // is what makes the seed transferable.  Dividing decode's flops by
      // prefill's duration instead yields a "rate" that predicts every decode
      // token will take as long as a whole 512-token prompt, and a provider
      // whose prefill overran the budget then had its millisecond-scale decode
      // rows skipped as too slow to measure.
      if (auto it = vr.prefill.find(kPrefillSeq);
          it != vr.prefill.end() && it->second.us > 0.0)
      {
        refDecodeRate = blockFlops(kPrefillSeq, kPrefillSeq) / it->second.us;
        // The float rows, split by the provider's own profiler where it has
        // one: what the int8 row's parts cost at full width.
        if (!v.qdq && v.wBlock == 0 && v.kvDtype == 0)
          logPrefillProfile(rt, ep, v, vr.qActDtype, warmupCount, "");
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
      if (!validateVariant(v, vr))
      {
        emitPrefillTo(testOps, v, vr);
        continue;
      }
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

      // The layer's floating-point parts again, in fp32.  On a GPU an int8
      // layer ships as int8 projections inside fp16 attention, and there that
      // is the faster form: TensorRT on an RTX 5060 read 93 TOPS against 79
      // with the float parts in fp32.  On a CPU, where W8A8 models ship with
      // fp32 float parts, it is the other way round: ONNX Runtime's CPU
      // provider read 0.73 TOPS against 0.33 on a Zen 2 Threadripper and 1.25
      // against 1.18 on an M1 Pro.  No single width is right, so the prompt
      // takes whichever this provider runs faster, and says which.  Both forms keep attention in floating point, the fp32 one
      // because `create` holds QDQ propagation off.  Decode stays fp16: it is
      // memory-bound, and its row counts the bytes the cache declares.
      //
      // The fp32 form is proven fused on its own (validateVariant) before it
      // is timed: unfused it would be float arithmetic, which on the x86 CPU
      // provider runs faster than its fused int8 and would win the race
      // under the int8 label.
      if (v.qdq && v.wDtype == ONNX_DT_INT8 && v.actDtype != ONNX_DT_FLOAT &&
          !clpeak::cancelRequested())
      {
        auto it = vr.prefill.find(kPrefillSeq);
        if (it != vr.prefill.end() && it->second.us > 0.0)
        {
          Variant v32 = v;
          v32.actDtype = ONNX_DT_FLOAT;
          VariantResult vr32;
          if (validateVariant(v32, vr32))
          {
            Point pt32;
            pt32.us = measure(rt, ep, v32, false, vr32.qActDtype, warmupCount,
                              forceIters, specifiedIters, pt32.error,
                              pt32.status, kDecodeKv, kPrefillSeq);
            CLPEAK_VLOG("onnx-block[%s/%s]: prefill_s%lld %.1f us with fp16 "
                        "float parts, %.1f us with fp32\n",
                        ep.providerKey.c_str(), v.label,
                        (long long)kPrefillSeq, it->second.us, pt32.us);
            if (pt32.us > 0.0 && pt32.us < it->second.us)
            {
              vr.prefillFp32 = true;
              vr.prefillFp16Us = it->second.us;
              vr.prefillProv = provenance(v32, vr32);
              it->second = pt32;
            }
            logPrefillProfile(rt, ep, v32, vr32.qActDtype, warmupCount,
                              " fp32 float parts");
          }
          else
            CLPEAK_VLOG("onnx-block[%s/%s]: fp32 float parts not measured: "
                        "%s\n",
                        ep.providerKey.c_str(), v.label,
                        vr32.skipReason.c_str());
          logPrefillProfile(rt, ep, v, vr.qActDtype, warmupCount,
                            " fp16 float parts");
        }
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
      uint64_t wBytes = weightBytes(v);
      uint64_t kvB = kvBytes(v, kDecodeKv);
      // fp32 tensors on a provider proven to hold them at half width move
      // two bytes an element, and that is what the row is credited with.
      const bool halfWidth =
          fp32Narrowed && v.wBlock == 0 &&
          (v.wDtype == ONNX_DT_FLOAT ||
           (v.kvDtype ? v.kvDtype : v.actDtype) == ONNX_DT_FLOAT);
      if (halfWidth && v.wDtype == ONNX_DT_FLOAT)
        wBytes /= 2;
      if (halfWidth && (v.kvDtype ? v.kvDtype : v.actDtype) == ONNX_DT_FLOAT)
        kvB /= 2;
      const double bytes = (double)(wBytes + kvB);
      logger::EmitOptions o;
      o.description = std::string(v.note) + "  One token with 2048 of context: " + std::to_string((unsigned long long)(wBytes >> 20)) + " MB of weights plus " + std::to_string((unsigned long long)(kvB >> 20)) + " MB of cached context, read in full for one token." + provenance(v, vr);
      if (halfWidth)
        o.description += "  This provider holds fp32 tensors at 16 bits -- its "
                         "fp32 results carry half-precision error, and it "
                         "streams fp32 elements as fast as fp16 ones -- so the "
                         "count above is the two bytes an element it moves, "
                         "not the four fp32 declares.";

      // Prefill settled most variants already; validateVariant answers from
      // what it recorded and only probes the ones it has not seen -- the
      // cache-format row, whose prefill has no reading to settle it.
      if (!validateVariant(v, vr))
      {
        test.skip(metric, vr.skipStatus, vr.skipReason, o);
        continue;
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
        if (it->second.us > 0.0)
        {
          const double bps = bytes / (it->second.us * 1.0e-6);
          // More traffic than this provider has been seen to move means it is
          // not moving what the format declares -- it stored the weights
          // narrower than asked, which several accelerators do to an fp32
          // graph by default.  Measured, so it holds for any provider.
          if (streamBps > 0.0 && bps > streamBps)
            o.description += "  This is more than the " +
                             std::to_string((long long)(streamBps / 1.0e9)) +
                             " GB/s this provider was measured streaming, so "
                             "it is not moving the bytes this precision "
                             "declares -- it is storing them narrower.";

          // The cheaper and far more sensitive version of the same question,
          // and it needs no second measurement: this row and the fp16
          // reference ran the identical layer and differ only in the width
          // they declare.  If one declares half again as many bytes as the
          // other and takes the same time to move them, the byte count is not
          // describing the traffic -- and which of the two is wrong depends on
          // the provider, so the row says what was seen rather than guessing.
          //
          // OpenVINO's GPU serves an fp32 graph at 16 bits, so its fp32 row
          // declares 235 MB against fp16's 117 and reads 104 GB/s against
          // 52.6 -- off the same 2.2 ms.  ONNX Runtime's x86 CPU EP has the
          // opposite fault, converting fp16 up, and lands in exactly the same
          // place from the other side.  The absolute check above misses both:
          // 104 GB/s is comfortably under what that device streams.
          if (refDecodeUs > 0.0 && refDecodeBytes > 0.0 && vi != kRefVariant)
          {
            const double byteRatio = bytes / refDecodeBytes;
            const double timeRatio = it->second.us / refDecodeUs;
            if (byteRatio >= 1.5 && timeRatio > 0.9 && timeRatio < 1.25)
              o.description += "  It declares " +
                               std::to_string((long long)(byteRatio * 10) / 10) +
                               "." +
                               std::to_string((long long)(byteRatio * 10) % 10) +
                               " times the bytes of the fp16 row and took the "
                               "same time, so one of the two is not moving "
                               "what it declares; the fp32 numeric-error row "
                               "says whether this provider computes fp32 at "
                               "full width.";
          }
          if (vi == kRefVariant)
          {
            refDecodeUs = it->second.us;
            refDecodeBytes = bytes;
          }
          test.emit(metric, (float)bps, o);
        }
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
        const std::string note = std::string("One pass over a 512-token prompt.  ") + v.note +
                                 seedNote(kPrefillSeq) +
                                 prefillProvenance(v, vr, kPrefillSeq);
        if (!vr.usable && !vr.skipReason.empty()) { test.skip(metric, vr.skipStatus, vr.skipReason, note); }
        else
        {
          // The prompt test timed this pass or said why not -- too slow to
          // measure, or a failure -- and its answer is this row's too.
          // Timing it again here, as this did, spent passes the prompt test
          // had judged unaffordable: 57, 343 and 413 s on the x64 QNN plugin,
          // which ran on a host backend.  Only a pass the prompt ladder never
          // reached is measured here, under the same budget.
          auto it = vr.prefill.find(kPrefillSeq);
          if (it == vr.prefill.end() && vr.usable)
          {
            Point pt;
            const double flops = blockFlops(kPrefillSeq, kPrefillSeq);
            if (refPrefillRate > 0.0 && flops / refPrefillRate > (double)kBlockBudgetUs)
            {
              pt.status = ResultStatus::Error;
              pt.error = "one pass would take about " +
                         std::to_string((long long)(flops / refPrefillRate / 1.0e6)) +
                         " s on this provider, too slow to measure";
            }
            else
              pt.us = measure(rt, ep, v, false, vr.qActDtype, warmupCount, forceIters,
                              specifiedIters, pt.error, pt.status, kDecodeKv, kPrefillSeq);
            vr.prefill[kPrefillSeq] = pt;
            it = vr.prefill.find(kPrefillSeq);
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
        const std::string note = "One generated token with " + std::to_string(kv) + " tokens of context behind it.  " + v.note + prov;
        if (!vr.usable && !vr.skipReason.empty()) { test.skip(metric, vr.skipStatus, vr.skipReason, note); continue; }
        // As for the prompt: a context the decode test answered keeps its
        // answer, and only the lengths it does not measure are timed here.
        auto it = vr.decode.find(kv);
        if (it == vr.decode.end())
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
