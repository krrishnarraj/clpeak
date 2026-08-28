#ifdef ENABLE_ONNX

// onnx-block: one fixed transformer decoder block, run in the two regimes
// that bound all LLM inference.  This is the rung above a raw GEMM peak and
// below tokens/second: an intermediate number that is meaningful, comparable
// across completely different hardware, and needs no model download.
//
//   prefill (64/512/2048 tokens at once)      compute-bound -> effective TFLOPS
//   decode  (1 token, 512/2048/8192 context)  memory-bound  -> effective GB/s
//
// Both numbers come out of the whole stack -- graph scheduling, layout
// conversions, softmax, the lot -- not just the matmuls, so they are what a
// real pipeline can actually reach rather than what the silicon could do in
// principle.  Multiply the latency rows by a model's layer count to sanity
// check any tokens/second claim made for this device.
//
// Six timings, three scopes, one unit each, and nothing restated: the prompt
// ladder is onnx-block-prefill (tflops), the 2048-context token is
// onnx-block-decode (gbps) because that is the row onnx-tensor-bw compares
// against, and every timing that is worth reading as a duration lands in
// onnx-block-latency (us).  Splitting a ladder's headline rung into a scope of
// its own is what to avoid here: the rung and the scope hold the same timing
// in the same unit, so one of the two rows is pure repetition.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <chrono>
#include <map>
#include <cmath>
#include <cstring>
#include <vector>

namespace
{

// Fixed geometry: llama-style proportions (SwiGLU hidden = 2.6875x d_model,
// 128-wide heads), sized so the weights -- 50.6M parameters, 101 MB at fp16
// -- overflow every cache on every device while still compiling in seconds
// on the NPU toolchains, which build graphs ahead of time.  A 7B block would
// be 4x this and push AOT compile times into minutes for no extra insight.
constexpr int64_t kDModel    = 2048;
constexpr int64_t kHeads     = 16;
constexpr int64_t kHeadDim   = 128;
constexpr int64_t kFfnHidden = 5504;
constexpr int64_t kPrefillSeq = 512;
constexpr int64_t kDecodeKv   = 2048;

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

// fp16 only.  Nobody serves an LLM in fp32, so a full-precision block would
// measure a configuration that does not exist -- and it would double both the
// model size and the run time to say so.
constexpr int kDtype = ONNX_DT_FLOAT16;

// Budget for one point's timed phase.  It doubles as the affordability test:
// a point whose single iteration costs more than the entire budget for
// measuring it cannot be measured properly anyway, so it is skipped.
constexpr unsigned int kBlockBudgetUs = 5000000;

int64_t weightParams()
{
  return 4 * kDModel * kDModel          // Wq, Wk, Wv, Wo
       + 2 * kDModel * kFfnHidden       // Wg, Wu
       + kFfnHidden * kDModel;          // Wd
}

// Multiply-accumulates x2, over every matmul in the block.
double blockFlops(int64_t seq, int64_t ctx)
{
  const double S = (double)seq, C = (double)ctx;
  const double d = (double)kDModel, ffn = (double)kFfnHidden;
  const double H = (double)kHeads, Dh = (double)kHeadDim;

  const double qkv  = 3.0 * 2.0 * S * d * d;
  const double attn = 2.0 * 2.0 * H * S * C * Dh;   // scores + context
  const double proj = 2.0 * S * d * d;
  const double ff   = 2.0 * 2.0 * S * d * ffn + 2.0 * S * ffn * d;
  return qkv + attn + proj + ff;
}

struct BlockRun
{
  OrtSession *session = nullptr;
  OrtValue   *inVal   = nullptr;
  std::vector<OrtValue *> outVals;
  std::vector<const char *> outNames;
  std::vector<uint8_t> inBuf;
  std::string error;
};

void destroyRun(const OrtRuntime &rt, BlockRun &r)
{
  if (r.inVal) rt.api->ReleaseValue(r.inVal);
  for (OrtValue *v : r.outVals)
    if (v) rt.api->ReleaseValue(v);
  if (r.session) rt.api->ReleaseSession(r.session);
  r.inVal = nullptr;
  r.outVals.clear();
  r.session = nullptr;
  r.inBuf.clear();
  r.inBuf.shrink_to_fit();
  // `error` survives: callers tear a failed run down and then report it.
}

BlockRun makeRun(const OrtRuntime &rt, const onnx_ep_info_t &ep, bool decode,
                 int64_t kvLen = kDecodeKv, int64_t prefillSeq = kPrefillSeq)
{
  BlockRun r;

  OnnxBlockShape sh;
  sh.dModel    = kDModel;
  sh.heads     = kHeads;
  sh.headDim   = kHeadDim;
  sh.ffnHidden = kFfnHidden;
  sh.seq       = decode ? 1 : prefillSeq;
  sh.kvLen     = decode ? kvLen : 0;

  {
    std::string model = onnxBlockModel(sh);
    auto ses = onnxCreateSession(rt, ep, model);
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
  r.inBuf.assign(2, 0);
  {
    uint16_t v = floatToHalf(1.0009765625f);
    std::memcpy(r.inBuf.data(), &v, 2);
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
        (ONNXTensorElementDataType)kDtype, &r.inVal);
  if (mi) rt.api->ReleaseMemoryInfo(mi);
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
      if (v) rt.api->ReleaseValue(v);
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
double measure(const OrtRuntime &rt, const onnx_ep_info_t &ep, bool decode,
               unsigned int warmup, bool forceIters, unsigned int forced,
               std::string &error, ResultStatus &status,
               int64_t kvLen = kDecodeKv, int64_t prefillSeq = kPrefillSeq)
{
  BlockRun r = makeRun(rt, ep, decode, kvLen, prefillSeq);
  if (!r.session)
  {
    error  = r.error;
    status = ResultStatus::Unsupported;
    return -1.0;
  }

  double per_iter_us = -1.0;
  if (timeRuns(rt, r, 1 + warmup) > 0.0)      // graph compile + warmup
    per_iter_us = timeRuns(rt, r, 3);         // calibration probe
  if (per_iter_us <= 0.0)
  {
    error  = r.error.empty() ? "run failed" : r.error;
    status = ResultStatus::Error;
    destroyRun(rt, r);
    return -1.0;
  }

  unsigned int iters = pickIters(per_iter_us, kBlockBudgetUs,
                                 forceIters ? forced : 0);
  double mean_us = timeRuns(rt, r, iters);
  if (mean_us <= 0.0)
  {
    error  = r.error.empty() ? "run failed" : r.error;
    status = ResultStatus::Error;
  }
  destroyRun(rt, r);
  return mean_us;
}

} // namespace

int OnnxPeak::runBlock(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                       benchmark_config_t &cfg)
{
  (void)cfg;

  // The one test here whose size cannot adapt: the geometry defines the
  // workload, so a small device has to be told it cannot run it rather than
  // handed a smaller layer whose numbers would not compare with anyone's.
  // The runtime keeps its own copy of the weights alongside ours, and the
  // longest context adds a 67 MB cache, so the peak is several times the
  // 101 MB on paper.
  {
    const uint64_t needed = (uint64_t)weightParams() * 2ull   // weights
                          + (67ull << 20)                     // longest KV cache
                          + (64ull << 20);                    // activations, slack
    const uint64_t budget = clpeak::memoryBudget(~0ull, 8);
    if (budget && budget < needed)
    {
      const char *reason = "not enough memory for the canonical block; its "
                           "geometry is fixed so the numbers stay comparable, "
                           "and a smaller layer would not be the same test";

      // Every scope still reports, so a reader sees why the rows are missing
      // rather than finding a gap where a test should have been.
      struct Scope { const char *tag, *display, *unit; };
      static const Scope kScopes[] = {
        {"onnx-block-prefill",     "Transformer block, prefill",           "tflops"},
        {"onnx-block-decode",      "Transformer block, decode",            "gbps"},
        {"onnx-block-latency",     "Transformer block latency",            "us"},
      };
      for (const Scope &sc : kScopes)
      {
        auto t = currentDeviceScope->beginTest(
            {sc.tag, sc.display, sc.unit, Category::Ai,
             "One whole transformer layer, in the regimes that bound "
             "language-model inference."});
        t.skip("unavailable", ResultStatus::Unsupported, reason);
        t.end();
      }

      CLPEAK_VLOG("onnx-block[%s]: needs %llu MB, budget %llu MB\n",
                  ep.providerKey.c_str(),
                  (unsigned long long)(needed >> 20),
                  (unsigned long long)(budget >> 20));
      return 0;
    }
  }

  const char *geometry =
      "One 2048-wide, 16-head decoder block with a SwiGLU feed-forward "
      "(50.6M parameters, 101 MB of fp16 weights).  ";

  // Both ladders are measured here, cheapest point first, and every scope
  // below reads from the results.  Ascending order is what makes the run
  // affordable: the first point establishes how fast this provider is, and
  // each later one is skipped when a single iteration of it would cost more
  // than the whole budget for measuring it -- the rule onnx-gemm and
  // onnx-conv already use, and which the block ladders were missing.  Without
  // it a provider that turns out to be very slow drags the block out for
  // tens of minutes, which is exactly what a stock CPU build of ONNX Runtime
  // 1.30 does with fp16.
  struct Point
  {
    double       us = -1.0;
    std::string  error;
    ResultStatus status = ResultStatus::Ok;
  };

  auto affordable = [&](double flops, double rate) {
    // rate is FLOP per microsecond from the previous, cheaper point.
    return rate <= 0.0 || (flops / rate) <= (double)kBlockBudgetUs;
  };

  std::map<int64_t, Point> prefill, decode;
  double prefillRate = 0.0, decodeRate = 0.0;

  for (int64_t seq : kPromptLadder)
  {
    if (clpeak::cancelRequested())
      break;
    Point pt;
    const double flops = blockFlops(seq, seq);
    if (!affordable(flops, prefillRate))
    {
      pt.status = ResultStatus::Error;
      pt.error  = "one pass would take about " +
                  std::to_string((long long)(flops / prefillRate / 1.0e6)) +
                  " s on this provider, too slow to measure";
      CLPEAK_VLOG("onnx-block[%s]: skipping prefill %lld, %s\n",
                  ep.providerKey.c_str(), (long long)seq, pt.error.c_str());
    }
    else
    {
      pt.us = measure(rt, ep, false, warmupCount, forceIters, specifiedIters,
                      pt.error, pt.status, kDecodeKv, seq);
      if (pt.us > 0.0)
        prefillRate = flops / pt.us;
    }
    prefill[seq] = pt;
  }

  // Decode work barely grows with context -- the weights dominate -- so the
  // rate from the shortest context predicts the rest closely enough to gate on.
  for (int64_t kv : kKvLadder)
  {
    if (clpeak::cancelRequested())
      break;
    Point pt;
    const double flops = blockFlops(1, kv);
    if (!affordable(flops, decodeRate))
    {
      pt.status = ResultStatus::Error;
      pt.error  = "one token would take about " +
                  std::to_string((long long)(flops / decodeRate / 1.0e6)) +
                  " s on this provider, too slow to measure";
      CLPEAK_VLOG("onnx-block[%s]: skipping decode kv%lld, %s\n",
                  ep.providerKey.c_str(), (long long)kv, pt.error.c_str());
    }
    else
    {
      pt.us = measure(rt, ep, true, warmupCount, forceIters, specifiedIters,
                      pt.error, pt.status, kv);
      if (pt.us > 0.0)
        decodeRate = flops / pt.us;
    }
    decode[kv] = pt;
  }

  const double decodeUs = decode.count(kDecodeKv) ? decode[kDecodeKv].us : -1.0;

  // ---- Prefill: compute-bound, so report the rate it sustains ------------
  {
    auto test = currentDeviceScope->beginTest(
        {"onnx-block-prefill", "Transformer block, prefill", "tflops",
         Category::Ai,
         "Speed of one whole transformer layer while it is chewing through a "
         "prompt, at three prompt lengths.  This is the phase that decides "
         "how long you wait before the first word appears.  Unlike the raw "
         "matmul rows, everything a real layer does is in here: attention, "
         "softmax, the feed-forward network and the data shuffling between "
         "them, so it is what a device actually delivers rather than what its "
         "silicon could do in principle.  A short prompt cannot fill wide "
         "hardware, so the rate climbs with the prompt and then flattens; "
         "where it flattens is how much text has to arrive together before "
         "batching requests stops helping."});

    for (int64_t seq : kPromptLadder)
    {
      if (clpeak::cancelRequested())
        break;

      const std::string metric = "s" + std::to_string(seq);
      const std::string note =
          std::string(geometry) + "A prompt of " + std::to_string(seq) +
          " tokens in one pass, counting every multiply in the layer.";

      const Point &pt = prefill[seq];
      if (pt.us > 0.0)
        test.emit(metric,
                  (float)(blockFlops(seq, seq) * 1.0e6 / pt.us / 1.0e12),
                  note.c_str());
      else
        test.skip(metric, pt.status, pt.error, note);
    }
    test.end();
  }

  // ---- Decode: memory-bound, so report how fast bytes move ---------------
  //
  // One rung, in GB/s, because this is the row that compares directly with
  // onnx-tensor-bw: how much of the device's raw streaming rate a complete
  // layer manages to keep.  How the cost grows with context is a latency
  // question and is answered in the latency scope below.
  {
    auto test = currentDeviceScope->beginTest(
        {"onnx-block-decode", "Transformer block, decode", "gbps",
         Category::Ai,
         "How fast the same layer streams its weights while generating one "
         "token with 2048 tokens of context behind it.  Generating text one "
         "token at a time is limited by memory, not arithmetic: every weight "
         "has to be read to produce a single word.  This is the number that "
         "actually sets how quickly words appear, and comparing it with the "
         "device's plain memory-bandwidth rows shows how much of that "
         "bandwidth the AI stack manages to use."});

    const double weightBytes = (double)weightParams() * 2.0;
    const double kvBytes     = 2.0 * (double)kHeads * (double)kDecodeKv *
                               (double)kHeadDim * 2.0;
    std::string note = std::string(geometry) +
        "One token with 2048 of context: 101 MB of weights plus 16 MB of "
        "cached context, all of which must be read to emit a single token.";
    if (decodeUs > 0.0)
      test.emit("kv2048",
                (float)((weightBytes + kvBytes) / (decodeUs * 1.0e-6) / 1.0e9),
                note.c_str());
    else
    {
      const Point &pt = decode[kDecodeKv];
      test.skip("kv2048", pt.status, pt.error, note);
    }
    test.end();
  }

  // ---- The same timings as latency, the per-layer TTFT/TPOT analog -------
  //
  // Both ladders in microseconds in one place: a prefill pass, and a decode
  // token at each context length.  Everything except attention costs the same
  // at every context length -- the weights are the weights -- so whatever the
  // decode rows add as the context grows is attention, and it is why a long
  // conversation answers more slowly than a short one.
  {
    auto test = currentDeviceScope->beginTest(
        {"onnx-block-latency", "Transformer block latency", "us",
         Category::Ai,
         "How long one layer takes, in microseconds.  Multiply by a model's "
         "layer count for a floor on that model's time-to-first-token and "
         "per-token time on this device -- a 32-layer 7B model runs 32 of "
         "these back to back per token.  It is the honest way to check a "
         "tokens-per-second claim without downloading anything.  The decode "
         "rows also show what a longer conversation costs: everything but "
         "attention takes the same time at every context length, so whatever "
         "they add as the context grows is attention.  A device whose time "
         "barely moves is reading its cached context efficiently; one that "
         "climbs steeply will feel fine in a demo and poor in use."});

    {
      const Point &pt = prefill[kPrefillSeq];
      const char *note = "One pass over a 512-token prompt.";
      if (pt.us > 0.0)
        test.emit("prefill_s512", (float)pt.us, note);
      else
        test.skip("prefill_s512", pt.status, pt.error, note);
    }

    for (int64_t kv : kKvLadder)
    {
      if (clpeak::cancelRequested())
        break;

      const Point &pt = decode[kv];
      const std::string label = "decode_kv" + std::to_string(kv);
      const std::string note =
          "One generated token with " + std::to_string(kv) +
          " tokens of context behind it.";
      if (pt.us > 0.0)
        test.emit(label, (float)pt.us, note.c_str());
      else
      {
        test.skip(label, pt.status, pt.error.empty() ? "run failed" : pt.error,
                  note);
        break;   // longer contexts will not fare better
      }
    }
    test.end();
  }

  return 0;
}

#endif // ENABLE_ONNX
