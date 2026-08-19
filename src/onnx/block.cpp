#ifdef ENABLE_ONNX

// onnx-block: one fixed transformer decoder block, run in the two regimes
// that bound all LLM inference.  This is the rung above a raw GEMM peak and
// below tokens/second: an intermediate number that is meaningful, comparable
// across completely different hardware, and needs no model download.
//
//   prefill (512 tokens at once) is compute-bound   -> effective TFLOPS
//   decode  (1 token, 2048 of context) is memory-bound -> effective GB/s
//
// Both numbers come out of the whole stack -- graph scheduling, layout
// conversions, softmax, the lot -- not just the matmuls, so they are what a
// real pipeline can actually reach rather than what the silicon could do in
// principle.  Multiply the latency rows by a model's layer count to sanity
// check any tokens/second claim made for this device.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <chrono>
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

// fp16 only.  Nobody serves an LLM in fp32, so a full-precision block would
// measure a configuration that does not exist -- and it would double both the
// model size and the run time to say so.
constexpr int kDtype = ONNX_DT_FLOAT16;

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

BlockRun makeRun(const OrtRuntime &rt, const onnx_ep_info_t &ep, bool decode)
{
  BlockRun r;

  OnnxBlockShape sh;
  sh.dModel    = kDModel;
  sh.heads     = kHeads;
  sh.headDim   = kHeadDim;
  sh.ffnHidden = kFfnHidden;
  sh.seq       = decode ? 1 : kPrefillSeq;
  sh.kvLen     = decode ? kDecodeKv : 0;

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

  const int64_t S = sh.seq;
  r.inBuf.assign((size_t)S * kDModel * 2, 0);
  {
    uint16_t *h = reinterpret_cast<uint16_t *>(r.inBuf.data());
    uint32_t s = 0x9e3779b9u;
    for (int64_t i = 0; i < S * kDModel; i++)
    {
      s ^= s << 13; s ^= s >> 17; s ^= s << 5;
      h[i] = floatToHalf((float)(s >> 8) / 16777216.0f - 0.5f);
    }
  }

  // Decode also returns the new K/V (the cache write); let ORT allocate
  // those, so only the input needs a bound buffer here.
  r.outNames.push_back("Y");
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
  {
    const int64_t shape[2] = {S, kDModel};
    st = rt.api->CreateTensorWithDataAsOrtValue(
        mi, r.inBuf.data(), r.inBuf.size(), shape, 2,
        (ONNXTensorElementDataType)kDtype, &r.inVal);
  }
  if (mi) rt.api->ReleaseMemoryInfo(mi);
  if (st)
  {
    r.error = onnxStatusText(rt, st);
    destroyRun(rt, r);
  }
  return r;
}

// Mean microseconds per block; negative on failure.
double timeRuns(const OrtRuntime &rt, BlockRun &r, unsigned int n)
{
  static const char *inNames[] = {"X"};

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
               std::string &error, ResultStatus &status)
{
  BlockRun r = makeRun(rt, ep, decode);
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

  unsigned int iters = pickIters(per_iter_us, 5000000u, forceIters ? forced : 0);
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

  const char *geometry =
      "One 2048-wide, 16-head decoder block with a SwiGLU feed-forward "
      "(50.6M parameters, 101 MB of fp16 weights).  ";

  // Both regimes are measured once; the three scopes below are three ways of
  // reading the same two timings.
  std::string prefillErr, decodeErr;
  ResultStatus prefillStatus = ResultStatus::Ok, decodeStatus = ResultStatus::Ok;

  double prefillUs = -1.0, decodeUs = -1.0;

  if (!clpeak::cancelRequested())
    prefillUs = measure(rt, ep, false, warmupCount, forceIters, specifiedIters,
                        prefillErr, prefillStatus);
  if (!clpeak::cancelRequested())
    decodeUs = measure(rt, ep, true, warmupCount, forceIters, specifiedIters,
                       decodeErr, decodeStatus);

  // ---- Prefill: compute-bound, so report the rate it sustains ------------
  {
    auto test = currentDeviceScope->beginTest(
        {"onnx-block-prefill", "Transformer block, prefill", "tflops",
         Category::Ai,
         "Speed of one whole transformer layer while it is chewing through a "
         "prompt -- 512 tokens in a single pass.  This is the phase that "
         "decides how long you wait before the first word appears.  Unlike "
         "the raw matmul rows, everything a real layer does is in here: "
         "attention, softmax, the feed-forward network and the data "
         "shuffling between them, so it is what a device actually delivers "
         "rather than what its silicon could do in principle."});

    std::string note = std::string(geometry) +
        "512 tokens at once, counting every multiply in the layer.";
    if (prefillUs > 0.0)
      test.emit("s512", (float)(blockFlops(kPrefillSeq, kPrefillSeq) * 1.0e6 /
                                prefillUs / 1.0e12), note.c_str());
    else
      test.skip("s512", prefillStatus, prefillErr, note);
    test.end();
  }

  // ---- Decode: memory-bound, so report how fast bytes move ---------------
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
      test.skip("kv2048", decodeStatus, decodeErr, note);
    test.end();
  }

  // ---- The same two timings as latency, the per-layer TTFT/TPOT analog ---
  {
    auto test = currentDeviceScope->beginTest(
        {"onnx-block-latency", "Transformer block latency", "us",
         Category::Ai,
         "How long one layer takes, in microseconds.  Multiply by a model's "
         "layer count for a floor on that model's time-to-first-token and "
         "per-token time on this device -- a 32-layer 7B model runs 32 of "
         "these back to back per token.  It is the honest way to check a "
         "tokens-per-second claim without downloading anything."});

    if (prefillUs > 0.0)
      test.emit("prefill_s512", (float)prefillUs,
                "One pass over a 512-token prompt.");
    else
      test.skip("prefill_s512", prefillStatus, prefillErr,
                "One pass over a 512-token prompt.");

    if (decodeUs > 0.0)
      test.emit("decode_kv2048", (float)decodeUs,
                "One generated token with 2048 tokens of context.");
    else
      test.skip("decode_kv2048", decodeStatus, decodeErr,
                "One generated token with 2048 tokens of context.");
    test.end();
  }

  return 0;
}

#endif // ENABLE_ONNX
