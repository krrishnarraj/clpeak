#ifdef ENABLE_ONNX

// onnx-activation: how fast the operations *between* the matrix multiplies
// run.
//
// A transformer layer is mostly matmul by arithmetic and mostly other things
// by op count -- softmax, normalisation, the gate in a feed-forward.  None of
// them does meaningful arithmetic: they read a tensor and write one back, so
// their ceiling is memory bandwidth, and their rate is reported here as the
// bandwidth they achieve.  Compare it with onnx-tensor-bw: an accelerator
// that streams weights at hundreds of gigabytes a second but normalises at a
// fraction of that will spend a surprising share of a layer doing the cheap
// part, and these are also the operations most likely to be handed back to
// the CPU by a provider that does not implement them.
//
// Each rate is measured against a reference graph that reads and reduces the
// same constant with no operation applied.  Subtracting it leaves the
// operation's own cost rather than the cost of the scaffolding around it.
// The reference depends only on the tensor size, so it is timed once per size
// and shared by all three operations rather than re-timed for each.
//
// Every rung is reported, at the same three working-set sizes onnx-tensor-bw
// uses, so the two ladders divide row for row: silu_32mb against that test's
// 32mb rung is "what share of its streaming rate this provider keeps when it
// has to apply a function to the data".
//
// Reporting a single number per operation was tried and cannot be made
// honest, because there are two real regimes and no rule picks between them
// without lying about one.  Taking the *fastest* rung reports whichever one
// the subtraction over-credited most -- the reference cannot be made cheap
// (something must consume the result or the optimiser deletes the operation
// under test), and on TensorRT its cost scales with the tensor, so at 8 MB
// 79% of the time is reference and the rung reads 2.8x the next one down.
// Taking the *largest* rung instead reports whichever cliff the ladder
// happened to fall off: the stop rule climbs one rung past the peak on
// purpose, so the last rung measured is usually the collapsed one, and
// whether it is taken at all turns on a fraction of a percent of jitter at
// the rung before.  On this M1 Pro that made SiLU alternate between 12.0 and
// 4.1 GB/s run to run.  Both cliffs are real and both regimes are worth
// knowing; a ladder says so and a single number cannot.

#include <onnx/onnx_peak.h>
#include "onnx_model.h"
#include "onnx_session.h"

#include <chrono>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace
{

  // A transformer-shaped tensor: rows of model-width vectors.  The sizes are
  // fixed and named for their working set rather than swept, because the point
  // of the row is the comparison against onnx-tensor-bw's rung of the same name
  // -- a sweep that stopped in a different place per operation would compare
  // two different working sets against each other.  They are the same three
  // sizes that test always measures, for the same reason it always measures
  // them: 8 MB sits in fast local memory nearly everywhere, 128 MB does not.
  constexpr int64_t kCols = 4096;

  struct Size
  {
    int64_t rows; // bytes = rows * kCols * 2
    const char *label;
  };

  const Size kSizes[] = {
      {1024, "8mb"},
      {4096, "32mb"},
      {16384, "128mb"},
  };

  // The rungs are fixed, so the budget check is the only thing standing between
  // a phone and an out-of-memory kill -- Android ends the process rather than
  // failing an allocation, so a rung has to be declined on an estimate rather
  // than attempted and recovered from.  The estimate is three times the tensor,
  // not one: the raw values and the model embedding them are both alive while
  // the model is built, and the model and ORT's own copy are both alive while
  // the session is created.
  static uint64_t maxTensorBytes() { return clpeak::memoryBudget(1ull << 30); }
  constexpr uint64_t kCopiesAtPeak = 3;

  constexpr unsigned int kSizeBudgetUs = 1000000;

  struct Variant
  {
    OnnxActivation act;
    const char *label;
    const char *note;
  };

  const Variant kVariants[] = {
      {OnnxActivation::Silu, "silu",
       "x times sigmoid(x) -- the gate in the feed-forward network of most "
       "current language models."},
      {OnnxActivation::Softmax, "softmax",
       "Softmax across the row, the operation at the heart of attention.  It "
       "needs two passes over the data and a maximum before it can divide, which "
       "makes it far harder for fixed-function hardware than its cost suggests."},
      {OnnxActivation::LayerNorm, "layernorm",
       "Layer normalisation: mean and variance across each row, then rescale.  "
       "Every transformer layer does this at least twice."},
  };

  struct Run
  {
    double us = -1.0;
    std::string error;
    ResultStatus status = ResultStatus::Ok;
  };

  Run measure(const OrtRuntime &rt, const onnx_ep_info_t &ep,
              OnnxActivation act, int64_t rows,
              unsigned int warmup, bool forceIters, unsigned int forced)
  {
    Run r;

    OrtSession *session = nullptr;
    {
      std::string xRaw((size_t)rows * kCols * 2, '\0');
      {
        uint16_t *h = reinterpret_cast<uint16_t *>(&xRaw[0]);
        uint32_t s = 0x9e3779b9u;
        for (int64_t i = 0; i < rows * kCols; i++)
        {
          s ^= s << 13;
          s ^= s >> 17;
          s ^= s << 5;
          h[i] = floatToHalf((float)(s >> 8) / 16777216.0f - 0.5f);
        }
      }
      std::string model = onnxResidentActivationModel(rows, kCols, act, xRaw);
      xRaw.clear();
      xRaw.shrink_to_fit();

      auto ses = onnxCreateSession(rt, ep, model, /*keepConstantsUnfolded=*/true);
      model.clear();
      model.shrink_to_fit();
      if (!ses.session)
      {
        r.error = ses.error;
        r.status = ResultStatus::Unsupported;
        return r;
      }
      session = ses.session;
    }

    uint16_t sVal = floatToHalf(1.0009765625f);
    std::vector<uint16_t> outBuf((size_t)kCols, 0);

    OrtMemoryInfo *mi = nullptr;
    OrtValue *inVal = nullptr, *outVal = nullptr;
    OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                                OrtMemTypeDefault, &mi);
    const int64_t outShape[1] = {kCols};
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, &sVal, sizeof(sVal), nullptr, 0,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &inVal);
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, outBuf.data(), outBuf.size() * 2, outShape, 1,
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, &outVal);
    if (mi)
      rt.api->ReleaseMemoryInfo(mi);

    auto run = [&](unsigned int n) -> double
    {
      static const char *ins[] = {"S"};
      static const char *outs[] = {"Y"};
      auto a = std::chrono::steady_clock::now();
      for (unsigned int i = 0; i < n; i++)
      {
        OrtStatus *rs = rt.api->Run(session, nullptr, ins,
                                    (const OrtValue *const *)&inVal, 1,
                                    outs, 1, &outVal);
        if (rs)
        {
          r.error = onnxStatusText(rt, rs);
          return -1.0;
        }
      }
      return std::chrono::duration<double, std::micro>(
                 std::chrono::steady_clock::now() - a)
                 .count() /
             (double)n;
    };

    if (!st && run(1 + warmup) > 0.0)
    {
      double probe = run(1);
      if (probe > 0.0)
      {
        unsigned int iters = pickIters(probe, kSizeBudgetUs,
                                       forceIters ? forced : 0, kOnnxMaxIters);
        // The probe was one whole pass; when the budget affords only one, it
        // already is the measurement.
        r.us = (iters > 1) ? run(iters) : probe;
      }
    }
    if (st)
      r.error = onnxStatusText(rt, st);
    if (r.us <= 0.0 && r.status == ResultStatus::Ok)
      r.status = ResultStatus::Error;

    if (inVal)
      rt.api->ReleaseValue(inVal);
    if (outVal)
      rt.api->ReleaseValue(outVal);
    rt.api->ReleaseSession(session);
    return r;
  }

} // namespace

int OnnxPeak::runActivation(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                            benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"onnx_activation", "ONNX activation throughput", "bps",
       Category::Bandwidth,
       "How fast this provider runs the operations between the matrix "
       "multiplies -- normalisation, softmax, the feed-forward gate.  They do "
       "almost no arithmetic, so their limit is how fast data moves, and the "
       "figure here is the bandwidth each one achieves.  Held against the "
       "resident-tensor rows, it shows how much of a layer goes on the cheap "
       "parts: hardware built for matrix multiplication often runs these at a "
       "small fraction of its streaming speed, and they are also the "
       "operations a provider is most likely to hand back to the CPU.  Each "
       "row is net of a reference graph that reads the same tensor and "
       "applies nothing, and is measured at the same three working-set sizes "
       "as the resident-tensor rows, so the two divide row for row.  Where a "
       "row drops sharply against the one above it, the activations have "
       "outgrown the memory the provider keeps close.",
       // Three operations across three working-set sizes: nine separate
       // measurements, no one of which stands for the rest.
       TestShape::Heterogeneous, "operation and size"});

  // The reference: same tensor, same read and reduction, no operation.  It
  // depends only on the size, so it is measured once per size and reused by
  // every variant -- three variants over one ladder otherwise pay for the
  // identical session three times over, and on providers that compile ahead
  // of time the session is the expensive part.
  std::map<int64_t, Run> floors;
  auto floorFor = [&](int64_t rows) -> const Run &
  {
    auto it = floors.find(rows);
    if (it == floors.end())
      it = floors.emplace(rows,
                          measure(rt, ep, OnnxActivation::None, rows,
                                  warmupCount, forceIters, specifiedIters))
               .first;
    return it->second;
  };

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;

    for (const Size &sz : kSizes)
    {
      if (clpeak::cancelRequested())
        break;

      const uint64_t bytes = (uint64_t)sz.rows * kCols * 2ull;
      const std::string metric = std::string(v.label) + "_" + sz.label;
      const std::string note =
          std::string(sz.label) + " of activations -- " + v.note +
          "  Read against onnx-tensor-bw's rung of the same name: the ratio is "
          "how much of its streaming rate this provider keeps once it has to "
          "apply a function to the data.";

      if (bytes * kCopiesAtPeak > maxTensorBytes())
      {
        test.skip(metric, ResultStatus::Unsupported,
                  "larger than this machine's memory budget allows", note);
        continue;
      }

      const Run &floor = floorFor(sz.rows);
      Run full = measure(rt, ep, v.act, sz.rows, warmupCount, forceIters,
                         specifiedIters);
      if (full.us <= 0.0)
      {
        test.skip(metric, full.status,
                  full.error.empty() ? "run failed" : full.error, note);
        continue;
      }

      // The operation has to account for a real share of the time, not one
      // microsecond of difference between two noisy measurements.  TensorRT
      // reported 249 us against a 248 us reference, and the microsecond
      // between them divided out to 17 TB/s -- forty times the card's memory
      // bandwidth, published as a peak.  A tenth is a low bar that still
      // rejects anything the reference's own jitter could account for.
      const double floorUs = (floor.us > 0.0) ? floor.us : 0.0;
      const double netUs = full.us - floorUs;
      if (netUs <= 0.1 * full.us)
      {
        CLPEAK_VLOG("onnx-activation[%s/%s]: %s lost in the noise "
                    "(%.0f us against a %.0f us reference)\n",
                    ep.providerKey.c_str(), v.label, sz.label,
                    full.us, floorUs);
        test.skip(metric, ResultStatus::Error,
                  "too close to the reference graph it is measured against",
                  note);
        continue;
      }

      // One pass in, one pass out.
      const double bps = 2.0 * (double)bytes / (netUs * 1.0e-6);
      CLPEAK_VLOG("onnx-activation[%s/%s]: %s -> %.1f GB/s (%.0f us, "
                  "floor %.0f us)\n",
                  ep.providerKey.c_str(), v.label,
                  sz.label, bps, full.us, floorUs);
      test.emit(metric, (float)bps, note.c_str());
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
