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
// and shared by all three operations rather than re-timed for each.  Both
// graphs scale the tensor by a runtime value before doing anything else, so
// neither is a constant expression a vendor compiler can evaluate at build
// time -- OpenVINO did exactly that to the earlier result-scaled form and
// every row read "too close to the reference", both being dispatch.
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
#include "onnx_probe.h"
#include "onnx_session.h"

#include <chrono>
#include <cstring>
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
    uint64_t bytes; // the working set; rows follow from the element width
    const char *label;
  };

  const Size kSizes[] = {
      {8ull << 20, "8mb"},
      {32ull << 20, "32mb"},
      {128ull << 20, "128mb"},
  };

  // The rungs are fixed, so the budget check is the only thing standing between
  // a phone and an out-of-memory kill -- Android ends the process rather than
  // failing an allocation, so a rung has to be declined on an estimate rather
  // than attempted and recovered from.  The estimate is what the session holds
  // (onnxHeldBytes): the tensor three times over as a constant, and every
  // tensor of its size the graph computes from it (Variant::intermediates).
  // The reference computes one, the scaled input, and every operation computes
  // that and more, so a reference fits wherever an operation of its size does.
  static uint64_t maxHeldBytes() { return clpeak::memoryBudget(1ull << 30); }

  constexpr unsigned int kSizeBudgetUs = 1000000;

  // The share of the work the operation itself has to account for before the
  // difference is worth reporting -- the work being the measurement less the
  // provider's submission charge, since both graphs pay that and it cancels
  // in the subtraction.
  //
  // A tenth, and the number is doing less than it looks.  What it rejects is
  // a remainder of zero or below: TensorRT fuses SiLU into a graph that then
  // costs the same as the reference (177 us against 176), and its softmax
  // comes out *faster* than doing nothing.  Those are not measurements and
  // never were.
  //
  // It cannot do more than that, and two attempts to make it are worth not
  // repeating.  Raising it to a fifth threw away every CUDA row -- all eight
  // sit between 15.1% and 17.7%, three operations across three working sets,
  // a band too tight to be noise.  And a ceiling drawn from what the provider
  // streams cannot be calibrated at any single size (see below).  The reason
  // neither works is that the rows this would catch are not distinguishable
  // by share: Core ML's softmax lands in the same 8-18% band CUDA's good rows
  // do, and differs only in that it does not reproduce.
  //
  // So the rows near this floor are the least trustworthy thing this backend
  // publishes, and the ladder is what protects a reader: three sizes and
  // three operations, where one row disagreeing with its neighbours is
  // visible.  Core ML's softmax is the known case -- it has read 19 and 139
  // GB/s on a device that streams 89 -- and that instability is a property of
  // that provider's compiler, which the row cannot outvote.
  constexpr double kMinOpShare = 0.10;

  struct Variant
  {
    OnnxActivation act;
    const char *label;
    const char *note; // "<note>, over <size> of activations."
    // The tensors of the input's size it computes, the scaled input among
    // them (onnxResidentActivationModel).
    int intermediates;
  };

  const Variant kVariants[] = {
      {OnnxActivation::Silu, "silu", "SiLU, x times sigmoid(x)", 3},
      {OnnxActivation::Softmax, "softmax", "Softmax across each row", 2},
      {OnnxActivation::LayerNorm, "layernorm",
       "Layer norm (each row's mean and variance, then a rescale)", 2},
  };

  struct Run
  {
    double us = -1.0;
    double createUs = 0.0; // the session's build; zero when none was made
    std::string error;
    ResultStatus status = ResultStatus::Ok;
    // A larger size of the same graph could only fail the same way: its run
    // failed, or its build ran out of memory or past the compile cap.  Not a
    // provider declining the graph or placing it on another unit -- Core ML's
    // planner keeps small work on the CPU and takes the next size up.
    bool endsLadder = false;
  };

  Run measure(const OrtRuntime &rt, const onnx_ep_info_t &ep, int dtype,
              OnnxActivation act, int64_t rows,
              unsigned int warmup, bool forceIters, unsigned int forced)
  {
    Run r;
    const size_t es = (size_t)onnxElemBytes(dtype, 1);

    OrtSession *session = nullptr;
    {
      // Builds the graph and its session in `view`; the tensor is generated
      // afresh each time so it never outlives the model built from it.
      auto create = [&](OnnxReduceView view, double &createUs) {
        std::string xRaw((size_t)rows * kCols * es, '\0');
        {
          float *f = reinterpret_cast<float *>(&xRaw[0]);
          uint16_t *h = reinterpret_cast<uint16_t *>(&xRaw[0]);
          uint32_t s = 0x9e3779b9u;
          for (int64_t i = 0; i < rows * kCols; i++)
          {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            const float v = (float)(s >> 8) / 16777216.0f - 0.5f;
            if (dtype == ONNX_DT_FLOAT)
              f[i] = v;
            else
              h[i] = floatToHalf(v);
          }
        }
        std::string model =
            onnxResidentActivationModel(rows, kCols, dtype, act, xRaw, view);
        xRaw.clear();
        xRaw.shrink_to_fit();

        auto createStart = std::chrono::steady_clock::now();
        auto ses = onnxCreateSession(rt, ep, model, /*keepConstantsUnfolded=*/true);
        createUs = std::chrono::duration<double, std::micro>(
                       std::chrono::steady_clock::now() - createStart)
                       .count();
        CLPEAK_VLOG("onnx-activation[%s]: %lld rows create %.1f s%s\n",
                    ep.providerKey.c_str(), (long long)rows, createUs / 1.0e6,
                    view == OnnxReduceView::Rank4 ? " (rank-4 reduction)" : "");
        return ses;
      };

      const OnnxReduceView view = onnxPrefersRank4Reduce(rt, ep)
                                      ? OnnxReduceView::Rank4
                                      : OnnxReduceView::Rows;
      double createUs = 0.0;
      auto ses = create(view, createUs);
      // The reduction that brings the row back can be refused where the
      // operation is not: the QNN Adreno backend declined every graph here on
      // its ReduceMax.  The reference graph -- the first built at each size,
      // with no operation to fail on -- tries the rank-4 view once, and a
      // provider that takes it keeps it for everything after
      // (onnxPrefersRank4Reduce).
      if (!ses.session && !ses.offDevice && act == OnnxActivation::None &&
          view == OnnxReduceView::Rows &&
          onnxFailureStatus(ses.error) == ResultStatus::Unsupported &&
          !onnxReasonIsOutOfMemory(ses.error) && !clpeak::cancelRequested())
      {
        CLPEAK_VLOG("onnx-activation[%s]: refused (%s); retrying the reduction "
                    "through a rank-4 view\n",
                    ep.providerKey.c_str(), ses.error.c_str());
        auto r4 = create(OnnxReduceView::Rank4, createUs);
        if (r4.session)
        {
          onnxNoteRank4Reduce(rt, ep);
          ses = r4;
        }
      }
      if (!ses.session)
      {
        r.error = ses.error;
        r.status = onnxFailureStatus(ses.error);
        r.endsLadder = r.status == ResultStatus::Error ||
                       onnxReasonIsOutOfMemory(ses.error);
        return r;
      }
      r.createUs = createUs;
      if (createUs > kOnnxMaxCreateUs)
      {
        CLPEAK_VLOG("onnx-activation[%s]: %lld rows create %.1f s > %.1f s, skipping\n",
                    ep.providerKey.c_str(), (long long)rows,
                    createUs / 1.0e6, kOnnxMaxCreateUs / 1.0e6);
        r.error = "session creation took " +
                  std::to_string((long long)(createUs / 1.0e6)) +
                  " s, exceeds " +
                  std::to_string((long long)(kOnnxMaxCreateUs / 1.0e6)) +
                  " s compilation budget";
        r.status = ResultStatus::Unsupported;
        r.endsLadder = true;
        rt.api->ReleaseSession(ses.session);
        return r;
      }
      session = ses.session;
    }

    // Exactly one: the scalar scales the tensor before the operation, so
    // one keeps every value the operation sees identical to the constant
    // -- and a compiler cannot know it is one.
    const std::string sVal = onnxFloatScalar(1.0f, dtype);
    std::vector<uint8_t> outBuf((size_t)kCols * es, 0);

    OrtMemoryInfo *mi = nullptr;
    OrtValue *inVal = nullptr, *outVal = nullptr;
    OrtStatus *st = rt.api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                                OrtMemTypeDefault, &mi);
    const int64_t outShape[1] = {kCols};
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, const_cast<char *>(sVal.data()), sVal.size(), nullptr, 0,
          (ONNXTensorElementDataType)dtype, &inVal);
    if (!st)
      st = rt.api->CreateTensorWithDataAsOrtValue(
          mi, outBuf.data(), outBuf.size(), outShape, 1,
          (ONNXTensorElementDataType)dtype, &outVal);
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
    if (r.us <= 0.0)
    {
      if (r.status == ResultStatus::Ok)
        r.status = ResultStatus::Error;
      r.endsLadder = true;
    }

    if (inVal)
      rt.api->ReleaseValue(inVal);
    if (outVal)
      rt.api->ReleaseValue(outVal);
    rt.api->ReleaseSession(session);
    return r;
  }

  constexpr size_t kNumSizes = sizeof(kSizes) / sizeof(kSizes[0]);

  std::string seconds(double us)
  {
    return std::to_string((long long)(us / 1.0e6 + 0.5)) + " s";
  }

  // One graph's climb through kSizes: the reference's, or one operation's.
  struct Ladder
  {
    double prevUs = 0.0, prevPrevUs = 0.0; // its last two builds
    size_t prevAt = 0;                     // the size the last was built at
    std::string ended;                     // why nothing larger is built
    ResultStatus endedStatus = ResultStatus::Ok;

    void record(size_t si, const Run &r)
    {
      if (r.createUs <= 0.0)
        return;
      prevPrevUs = prevUs;
      prevUs = r.createUs;
      prevAt = si;
    }

    // The compile cap, checked before paying for the build
    // (onnxPredictCreateUs); a graph's first build is always paid for, and
    // seeds the prediction.  Each size is four times the one below it, as
    // each of onnx-gemm's widths holds four times the weights of the last.
    // QNN's HTP on a Galaxy S24 Ultra spent 17 s and then 68 s building and
    // timing softmax at 8 and 32 MB: four times as long at each size.
    std::string pastCompileCap(size_t si) const
    {
      if (prevUs <= 0.0)
        return std::string();
      double us = prevUs, before = prevPrevUs;
      for (size_t k = prevAt; k < si; k++)
      {
        const double next = onnxPredictCreateUs(us, before, /*confirming=*/false);
        before = us;
        us = next;
      }
      if (us <= kOnnxMaxCreateUs)
        return std::string();
      return "its " + std::string(kSizes[prevAt].label) + " session took " +
             seconds(prevUs) + " to create, which predicts about " +
             seconds(us) + " here, past the " + seconds(kOnnxMaxCreateUs) +
             " compilation budget";
    }
  };

} // namespace

int OnnxPeak::runActivation(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                            benchmark_config_t &cfg)
{
  (void)cfg;

  auto test = currentDeviceScope->beginTest(
      {"onnx_activation", "ONNX activation throughput", "bps",
       Category::Bandwidth,
       "Bandwidth of the operations between the matmuls -- SiLU gate, "
       "softmax, layer norm -- over a transformer-shaped tensor.  Each is net "
       "of a reference graph that reads the same tensor and applies nothing.",
       // Three operations across three working-set sizes: nine separate
       // measurements, no one of which stands for the rest.
       TestShape::Heterogeneous, "operation and size"});

  // The width this provider streams fastest (see onnxStreamDtype): fp16
  // wherever the accelerator has fp16 kernels, fp32 on a provider that would
  // otherwise convert every fp16 tensor on the way in and report the
  // conversion under this heading.  The working sets are in bytes, so the
  // row count halves in fp32 and each rung still names its size truthfully.
  const int dtype = onnxStreamDtype(rt, ep);
  const size_t es = (size_t)onnxElemBytes(dtype, 1);
  const char *widthNote =
      (dtype == ONNX_DT_FLOAT)
          ? "  Measured in fp32, which this provider streams faster than fp16."
          : "";

  // One row of the reference graph: 8-16 KB, so essentially all of its time
  // is what the provider charges to accept a submission.  The guard below
  // needs it because on a provider that charges 156 us the measurement and
  // its reference are mostly that charge, and the share the operation
  // accounts for looks far smaller than it is.  It may fail where the sizes
  // do not -- Core ML's planner keeps work this small off the Neural Engine
  // -- and then the guard takes the whole measurement as work, which only
  // makes it stricter.
  const Run dispatch = measure(rt, ep, dtype, OnnxActivation::None, 1,
                               warmupCount, forceIters, specifiedIters);
  const double dispatchUs = (dispatch.us > 0.0) ? dispatch.us : 0.0;
  CLPEAK_VLOG("onnx-activation[%s]: submission floor %.0f us\n",
              ep.providerKey.c_str(), dispatchUs);

  // The reference: same tensor, same read and reduction, no operation.  It
  // depends only on the size, so it is measured once per size and reused by
  // every variant -- three variants over one ladder otherwise pay for the
  // identical session three times over, and on providers that compile ahead
  // of time the session is the expensive part.
  //
  // A size whose reference failed has no rows at all.  Subtracting nothing
  // would publish the scaffolding's time -- the read, the multiply and the
  // reduction -- as the operation's own: QNN's HTP on a Galaxy S24 Ultra
  // failed the 128 MB reference at run time (QNN_GRAPH_ERROR_INVALID_HANDLE),
  // which left silu_128mb with nothing to subtract.  A failure a larger graph
  // could only repeat (Run::endsLadder) ends the reference's ladder, and with
  // it every operation's: that run went on to build softmax at 128 MB, and
  // the process died there.
  struct Reference
  {
    bool tried = false;
    Run run;
    std::string reason; // why this size has no reference
  };
  Reference refs[kNumSizes];
  Ladder refLadder;
  auto referenceAt = [&](size_t si, int64_t rows) -> const Reference &
  {
    Reference &ref = refs[si];
    if (ref.tried)
      return ref;
    ref.tried = true;
    if (!refLadder.ended.empty())
    {
      ref.run.status = refLadder.endedStatus;
      ref.reason = refLadder.ended;
      return ref;
    }
    const std::string pastCap = refLadder.pastCompileCap(si);
    if (!pastCap.empty())
    {
      ref.run.status = ResultStatus::Unsupported;
      ref.reason = "reference graph not built: " + pastCap;
      return ref;
    }
    ref.run = measure(rt, ep, dtype, OnnxActivation::None, rows, warmupCount,
                      forceIters, specifiedIters);
    refLadder.record(si, ref.run);
    if (ref.run.us > 0.0)
      return ref;
    const std::string error =
        ref.run.error.empty() ? std::string("run failed") : ref.run.error;
    ref.reason = "reference graph failed: " + error;
    CLPEAK_VLOG("onnx-activation[%s]: %s reference failed (%s)%s\n",
                ep.providerKey.c_str(), kSizes[si].label, error.c_str(),
                ref.run.endsLadder ? "; no larger size is attempted" : "");
    if (ref.run.endsLadder)
    {
      refLadder.ended = "reference graph failed at " +
                        std::string(kSizes[si].label) +
                        ", so no larger size is attempted: " + error;
      refLadder.endedStatus = ref.run.status;
    }
    return ref;
  };

  for (const Variant &v : kVariants)
  {
    if (clpeak::cancelRequested())
      break;

    Ladder ladder;
    for (size_t si = 0; si < kNumSizes; si++)
    {
      if (clpeak::cancelRequested())
        break;

      const Size &sz = kSizes[si];
      const uint64_t bytes = sz.bytes;
      const int64_t rows = (int64_t)(bytes / ((uint64_t)kCols * es));
      const std::string metric = std::string(v.label) + "_" + sz.label;
      const std::string note = std::string(v.note) + ", over " +
                               std::to_string(bytes >> 20) +
                               " MB of activations." + widthNote;

      if (onnxHeldBytes(ep, bytes, (uint64_t)v.intermediates * bytes) >
          maxHeldBytes())
      {
        test.skip(metric, ResultStatus::Unsupported,
                  "larger than this machine's memory budget allows", note);
        continue;
      }
      if (!ladder.ended.empty())
      {
        test.skip(metric, ladder.endedStatus, ladder.ended, note);
        continue;
      }
      // Before the reference, which this size would otherwise build for
      // nothing.
      const std::string pastCap = ladder.pastCompileCap(si);
      if (!pastCap.empty())
      {
        CLPEAK_VLOG("onnx-activation[%s/%s]: %s not built: %s\n",
                    ep.providerKey.c_str(), v.label, sz.label,
                    pastCap.c_str());
        test.skip(metric, ResultStatus::Unsupported, pastCap, note);
        continue;
      }

      const Reference &ref = referenceAt(si, rows);
      if (ref.run.us <= 0.0)
      {
        test.skip(metric, ref.run.status, ref.reason, note);
        continue;
      }
      Run full = measure(rt, ep, dtype, v.act, rows, warmupCount, forceIters,
                         specifiedIters);
      ladder.record(si, full);
      if (full.us <= 0.0)
      {
        const std::string error =
            full.error.empty() ? std::string("run failed") : full.error;
        test.skip(metric, full.status, error, note);
        if (full.endsLadder)
        {
          ladder.ended =
              metric + " failed, so no larger size is attempted: " + error;
          ladder.endedStatus = full.status;
        }
        continue;
      }

      // The operation has to account for a real share of the work, not one
      // microsecond of difference between two noisy measurements.  TensorRT
      // reported 249 us against a 248 us reference, and the microsecond
      // between them divided out to 17 TB/s -- forty times the card's memory
      // bandwidth, published as a peak.
      //
      // The share is taken of the work, not of the wall time: both graphs pay
      // the provider's submission charge and it cancels in the subtraction,
      // so leaving it in the denominator makes a well-resolved operation look
      // marginal wherever dispatch is expensive.  DirectML charges 156 us and
      // CUDA 69 us against measurements of a few hundred.
      const double floorUs = ref.run.us;
      const double netUs = full.us - floorUs;
      const double workUs = full.us - dispatchUs;
      if (netUs <= 0.0 || workUs <= 0.0 || netUs <= kMinOpShare * workUs)
      {
        CLPEAK_VLOG("onnx-activation[%s/%s]: %s lost in the noise "
                    "(%.0f us against a %.0f us reference, %.0f us of "
                    "submission)\n",
                    ep.providerKey.c_str(), v.label, sz.label,
                    full.us, floorUs, dispatchUs);
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
                  sz.label, bps / 1.0e9, full.us, floorUs);
      test.emit(metric, (float)bps, note.c_str());
    }
  }

  test.end();
  return 0;
}

#endif // ENABLE_ONNX
