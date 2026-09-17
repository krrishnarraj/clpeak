#ifdef ENABLE_LITERT

// litert-dispatch-latency: what it costs to ask this accelerator to do
// anything at all.
//
// An accelerator is reached through a runtime, a driver and on an NPU an
// ahead-of-time compiled graph, and every one of those layers charges a
// toll per submission.  That toll is why a device advertising tens of TOPS
// can still lose to a CPU on small work, and it is invisible in every
// throughput row in this backend.  Two readings bracket it: the smallest
// graph that can be expressed at all, and a 256-cube matmul -- 34 MFLOP,
// which any accelerator here should finish in well under a millisecond, so
// whatever the row shows above that floor is overhead rather than
// arithmetic.
//
// Model creation is timed too.  On the CPU it is XNNPACK packing weights;
// on the GPU, shader compilation and upload; on an NPU, the vendor
// compiler, which is the reason cold-start behaviour differs so sharply
// from steady-state throughput.

#include <litert/litert_peak.h>
#include "litert_bench.h"
#include "litert_model.h"
#include "litert_session.h"

#include <cstring>
#include <string>
#include <vector>

namespace
{

constexpr int64_t kTrivialWidth = 64;    // smallest graph worth expressing
constexpr int64_t kSmallMatMul = 256;    // 34 MFLOP: arithmetic is negligible
constexpr unsigned int kBudgetUs = 500000;
constexpr unsigned int kMaxIters = 20000;

struct Run
{
  double perRunUs = -1.0;
  double createUs = 0.0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

Run measure(const LitertRuntime &rt, const litert_device_info_t &dev, const LitertPlan &plan,
            clpeak_tflite::TfliteBytes &&model, const std::string &input, unsigned warmup,
            bool forceIters, unsigned forced)
{
  Run r;
  std::string err;
  auto s = LitertSession::create(rt, dev, std::move(model), litertConfigFor(plan), err);
  if (!s)
  {
    r.error = err;
    r.status = ResultStatus::Unsupported;
    return r;
  }
  r.createUs = s->createUs;
  if (!s->onDevice())
  {
    r.error = s->offDevice();
    r.status = ResultStatus::Unsupported;
    return r;
  }
  if (!s->writeInput(0, input.data(), input.size(), err))
  {
    r.error = err;
    r.status = ResultStatus::Error;
    return r;
  }
  // Waited for one at a time: the toll is per piece of work handed over
  // and taken back, not per submission into a queue.
  auto m = litertMeasure(*s, warmup, kBudgetUs, forceIters, forced, kMaxIters, true);
  if (m.meanUs <= 0.0)
  {
    r.error = m.error;
    r.status = m.status;
    return r;
  }
  r.perRunUs = m.meanUs;
  return r;
}

} // namespace

int LitertPeak::runDispatchLatency(const LitertRuntime &rt, const litert_device_info_t &dev,
                                   benchmark_config_t &cfg)
{
  (void)cfg;
  const LitertPlan plan = litertPlanFor(LitertFormat::Fp32, dev.accel);

  auto test = currentDeviceScope->beginTest(
      {"litert_dispatch_latency", "LiteRT dispatch latency", "s", Category::Latency,
       "The fixed cost of handing this accelerator one piece of work and taking "
       "the result back: a one-operator graph, a 256-cubed matmul whose "
       "arithmetic is negligible, and creating a compiled model in the first "
       "place.",
       TestShape::Heterogeneous, "what is submitted"});

  std::string trivialIn;
  for (int64_t i = 0; i < kTrivialWidth; i++)
    trivialIn += litertScalarBytes(plan.act, 0.5f);
  Run trivial = measure(rt, dev, plan, litertTrivialModel(plan, kTrivialWidth), trivialIn, warmupCount,
                        forceIters, specifiedIters);

  const std::string trivialNote =
      "One elementwise multiply over 64 values, the smallest graph there is: its "
      "whole time is per-inference overhead.";
  if (trivial.perRunUs > 0.0)
    test.emit("trivial_op", (float)(trivial.perRunUs * 1e-6), trivialNote.c_str());
  else
    test.skip("trivial_op", trivial.status, trivial.error, trivialNote);

  const std::string scalar = litertScalarBytes(plan.act, 1.0009765625f);
  Run matmul = measure(rt, dev, plan, litertMatMulModel(plan, kSmallMatMul, kSmallMatMul, kSmallMatMul),
                       scalar, warmupCount, forceIters, specifiedIters);
  const std::string mmNote =
      "A 256-cubed matmul with resident operands, 34 MFLOP: the time above the "
      "trivial row is what a real kernel adds in setup, not arithmetic.";
  if (matmul.perRunUs > 0.0)
    test.emit("matmul_256", (float)(matmul.perRunUs * 1e-6), mmNote.c_str());
  else
    test.skip("matmul_256", matmul.status, matmul.error, mmNote);

  const std::string createNote =
      "Compiling the one-operator graph, what an application pays at start-up "
      "before its first inference; on an NPU this is the vendor compiler.";
  if (trivial.createUs > 0.0)
    test.emit("session_create", (float)(trivial.createUs * 1e-6), createNote.c_str());
  else
    test.skip("session_create", trivial.status, trivial.error, createNote);

  test.end();
  return 0;
}

#endif // ENABLE_LITERT
