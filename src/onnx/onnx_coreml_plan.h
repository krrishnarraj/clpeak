#ifndef CLPEAK_ONNX_COREML_PLAN_H
#define CLPEAK_ONNX_COREML_PLAN_H

// The Core ML compute plan behind a CoreML-provider session: which compute
// unit Core ML gave each operation, read from the lines the provider logs
// when asked for it.
//
// ORT's fallback guard (session.disable_cpu_ep_fallback) proves the CoreML
// provider took every node.  It cannot prove Core ML *ran* them where the
// row says: there is no Neural-Engine-only configuration, CPUAndNeuralEngine
// is the strictest request, and Core ML moves an operation the ANE cannot
// take -- or one its planner judges too small to send -- onto the CPU
// without a word.  On an M1 Pro that CPU path (BNNS on the AMX units) is
// faster than ORT's own CPU provider at everything, so "the accelerator row
// is far above the CPU-EP row" holds for a CPU number too, and the fp32
// matmul, every fp32 convolution and block row, the small resident-tensor
// rungs, the dispatch rows and a 64-token prefill were all published under
// the Neural Engine's name (2.37 TFLOPS fp32 matmul: the Core ML CPU unit's
// 2.36 to the digit).
//
// The provider option ProfileComputePlan=1 (ONNX Runtime 1.20+) makes the
// CoreML EP load MLComputePlan for the compiled model and NSLog one line per
// operation, synchronously, before the session is initialised:
//
//   Operation: ios18.matmul, Device Usage: <MLNeuralEngineComputeDevice: 0x…>, Estimated Cost: 0.049300
//
// NSLog writes to stderr, so a console capture around session creation
// holds the plan.  This header parses that text and judges it the way the
// native Core ML backend judges its own plan (src/coreml/coreml_session.mm,
// onDevice()): a session is on its unit when the operations placed
// elsewhere carry under 5% of the estimated cost.  The share matters
// because the tail of every result-scaled graph -- a reduce and a scalar
// multiply over one row -- may stay on the CPU beside a 2048-cube matmul
// on the ANE, and refusing the multiply over that would be absurd.
//
// The plan does not say whether the unit *could* have taken an operation
// (the native backend reads supportedComputeDevices; the provider logs only
// the preferred one), so a refusal here names the placement and nothing
// more, and a ladder that meets one at its smallest size climbs on: too
// small for the planner is not too big for the unit.
//
// Pure string handling, no Apple headers: it builds everywhere and the
// judgement can be read without a Mac.

#include <string>
#include <vector>

struct OnnxCoremlPlanOp
{
  std::string op;       // "ios18.matmul"
  std::string device;   // "MLNeuralEngineComputeDevice", "MLCPUComputeDevice", "MLGPUComputeDevice"
  double cost = 0.0;    // the plan's estimated share of the model's cost
};

struct OnnxCoremlPlan
{
  bool known = false;   // at least one placement line was seen
  std::string failure;  // Core ML's own words when the plan could not be loaded
  std::vector<OnnxCoremlPlanOp> ops;
};

// Every placement line in `consoleText`, in order.  Lines that are not the
// provider's plan output are ignored, so the whole capture can be handed in.
OnnxCoremlPlan onnxParseCoremlPlan(const std::string &consoleText);

// Empty when the plan puts the work on `unitClass` (an MLComputeDevice
// class name, as the lines spell it); otherwise the one-line reason a row
// reports, naming the operations that moved and their share of the cost.
// `unitName` is how the reason calls the unit ("the Neural Engine").
//
// A plan whose every cost is zero -- Core ML answers that for a model it
// costs on the CPU end to end, as it did for the block's 64-token prefill
// -- is judged by operation count instead, since a zero-weighted operation
// off the unit is still an operation off the unit.
std::string onnxCoremlPlanOffDeviceReason(const OnnxCoremlPlan &plan,
                                          const std::string &unitClass,
                                          const std::string &unitName);

// The MLComputeDevice class the CoreML provider's MLComputeUnits request
// names, and how a reason refers to it: "CPUAndNeuralEngine" ->
// {"MLNeuralEngineComputeDevice", "the Neural Engine"}.  Empty for a
// request no single unit answers for (CPUOnly is judged by nothing).
struct OnnxCoremlUnit
{
  std::string cls;
  std::string name;
};
OnnxCoremlUnit onnxCoremlUnitFor(const std::string &computeUnits);

#endif // CLPEAK_ONNX_COREML_PLAN_H
