#ifndef CLPEAK_COREML_SESSION_H
#define CLPEAK_COREML_SESSION_H

// One compiled, loaded Core ML model on one compute device, plus what the
// compute plan says about where its operations will run.  Pure C++ surface
// so the tests stay .cpp files; the Objective-C lives in coreml_session.mm.

#include <coreml/coreml_peak.h>
#include "coreml_model.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// "Neural Engine" / "GPU" / "CPU".
const char *coremlKindName(CoremlDeviceKind k);

// Where the compute plan says one operation will run.  `preferred` is the
// planner's choice -- a cost decision, and the unit that actually runs it;
// `capable` says whether the session's own device was among those that
// could have.  The two answer different questions: an operation the Neural
// Engine cannot run at all, and one Core ML would not bother sending it for
// work this small, both land on the CPU.
struct CoremlPlacement
{
  std::string opType;     // "ios17.matmul"
  CoremlDeviceKind preferred;
  bool capable;
  double weight;          // the plan's estimated share of the model's cost; -1 unknown
                          // (also when the whole plan is costed at zero, so that
                          // every operation counts whole -- see create())
};

// The share of a model's estimated cost that may run off its device before
// the session is refused.  A [1, N] scalar multiply on the CPU after a
// 2048-cube matmul on the Neural Engine is not a CPU measurement -- and the
// plan never lists the GPU as able to run a standalone elementwise op at
// all -- while a matmul moved off the device carries most of the cost and
// fails whatever else stayed.
constexpr double kCoremlOffDeviceShare = 0.05;

class CoremlSession
{
public:
  // Writes the program as an .mlpackage in the temp directory, compiles it,
  // loads it for `dev`, and reads its compute plan.  Null with `error` set
  // when any step fails -- the compiler's own message, which for a datatype
  // or operation Core ML does not have is the answer the row reports.
  static std::unique_ptr<CoremlSession> create(const coreml_device_info_t &dev,
                                               const CoremlProgram &prog,
                                               std::string &error);
  ~CoremlSession();

  // What creation cost, in microseconds: writing and compiling the package,
  // loading the compiled model (where the Neural Engine compiler runs), and
  // reading the plan.
  double compileUs = 0.0;
  double loadUs = 0.0;
  double planUs = 0.0;

  // The plan, one entry per operation that has a device (constants and the
  // compile-time decompressions do not).  Empty when the OS cannot report
  // one, in which case `planKnown` is false and the placement is unverified.
  std::vector<CoremlPlacement> placement;
  bool planKnown = false;

  // Did the work land on the device this session was created for -- every
  // operation, or all but glue worth under kCoremlOffDeviceShare of the
  // plan's cost estimate?  Always true for the CPU.  When false, `offDevice()`
  // names the operations that moved, which is what the row reports instead
  // of a number: Core ML keeps the CPU as a fallback in every configuration,
  // and a matmul it moved off the Neural Engine would otherwise be measured
  // on the CPU under the accelerator's name.  `offDeviceCapable()` is true
  // when every moved operation *could* have run here and the planner simply
  // chose not to send it -- what happens to small work.  `glue()` names the
  // negligible operations that ran elsewhere on a session that passed.
  bool onDevice() const;
  std::string offDevice() const;
  bool offDeviceCapable() const;
  std::string glue() const;

  // Bind a model input to a host buffer the session owns; the returned
  // pointer stays valid for the session's lifetime and may be rewritten
  // between runs.  `size` is the buffer's byte length.
  void *bindInput(const std::string &name, int dtype, const std::vector<int64_t> &dims,
                  size_t size, std::string &error);

  // One prediction.
  bool run(std::string &error);

  // Mean microseconds per prediction over `n` of them; negative with `error`
  // set on failure.
  double timeRuns(unsigned n, std::string &error);

  // Bytes of a named output after the last run().
  bool outputBytes(const std::string &name, std::vector<uint8_t> &out, std::string &error);

  struct Impl;

private:
  CoremlSession() = default;
  Impl *impl = nullptr;
};

#endif // CLPEAK_COREML_SESSION_H
