#ifndef CLPEAK_ONNX_GEMM_SETUP_H
#define CLPEAK_ONNX_GEMM_SETUP_H

#ifdef ENABLE_ONNX

// Everything the MatMul ladder (gemm.cpp) and the per-provider shape probe
// (onnx_probe.cpp) share: the variant table, the operand generator, and how
// a resident-operand session is built, bound and driven.  One definition,
// so the probe cannot drift from the graph it is choosing a shape for.

#include "onnx_model.h"
#include "onnx_session.h"

#include <cstdint>
#include <string>
#include <vector>

struct onnx_ep_info_t;

namespace onnxgemm
{

// Tiny probe dimension: large enough to exercise the same kernels, small
// enough that AOT compilation stays cheap (QNN HTP: 32^3 ~0.3 s against
// 1024^3 tens of seconds, TensorRT 32^3 ~0.8 s vs 64^3 3.6 s).
constexpr int64_t kProbeDim = 32;

// NVFP4's second scale.  A power of two so factoring it out of the block
// scales is exact and the row measures the format rather than an arithmetic
// accident.
constexpr float kNvfp4GlobalScale = 0.125f;

struct Variant
{
  int dtype;         // element type of the graph's input/output
  bool qdq;          // build the quantized (DequantizeLinear/MatMul/Q) form
  const char *label;
  const char *note;  // the row's description, after the sweep sentence
  int64_t blockSize; // >0: blocked, one scale per this many elements
  bool nvfp4;        // blocked on *both* operands, with a second scale
};

// The rows, in the order they are measured and reported.  int8 QDQ is the
// one integer row and carries its own unit (ops); everything else is flops.
extern const Variant kFpVariants[];
extern const size_t  kFpVariantCount;
extern const Variant kIntVariants[];
extern const size_t  kIntVariantCount;

inline bool isIntVariant(const Variant &v)
{
  return v.dtype == ONNX_DT_INT8 && v.qdq;
}

// Quantization schemes, tried in order until one fuses.  There is no single
// choice that works everywhere: TensorRT rejects unsigned activations and
// demands a zero point of zero, while x86 MLAS without VNNI implements only
// the unsigned form and quietly declines to fuse the signed one.  Trying is
// the only way to know, and the fusion check is what decides.
struct QuantScheme
{
  int actDtype;
  int wDtype;
  const char *name;
};

// Fills `out` (room for two) and returns how many apply to `v`.
size_t schemesFor(const Variant &v, QuantScheme out[2]);

// Which live shapes a variant can be built in, most preferred first.  The
// probe walks this list and keeps the first that builds, runs in the width
// it was given and -- for the quantized rows -- fuses.
std::vector<OnnxLiveShape> liveShapesFor(const Variant &v);

// Bytes of operands the model for (variant, D, shape) embeds: what the size
// ladder checks against the memory budget and the protobuf ceiling.
uint64_t operandBytes(const Variant &v, int64_t D, OnnxLiveShape shape);

size_t dtypeSize(int dtype);

// Deterministic values, generated once and reused for inputs and weights.
// Floats land in [-0.5, 0.5) and int8 in [-127, 127]: small magnitudes keep
// fp16 accumulation over a 4096-deep dot product far from overflow, and
// avoid the NaN/denormal slow paths raw random bit patterns would hit.
void fillTensor(std::string &raw, int dtype, int64_t count, uint32_t seed);

// Output scale for the QDQ form: four sigma of a K-deep dot product mapped
// onto the widest code the output type has.
float qdqOutputScale(int64_t K, int outDtype);

// One built session with its bound scalar input(s) and reduced output.
struct GemmSetup
{
  OrtSession *session = nullptr;
  OrtValue *inVal = nullptr;  // the runtime scalar S
  OrtValue *zaVal = nullptr;  // Add forms only: the literal zero ZA
  OrtValue *outVal = nullptr; // reduced row
  std::vector<uint8_t> inBuf, zaBuf, outBuf;
  std::string error;

  // Not copyable, and the compiler has to enforce it: inVal and outVal are
  // OrtValues built over inBuf and outBuf, so a copy leaves them pointing at
  // the original's buffers.  Moving is fine -- a moved vector keeps its
  // allocation -- which is why returning one of these by value works and
  // handing back a reference to one does not.
  GemmSetup() = default;
  GemmSetup(const GemmSetup &) = delete;
  GemmSetup &operator=(const GemmSetup &) = delete;
  GemmSetup(GemmSetup &&) = default;
  GemmSetup &operator=(GemmSetup &&) = default;
};

void destroySetup(const OrtRuntime &rt, GemmSetup &g);

// Build model + session + bound tensors for one (variant, D) in `shape`.
// `actDtype`/`wgtDtype` apply to the QDQ form; `reduceInFloat` to the plain
// one.  On failure `error` is set and `session` is null.
GemmSetup makeSetup(const OrtRuntime &rt, const onnx_ep_info_t &ep,
                    const Variant &v, int64_t D, bool profile,
                    int actDtype, bool reduceInFloat, int wgtDtype,
                    OnnxLiveShape shape);

// Mean microseconds per Run() over n runs; negative on failure.
double timeRuns(const OrtRuntime &rt, GemmSetup &g, unsigned int n);

} // namespace onnxgemm

#endif // ENABLE_ONNX
#endif // CLPEAK_ONNX_GEMM_SETUP_H
