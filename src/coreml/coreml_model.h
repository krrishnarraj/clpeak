#ifndef CLPEAK_COREML_MODEL_H
#define CLPEAK_COREML_MODEL_H

// In-memory Core ML model construction.  The graphs clpeak benchmarks are a
// handful of operations, so it emits the ML Program (MIL) protobuf wire
// format and the weight blob directly -- no protobuf library, no coremltools,
// no .mlpackage shipped as an asset, and byte-identical models on every
// device, which is what makes the Neural-Engine-vs-GPU-vs-CPU comparison
// mean anything.  The layout it produces is exactly what coremltools writes:
// Model.proto wrapping a MILSpec.Program, weights in the MIL storage-format
// blob file, packaged as an .mlpackage directory (coreml_session.mm writes
// that directory; this file only produces its two byte strings).

#include <cstdint>
#include <string>
#include <vector>

// MILSpec.DataType values.
enum CoremlDtype : int
{
  CML_BOOL   = 1,
  CML_STRING = 2,
  CML_FP16   = 10,
  CML_FP32   = 11,
  CML_BF16   = 13,   // in the enum, accepted by no operation -- see gemm.cpp
  CML_INT8   = 21,
  CML_INT16  = 22,
  CML_INT32  = 23,
  CML_INT4   = 25,
  CML_UINT8  = 31,
  CML_UINT4  = 35,
  CML_FP8E4M3 = 40,  // storage types the compression ops may take, iOS 26+
  CML_FP8E5M2 = 41,
};

using CoremlDims = std::vector<int64_t>;   // empty = scalar

// Bytes a tensor of `count` elements of `dtype` occupies, nibble packing
// included.
uint64_t coremlElemBytes(int dtype, int64_t count);

// The MIL opset string a specification version declares: spec 8 (iOS 17) is
// "CoreML7", spec 9 (iOS 18) "CoreML8", spec 10 (iOS 26) "CoreML9".
std::string coremlOpsetName(int specVersion);

// The lowest specification version whose opset can express `dtype` as a
// weight in the compression ops used here, or as an activation type.  The
// caller compares it with coremlSpecVersion() and reports "needs macOS X"
// instead of letting the compiler fail on an op it has never heard of.
int coremlSpecForDtype(int dtype);

// Human-readable OS floor for a specification version: "macOS 15 / iOS 18".
std::string coremlOsForSpec(int specVersion);

// ---------------------------------------------------------------------------
// The program builder
// ---------------------------------------------------------------------------

// Builds one ML Program with a single `main` function.  Operations must be
// added in topological order (MIL is SSA and no sort is done here).  Every
// operation input is a *name* -- a graph input, an earlier operation's
// output, or a `const` operation this builder emits for it -- so scalar
// parameters (axes, flags, dtype strings) are constants like everything else.
class CoremlProgram
{
public:
  explicit CoremlProgram(int specVersion);

  int specVersion() const { return m_spec; }

  // Function inputs / block outputs.  These are also the model's feature
  // descriptions, so dtype must be one Core ML accepts at the boundary: fp16,
  // fp32 or int32.
  void input(const std::string &name, int dtype, const CoremlDims &dims);
  void output(const std::string &name, int dtype, const CoremlDims &dims);

  // ---- Constants ---------------------------------------------------------
  // Each returns the name it was given, so a call can sit inline in an op's
  // input list.  Small values are immediate; tensors go to the weight blob
  // (Core ML mmaps that file and the compilers treat it as model weights,
  // which is what a real model's constants are).
  std::string constBool(const std::string &name, bool v);
  std::string constInt(const std::string &name, int32_t v);
  std::string constInts(const std::string &name, const std::vector<int32_t> &v);
  std::string constString(const std::string &name, const std::string &s);
  // A floating-point scalar in `dtype` (fp16 or fp32).
  std::string constFloat(const std::string &name, int dtype, float v);
  // An int8 scalar (quantization zero points).
  std::string constInt8(const std::string &name, int8_t v);
  // A tensor of `dtype`, `raw` being its little-endian element bytes, nibble
  // types packed two per byte with the first element in the low nibble.
  std::string constTensor(const std::string &name, int dtype,
                          const CoremlDims &dims, const std::string &raw);

  // ---- Operations --------------------------------------------------------
  struct Out
  {
    std::string name;
    int dtype;
    CoremlDims dims;
  };
  using Inputs = std::vector<std::pair<std::string, std::string>>; // param -> name

  // A generic operation.  `type` is the MIL op name ("matmul", "softmax",
  // ...); the opset prefix is the function's.
  void op(const std::string &type, const Inputs &ins, const std::vector<Out> &outs);
  void op(const std::string &type, const Inputs &ins, const Out &out)
  {
    op(type, ins, std::vector<Out>{out});
  }

  // ---- Compressed weights ------------------------------------------------
  // Each emits the constexpr operation that decompresses a stored weight
  // into `out` (fp16 or fp32).  The parameters are Values rather than names,
  // because that is how the ops are serialized -- attributes on the iOS 16/17
  // ops, Value-bound inputs on the iOS 18 ones -- and the tensor data goes to
  // the blob like any other constant.

  // int8 (or uint8) weights with one scale per element along `axis`
  // (per-channel), or a single scale when `scales` holds one value.
  // Symmetric: the zero point is 0.  iOS 16 / macOS 13.
  void affineDequantize(const Out &out, int srcDtype, const std::string &packed,
                        const std::string &scalesRaw, int64_t scaleCount,
                        int scaleDtype, int32_t axis);

  // Block-quantized weights: `data` of `srcDtype` (int4/uint4/int8/uint8, or
  // fp8 where the OS accepts it) with a scale tensor of `scaleDims` -- the
  // block size along each axis is data.dims / scale.dims.  Symmetric, no
  // offset.  iOS 18 / macOS 15.
  void blockwiseDequantize(const Out &out, int srcDtype, const CoremlDims &dataDims,
                           const std::string &packedData, int scaleDtype,
                           const CoremlDims &scaleDims, const std::string &scalesRaw);

  // Palettized weights: `nbits`-wide indices into one lookup table of
  // 2^nbits entries in `out.dtype`.  Per-tensor, scalar palettization.
  // iOS 18 / macOS 15.
  void lutToDense(const Out &out, int nbits, const std::string &packedIndices,
                  const std::string &lutRaw);

  // ---- Serialization -----------------------------------------------------
  std::string buildModel() const;             // Model.proto bytes
  const std::string &weightBytes() const { return m_blob; }

private:
  struct Feature
  {
    std::string name;
    int dtype;
    CoremlDims dims;
  };

  std::string constValue(const std::string &name, int dtype, const CoremlDims &dims,
                         const std::string &valueMsg);
  // Append `raw` to the weight blob and return the Value message referring
  // to it.
  std::string blobValue(int dtype, const CoremlDims &dims, const std::string &raw);
  std::string opName(const std::string &type);

  int m_spec;
  std::vector<Feature> m_inputs, m_outputs;
  std::string m_ops;      // serialized Block.operations entries
  std::string m_blob;     // the weight file
  int m_opCount = 0;
};

// ---------------------------------------------------------------------------
// Weight formats and the projection recipe shared by every test
// ---------------------------------------------------------------------------

// How a matrix multiply's weight operand is stored.  Core ML's arithmetic is
// fp16 or fp32 and nothing else; every narrow format is *storage*, unpacked
// on the way into a float multiply by the compression ops above -- except
// the quantized-activation form, where the Neural Engine on A17 Pro / M4 and
// later runs the multiply in 8-bit integers.
enum class CoremlWeight
{
  Fp32,        // plain fp32 constant
  Fp16,        // plain fp16 constant
  Bf16,        // a bfloat16 constant -- the enum has the type, no op takes it
  Int8Channel, // int8, one fp scale per output column (constexpr_affine_dequantize)
  Int4Block,   // int4, one fp scale per block of kWeightBlock along K
  Int4Lut,     // 4-bit indices into a 16-entry lookup table (palettized)
  Fp8Block,    // float8 E4M3, block scales like Int4Block -- attempted, iOS 26+
  Int8Qdq,     // int8 weights and int8 activations (quantize/dequantize)
};

// One scale per this many weights along the reduction axis, for the blocked
// formats -- the grouping AWQ, GPTQ and the ONNX backend's rows all use, so a
// Core ML reading divides by an ONNX one.
constexpr int64_t kCoremlWeightBlock = 32;

// The activation (arithmetic) width a format multiplies in.
int coremlActDtype(CoremlWeight w);

// The element type the weights are *stored* in.
int coremlStoredDtype(CoremlWeight w);

// Bytes a [K, N] weight in this format occupies, scales included -- what the
// decode rows count as traffic.
uint64_t coremlWeightBytes(CoremlWeight w, int64_t K, int64_t N);

// The specification version a format needs, and what it is called.
int coremlSpecForWeight(CoremlWeight w);

// The specification version a model should *declare*: the lowest that
// expresses its weight format (and fused attention, which is iOS 18), never
// the OS's newest.  A newer opset changes which operation versions a model
// carries, and the compute units do not keep up uniformly: at the macOS 26
// opset the M1's GPU has no `mul`, `quantize` or `dequantize`, and a graph
// that ran entirely on the GPU at the iOS 18 opset lands on the CPU.  This
// is also what coremltools does with its minimum deployment target.
int coremlSpecNeeded(CoremlWeight w, bool fusedAttention = false);
const char *coremlWeightLabel(CoremlWeight w);

// Deterministic weight value at a position, in [-0.5, 0.5).  A hash of the
// position rather than a running sequence, so a matrix can be visited twice
// -- once to find a block's maximum, once to quantize against it -- without
// holding it in floats.
float coremlWeightAt(int64_t i, int64_t j, uint32_t seed);

// The quantized-activation scales, shared by every W8A8 projection: the
// input scale maps a K-deep dot product of `magnitude`-bounded operands onto
// int8, the output scale the result of one.
float coremlQdqActScale(float magnitude);
float coremlQdqOutScale(int64_t K, float magnitude);

// Emit one weight matrix W[K, N] of values magnitude * coremlWeightAt(i, j,
// seed), stored in format `w`, and the operation that presents it to a
// multiply as `outName` in the format's activation type.  Returns the name
// to feed the matmul.  When `dequantized` is non-null it receives the exact
// values the device will multiply -- the stored codes widened -- which is
// what an accuracy reference has to use.
std::string coremlEmitWeight(CoremlProgram &p, const std::string &outName,
                             CoremlWeight w, int64_t K, int64_t N, uint32_t seed,
                             float magnitude, std::vector<float> *dequantized = nullptr);

// out = in * W in the format's arithmetic; for Int8Qdq the activations are
// quantized on the way in and the result quantized and dequantized on the way
// out, the shape a W8A8 layer has.  `in` is [M, K] in coremlActDtype(w).
void coremlEmitProjection(CoremlProgram &p, const std::string &out,
                          const std::string &in, int64_t M, int64_t K, int64_t N,
                          CoremlWeight w, uint32_t seed, float magnitude,
                          std::vector<float> *dequantized = nullptr);

// ---------------------------------------------------------------------------
// Recipes
// ---------------------------------------------------------------------------
// Every recipe takes the OS's specification version so a caller has one
// number to gate on, but declares only the version it needs (see
// coremlSpecNeeded).

// Throughput-shaped GEMM: both operands are constants and the result is
// reduced to one row, so nothing large crosses the host boundary per run.
// A runtime scalar `s` keeps the graph from being a constant expression.
//
//   resultScaled = true   s scales the reduced result -- cheapest, and the
//                         one a compiler could fold; the ladder checks that
//                         its timings grow with the work.
//   resultScaled = false  s scales A before the multiply, one elementwise
//                         pass that no compiler can fold away.  The Neural
//                         Engine compiles a dynamic operand to a slower
//                         program (about 20%), so it is the fallback shape.
//
// The W8A8 form always scales the operand: its activations are quantized on
// device from a live value, the way a layer's are.
CoremlProgram coremlResidentMatMulModel(int spec, int64_t M, int64_t K, int64_t N,
                                        CoremlWeight w, bool resultScaled);

// y[M, N] = x[M, N] * W, x a model input and y the full result -- the plain
// shape the accuracy rows need, since they compare actual values.
CoremlProgram coremlPlainMatMulModel(int spec, int64_t M, int64_t K, int64_t N,
                                     CoremlWeight w, std::vector<float> *dequantized);

// y[1, cols] = x[1, d] * W[d, cols] in fp16: one matrix-vector product
// against a resident weight, the operation generating a token performs.
CoremlProgram coremlGemvModel(int spec, int64_t d, int64_t cols, uint32_t seed);

// Throughput-shaped 2-D convolution over a resident [1, C, S, S] input and a
// [C, C/group, k, k] weight, "same" padding, result reduced per channel.
CoremlProgram coremlConvModel(int spec, int64_t channels, int64_t spatial,
                              int64_t kernel, int64_t group, int dtype);

enum class CoremlActivation { None, Silu, Softmax, LayerNorm };

// One activation over a resident [rows, cols] fp16 constant scaled by a
// runtime value, reduced to one row on the way out.  `None` is the reference
// graph: the same read, scale and reduction with no operation applied.
CoremlProgram coremlActivationModel(int spec, int64_t rows, int64_t cols,
                                    CoremlActivation act);

enum class CoremlTransfer { Resident, ToDevice, RoundTrip };

// Three spellings of one fp16 matmul, x[rows, K] * W[K, N], that differ
// only in what crosses the host boundary -- so the differences between their
// times are the transfers alone.  See transfer.cpp.
//
//   Resident   x is a constant, the result is reduced to one row and scaled
//              by the runtime input `s`: nothing large crosses either way.
//              (`resultScaled` false: `s` scales x instead, the fallback when
//              the reduced form is caught folding.)
//   ToDevice   x is a model input; the result is still reduced.
//   RoundTrip  x is a model input and the whole result y[rows, N] returns.
CoremlProgram coremlTransferModel(int spec, CoremlTransfer dir, int64_t rows, int64_t K,
                                  int64_t N, bool resultScaled = true);

// The smallest graph worth expressing: y = x * K over [rows, cols] fp16, K a
// full-size constant.  `salt` perturbs K so every process compiles a model
// Core ML has not cached -- the dispatch-latency test's create timing is
// meant to be a cold compile.
CoremlProgram coremlTrivialModel(int spec, int64_t rows, int64_t cols, uint32_t salt);

// A 256-cube matmul with a live input: arithmetic negligible, so what the
// row shows above the trivial graph is submission overhead.
CoremlProgram coremlSmallMatMulModel(int spec, int64_t d);

// ---------------------------------------------------------------------------
// Transformer decoder block
// ---------------------------------------------------------------------------

struct CoremlBlockShape
{
  int64_t dModel;
  int64_t heads;
  int64_t headDim;
  int64_t ffnHidden;
  int64_t seq;      // tokens processed this pass (prefill: many, decode: 1)
  int64_t kvLen;    // decode only: length of the cached context (0 = prefill)

  // Precision is a pair: the projection weight format (which fixes the
  // arithmetic width) and, separately, how the decode KV cache is stored.
  CoremlWeight weights = CoremlWeight::Fp16;
  bool int8Kv = false;   // decode: the cache as int8, dequantized into attention

  // Attention as Core ML's fused scaled_dot_product_attention (iOS 18+) or
  // as explicit matmul / softmax / matmul.  Set by the caller from the spec.
  bool fusedAttention = true;
};

// One llama-style decoder block: QKV projection, multi-head attention, output
// projection + residual, SwiGLU feed-forward + residual, at the precision the
// shape names.  Weights and the KV cache are constants; the activations are a
// constant scaled by the runtime input `s`; the result leaves as one reduced
// row.  Decode additionally returns the new K/V, the cache write a real step
// performs and what keeps those projections live.
CoremlProgram coremlBlockModel(int spec, const CoremlBlockShape &sh);

// ---------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------

uint16_t coremlFloatToHalf(float f);
float    coremlHalfToFloat(uint16_t h);
uint16_t coremlFloatToBf16(float f);
uint8_t  coremlFloatToFp8E4M3(float f);
float    coremlFp8E4M3ToFloat(uint8_t v);

// One fp16 / fp32 scalar as the raw bytes a tensor holds.
std::string coremlFloatScalar(float v, int dtype);

// Fill `count` values of `dtype` (fp16 / fp32) from a xorshift sequence in
// [-0.5, 0.5) * magnitude.
std::string coremlFillFloats(int dtype, int64_t count, uint32_t seed,
                             float magnitude = 1.0f);

#endif // CLPEAK_COREML_MODEL_H
