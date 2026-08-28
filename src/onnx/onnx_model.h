#ifndef CLPEAK_ONNX_MODEL_H
#define CLPEAK_ONNX_MODEL_H

// In-memory ONNX model construction.  The graphs clpeak benchmarks are a
// handful of nodes, so it emits the protobuf wire format directly -- no
// protobuf library, no .onnx files on disk, and byte-identical models on
// every platform, which is what makes the cross-vendor comparison mean
// anything.

#include <cstdint>
#include <string>
#include <vector>

// ONNX TensorProto.DataType values.  The ONNXTensorElementDataType enum in
// the ORT C API uses the same numbering, so these cross over directly.
enum OnnxDtype : int
{
  ONNX_DT_FLOAT    = 1,
  ONNX_DT_UINT8    = 2,
  ONNX_DT_INT8     = 3,
  ONNX_DT_INT32    = 6,
  ONNX_DT_INT64    = 7,
  ONNX_DT_FLOAT16  = 10,
  ONNX_DT_BFLOAT16 = 16,
};

using OnnxDims = std::vector<int64_t>;   // empty = scalar

// One node attribute.  Only the two forms clpeak's graphs need: a single
// int (Softmax axis) and a list of ints (Transpose perm).
struct OnnxAttr
{
  std::string          name;
  bool                 isList = false;
  int64_t              i      = 0;
  std::vector<int64_t> ints;

  static OnnxAttr num(const std::string &n, int64_t v)
  {
    OnnxAttr a; a.name = n; a.i = v; return a;
  }
  static OnnxAttr list(const std::string &n, std::vector<int64_t> v)
  {
    OnnxAttr a; a.name = n; a.isList = true; a.ints = std::move(v); return a;
  }
};

// Builds one GraphProto and wraps it in a ModelProto.  Nodes must be added
// in topological order (ONNX requires it, and no sort is done here).
class OnnxGraph
{
public:
  void input(const std::string &name, int dtype, const OnnxDims &dims);
  void output(const std::string &name, int dtype, const OnnxDims &dims);

  // Constant tensor embedded in the model.  `raw` is little-endian element
  // data, exactly as ONNX raw_data expects.
  void initializer(const std::string &name, int dtype, const OnnxDims &dims,
                   const std::string &raw);

  void node(const std::string &opType,
            const std::vector<std::string> &inputs,
            const std::vector<std::string> &outputs,
            const std::vector<OnnxAttr> &attrs = {});

  // Convenience for the int64 shape tensors Reshape takes as an input.
  void shapeInitializer(const std::string &name, const OnnxDims &shape);

  std::string build() const;   // ModelProto bytes

private:
  std::string m_nodes, m_inits, m_inputs, m_outputs;
  int         m_nodeCount = 0;
};

// C[M,N] = MatMul(A[M,K], B[K,N]) in `dtype`.  A is a graph input; B is an
// embedded initializer, so the EP sees it as constant weights it may
// pre-pack -- the weight-stationary GEMM inference is built from.
std::string onnxMatMulModel(int64_t M, int64_t K, int64_t N, int dtype,
                            const std::string &weightRaw);

// The same GEMM in QDQ form, the shape every NPU actually wants:
//
//   A_q(int8) -> DequantizeLinear -\
//                                   MatMul -> QuantizeLinear -> C_q(int8)
//   B_q(int8) -> DequantizeLinear -/
//
// Input and output are int8, so no float conversion sits in the timed path.
// ORT's QDQ handling fuses this into the EP's own quantized matmul.
std::string onnxQdqMatMulModel(int64_t M, int64_t K, int64_t N,
                               const std::string &weightRawInt8,
                               float aScale, float bScale, float cScale,
                               int actDtype);

// Throughput-shaped GEMM: both operands are initializers and the result is
// summed down to one row, so nothing large crosses the host boundary on each
// run.  With A as a graph input and C returned to the host -- the obvious
// shape -- a discrete GPU is measured through its PCIe bus instead of its
// tensor cores: on an RTX 5060 that reported 15 TFLOPS for a fp16 matmul
// while a whole transformer block, whose weights are resident, reached 28.
//
// A scalar input scales the reduced result, so the graph has a runtime
// dependency, and the session must disable constant folding or ORT will
// evaluate the entire matmul once at load time.  `gemm.cpp` cross-checks that
// timings still scale with the cube of the size, which is what folding would
// break.
std::string onnxResidentMatMulModel(int64_t M, int64_t K, int64_t N, int dtype,
                                    const std::string &aRaw,
                                    const std::string &bRaw);

// Same idea in QDQ form.  The DequantizeLinear/MatMul/QuantizeLinear pattern
// is left untouched -- inserting anything between the dequantize and the
// matmul stops ORT recognising it as a quantized matmul at all, which would
// silently measure float arithmetic.
// Activations are uint8 and weights int8 -- the combination ONNX Runtime's
// own quantizer emits for deployment, and the one x86 implements without
// VNNI.  Signed activations fuse on ARM but not there, which showed up as an
// unfused graph on a Threadripper while the same code fused on an M1.
// `actDtype` is ONNX_DT_UINT8 (zero point 128) or ONNX_DT_INT8 (zero point 0).
// No single choice works everywhere: x86 MLAS without VNNI implements uint8
// activations against int8 weights and will not fuse signed ones, while
// TensorRT rejects uint8 outright and requires a zero point of zero.  The
// caller picks by trying, and `gemm.cpp` uses the fusion check to decide.
std::string onnxResidentQdqMatMulModel(int64_t M, int64_t K, int64_t N,
                                       const std::string &aRaw,
                                       const std::string &bRawInt8,
                                       float aScale, float bScale, float cScale,
                                       int actDtype);

// Throughput-shaped 2-D convolution, built like the resident GEMM above:
// input and weights are constants, the result is reduced to one value per
// output channel, and a runtime scalar keeps the graph live.  NCHW layout,
// stride 1, padding chosen so the output keeps the input's spatial size.
//
// `group` == 1 is an ordinary convolution; `group` == channels is a depthwise
// one, which has the same shape but a fraction of the arithmetic and is where
// accelerators built around dense multiply-accumulate arrays tend to fall
// down.
std::string onnxResidentConvModel(int64_t channels, int64_t spatial,
                                  int64_t kernel, int64_t group, int dtype,
                                  const std::string &xRaw,
                                  const std::string &wRaw);

// The non-matmul operations a transformer layer is padded with.  `None` is
// the reference graph: the same constant read and reduced with no operation
// applied, so subtracting its time leaves the operation's own cost.
enum class OnnxActivation { None, Silu, Softmax, LayerNorm };

// One activation applied to a resident [rows, cols] fp16 constant, reduced to
// a single row on the way out.  Built like the other throughput models: the
// operand never crosses the host boundary, and a runtime scalar scales the
// reduced result so the graph is not entirely constant.
std::string onnxResidentActivationModel(int64_t rows, int64_t cols,
                                        OnnxActivation act,
                                        const std::string &xRaw);

// Which direction a transfer model exercises.
enum class OnnxTransfer { ToDevice, RoundTrip, ComputeOnly };

// Models that deliberately do the opposite of the throughput ones: they push
// a large tensor across the host boundary and compute almost nothing, so what
// is timed is the handover.  All three take the whole tensor in; they differ
// in what comes back.
//
//   ToDevice     large in, one gathered element out -- the trip in
//   RoundTrip    large in, large out
//   ComputeOnly  large in, same operation as RoundTrip, one gathered element
//                out -- everything the round trip does except ship the result
//                back, so the difference between them is the trip back
//
// One element is *gathered*, never reduced: a reduction reads the whole
// tensor on the device, and that pass lands in whatever the test was trying
// to isolate.  A graph input is materialised in full before any kernel sees
// it, so gathering still forces the transfer.
std::string onnxTransferModel(OnnxTransfer dir, int64_t elems);

// ---------------------------------------------------------------------------
// Transformer decoder block
// ---------------------------------------------------------------------------

// Geometry of one llama-style decoder block.  Fixed by the caller so every
// device runs byte-identical work; see src/onnx/block.cpp for the values and
// why they are what they are.
struct OnnxBlockShape
{
  int64_t dModel;
  int64_t heads;
  int64_t headDim;
  int64_t ffnHidden;
  int64_t seq;      // tokens processed this pass (prefill: many, decode: 1)
  int64_t kvLen;    // decode only: length of the cached context (0 = prefill)
};

// One fp16 decoder block: QKV projection, multi-head attention, output
// projection + residual, SwiGLU feed-forward + residual.
//
//   prefill (kvLen == 0): attention is self-attention over `seq` tokens.
//   decode  (kvLen  > 0): `seq` is 1 and attention reads a constant KV cache,
//                         with the new K/V exposed as graph outputs -- a real
//                         step writes them to the cache, and making them
//                         outputs is also what stops the projections being
//                         dead-code eliminated.
//
// Weights and KV cache are initializers, so the EP sees them as constants it
// may pre-pack, exactly as it would a real model's.
std::string onnxBlockModel(const OnnxBlockShape &s);

// Scalar float -> IEEE fp16 / bfloat16 bit conversion for weight buffers.
uint16_t floatToHalf(float f);
uint16_t floatToBf16(float f);
float    halfToFloat(uint16_t h);

#endif // CLPEAK_ONNX_MODEL_H
