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
  // Float8, ONNX 1.14 / opset 19.  FN means "finite": E4M3FN has no infinity,
  // spending that encoding on one more magnitude instead, so its maximum is
  // 448.  E5M2 keeps IEEE's infinities and reaches 57344 with one fewer
  // mantissa bit.  Which of the two a device prefers is the whole question the
  // pair of rows exists to answer.
  ONNX_DT_FLOAT8E4M3FN = 17,
  ONNX_DT_FLOAT8E5M2   = 19,
  // Int4, ONNX 1.16 / opset 21.  Two elements to a byte, first in the low
  // nibble.  There is no int4 matmul operator in ONNX -- MatMulInteger and
  // QLinearMatMul are both 8-bit -- so int4 only ever appears on the weights,
  // dequantized on the way into a floating-point multiply.
  ONNX_DT_UINT4 = 21,
  ONNX_DT_INT4  = 22,
  // Float4, ONNX 1.18 / opset 23.  One sign bit, two exponent, one mantissa,
  // and therefore eight magnitudes in total: 0, 0.5, 1, 1.5, 2, 3, 4, 6.  No
  // infinity and no NaN -- every bit pattern is a number.  Packed two to a
  // byte like int4.
  ONNX_DT_FLOAT4E2M1 = 23,
};

// The lowest opset that can express `dtype` at all.  Datatypes arrived in the
// standard in waves and a model has to declare an opset high enough for the
// ones it names, so every recipe asks this rather than hard-coding a number.
// 17 is the floor: it is what the oldest runtime this backend speaks to
// understands, and everything expressible there stays there.
int onnxOpsetForDtype(int dtype);

// The oldest ONNX Runtime that parses a model at `opset`, as an OrtApi
// version -- ORT numbers its API after its own minor version, so this is
// directly comparable with OrtRuntime::apiVersion.  A model handed to an
// older runtime fails to load with a message about the opset, which is a
// confusing way to learn that a datatype is simply newer than the install.
uint32_t onnxMinOrtApiForOpset(int opset);

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

  // Opset this model declares; the IR version follows from it.  Defaults to
  // 17, which every recipe that predates the low-precision datatypes uses and
  // which the oldest supported runtime understands.  Raising it is per model
  // on purpose: a global bump would make every row fail on a runtime that is
  // merely old, rather than the one row whose datatype is genuinely newer.
  void setOpset(int opset);

  // ReduceMax over `axes`, spelled for whatever opset this graph declares.
  // Opset 18 moved `axes` from an attribute to an input, so a recipe that
  // raises its opset for a datatype would otherwise break on a node that has
  // nothing to do with that datatype.  Every reduction here goes through this.
  void reduceMax(const std::string &in, const std::string &out,
                 const OnnxDims &axes);

  // Convenience for the int64 shape tensors Reshape takes as an input.
  void shapeInitializer(const std::string &name, const OnnxDims &shape);

  std::string build() const;   // ModelProto bytes

private:
  std::string m_nodes, m_inits, m_inputs, m_outputs;
  int         m_nodeCount = 0;
  int         m_opset     = 17;
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
// `wDtype` is the weight element type and `actDtype` the activation one.  Both
// may be int8/uint8 or one of the float8 formats; the model's opset follows
// whichever is newer.
//
// `floatIo` keeps the quantized type off the graph boundary: the input arrives
// as fp32 and is quantized on device, and the result is dequantized before it
// leaves.  Needed because an EP may implement a datatype internally and still
// refuse it as a tensor it must receive -- TensorRT imports float8
// initializers happily and answers "input onnx tensor data type: 17 not
// supported" for the same type as a graph input.  The values are unchanged:
// every number handed in is already exactly representable in the target type,
// so the added QuantizeLinear round-trips it rather than rounding it again.
std::string onnxQdqMatMulModel(int64_t M, int64_t K, int64_t N,
                               const std::string &weightRaw,
                               float aScale, float bScale, float cScale,
                               int actDtype, int wDtype = ONNX_DT_INT8,
                               bool floatIo = false);

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
//
// `reduceInFloat` casts the product to fp32 before reducing it, and makes the
// runtime scalar and the output fp32 with it.  A provider can have a matmul
// for a datatype and no reduction for it -- the CUDA EP multiplies bf16 and
// has no bf16 ReduceMax, and ORT does not fail that cleanly, it throws out of
// its memcpy transformer complaining that the node has no provider.  Only
// used when the native-dtype form is refused, since the cast is a full pass
// over the result and costs a few percent at the large sizes and more at the
// small ones.
// Weight-only quantized GEMM, throughput-shaped: fp16 activations against a
// blocked-quantized weight matrix, which is the form quantized language models
// actually ship in (AWQ, GPTQ, ORT's own MatMulNBits all have this shape).
//
// The arithmetic is fp16 -- the weights are dequantized on the way into the
// multiply, by a fused kernel where the provider has one -- so what a narrow
// weight type buys here is *weight traffic*, not arithmetic rate.  Reporting it
// in TFLOPS rather than TOPS is deliberate for that reason.
//
// `wScalesRaw` holds one scale per block of `blockSize` rows per column, so its
// shape is [K / blockSize, N] and its element type is fp16, which is also the
// type the dequantize produces.  Zero points are omitted: the quantization is
// symmetric, and a blocked zero-point tensor would have to be packed too.
// NVFP4 on both operands: the shape a Blackwell-class float4 tensor core wants,
// and the one TensorRT asked for when it refused a per-tensor E2M1 graph with
// "CHECK(output_quantize_axis_.has_value()) failed" -- it wants a quantization
// axis, which only block scaling has.
//
// Two levels of scale, which is what makes it NVFP4 rather than plain blocked
// float4: an E4M3 scale per block of 16 along the reduction axis, and one fp32
// scale for the whole tensor.  ONNX expresses that as a dequantize feeding a
// dequantize -- the block scales are themselves dequantized by the global one
// before they scale the data.  That second level sits on the *scale* path, so
// nothing is inserted between the data's dequantize and the matmul, which is
// the arrangement ORT needs to keep recognising a quantized matmul.
//
// Scales are E4M3 and therefore reachable at float4's own opset 23; MXFP4's
// E8M0 scale is a later opset than anything here emits.
std::string onnxResidentNvfp4MatMulModel(int64_t M, int64_t K, int64_t N,
                                         int64_t blockSize,
                                         const std::string &aPacked,
                                         const std::string &aBlockScales,
                                         const std::string &bPacked,
                                         const std::string &bBlockScales,
                                         float globalScale);

std::string onnxResidentWeightOnlyMatMulModel(int64_t M, int64_t K, int64_t N,
                                              int wDtype, int64_t blockSize,
                                              const std::string &aRaw,
                                              const std::string &wPacked,
                                              const std::string &wScalesRaw);

std::string onnxResidentMatMulModel(int64_t M, int64_t K, int64_t N, int dtype,
                                    const std::string &aRaw,
                                    const std::string &bRaw,
                                    bool reduceInFloat = false);

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
                                       const std::string &bRaw,
                                       float aScale, float bScale, float cScale,
                                       int actDtype,
                                       int wDtype = ONNX_DT_INT8);

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
float    bf16ToFloat(uint16_t h);

// Float8, round-to-nearest-even, saturating rather than overflowing to
// infinity -- the values these graphs carry sit inside [-1, 1] so the extreme
// paths are unreachable in practice, but a quantization helper that silently
// produced a NaN would poison an accuracy row rather than fail it.
uint8_t floatToFp4E2M1(float f);   // returns the 4-bit code, 0..15
float   fp4E2M1ToFloat(uint8_t code);
uint8_t floatToFp8E4M3(float f);
uint8_t floatToFp8E5M2(float f);
float   fp8E4M3ToFloat(uint8_t v);
float   fp8E5M2ToFloat(uint8_t v);

// Is `dtype` one of the quantized element types the QDQ recipes accept?
bool onnxIsQuantElem(int dtype);

// Can ORT's QDQ selector legally fuse a graph over this element type?
//
// The fusion target is QLinearMatMul, which is an 8-bit *integer* operator: it
// carries int8 and uint8 and nothing else.  Let the selector fire on any other
// quantized type and it rewrites a valid model into one that fails its own type
// check.  A provider with real hardware for such a type consumes the QDQ nodes
// itself and never wanted the rewrite, so holding it off costs nothing.
bool onnxQdqFusionIsLegal(int dtype);

// The scale that maps this quantized type's stored values back onto [-1, 1],
// which is the range every QDQ recipe here dequantizes into.  Keeping the
// dequantized range identical across types is what makes their accuracy rows
// comparable: the operands differ only in how they were rounded.
float onnxQuantScaleFor(int dtype);

// Encode `v` (already in [-1, 1]) into one element of `dtype`, little-endian.
void onnxStoreQuantElem(void *dst, int64_t index, int dtype, float v);

// Store one signed 4-bit value at `index` of a packed nibble array.  ONNX packs
// two elements per byte in flattened order with the first in the low nibble.
// `q` is clamped to [-8, 7].
void    onnxStoreNibble(void *dst, int64_t index, uint8_t nib);
uint8_t onnxLoadNibble(const void *src, int64_t index);
void onnxStoreInt4(void *dst, int64_t index, int q);
int  onnxLoadInt4(const void *src, int64_t index);

// Deterministic weight value at a position, in [-0.5, 0.5).
//
// A hash of the position rather than a running sequence, so a block can be
// visited twice -- once to find its maximum, once to quantize against it --
// without holding the matrix in floats.  At 16384 square that would be a
// gigabyte of scratch to produce 128 MB of weights.
float onnxWeightAt(int64_t i, int64_t j, uint32_t seed);

// Quantize a [rows, cols] matrix into NVFP4: packed E2M1 values plus one E4M3
// scale per block of `blockSize` along the blocked axis, with `globalScale`
// factored out of those scales.  `blockAxis` is 0 when blocks run down rows and
// 1 when they run along columns; both operands of a matmul block along the
// reduction axis, which is a different axis number for each.
//
void onnxFillNvfp4(std::string &packed, std::string &blockScales,
                   int64_t rows, int64_t cols, int blockAxis, int64_t blockSize,
                   float globalScale, uint32_t seed);

// Bytes a tensor of `count` elements of `dtype` occupies, nibble packing
// included -- dtypeSize()-style helpers cannot express a half.
uint64_t onnxElemBytes(int dtype, int64_t count);

#endif // CLPEAK_ONNX_MODEL_H
