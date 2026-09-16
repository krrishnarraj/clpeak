#ifndef CLPEAK_TFLITE_MODEL_H
#define CLPEAK_TFLITE_MODEL_H

// .tflite models emitted as FlatBuffer bytes, without flatc, the FlatBuffers
// library or the 1.1 MB schema_generated.h -- the same choice the ONNX and
// Core ML backends make with protobuf.  A model is a handful of tables
// (Model, SubGraph, Tensor, Buffer, OperatorCode, Operator, a few options
// tables) and a minimal back-to-front builder writes them exactly as the
// reference implementation would: vtables, forward uoffsets, 16-byte-aligned
// weight buffers, the "TFL3" file identifier.
//
// Field ids and enum values are transcribed from tflite/converter/schema/
// schema.fbs at the LiteRT release pinned in third_party/litert/README.md.
// A FlatBuffer reader tolerates a *missing* field (it reads the default),
// so a table written here with fewer fields than the schema has is valid;
// what must be exact is the id of every field written and the position of
// every union member, and each is commented with its schema source below.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace clpeak_tflite
{

// schema.fbs `enum TensorType : byte`.
enum class TfType : int8_t
{
  F32 = 0,
  F16 = 1,
  I32 = 2,
  U8 = 3,
  I64 = 4,
  Bool = 6,
  I16 = 7,
  I8 = 9,
  U16 = 16,
  I4 = 17,
  BF16 = 18,
  I2 = 19,
  U4 = 20,
  F8E4M3 = 21,
  F8E5M2 = 22,
};

// Bytes per element; nibble types count 0.5 rounded up per element only in
// packedBytes(), which is what callers want for a buffer.
size_t tfliteElementBits(TfType t);
size_t tflitePackedBytes(TfType t, size_t count);
const char *tfliteTypeName(TfType t);

// schema.fbs `enum BuiltinOperator`, the subset the recipes use.
enum class TfOp : int32_t
{
  Add = 0,
  Concatenation = 2,
  Conv2d = 3,
  DepthwiseConv2d = 4,
  Dequantize = 6,
  FullyConnected = 9,
  Logistic = 14,
  Mul = 18,
  Relu = 19,
  Relu6 = 21,
  Reshape = 22,
  Softmax = 25,
  Tanh = 28,
  Transpose = 39,
  Mean = 40,
  Sub = 41,
  Div = 42,
  StridedSlice = 45,
  Exp = 47,
  Cast = 53,
  Sum = 74,
  Sqrt = 75,
  Rsqrt = 76,
  Pow = 78,
  ReduceMax = 82,
  Square = 92,
  SquaredDifference = 99,
  Quantize = 114,
  BatchMatMul = 126,
  Gelu = 150,
  StablehloComposite = 206,
};

// schema.fbs `enum ActivationFunctionType` / `enum Padding`.
enum class TfActivation : int8_t { None = 0, Relu = 1, ReluN1To1 = 2, Relu6 = 3, Tanh = 4 };
enum class TfPadding : int8_t { Same = 0, Valid = 1 };

// Quantization of one tensor (schema `QuantizationParameters`).  Affine:
// `scale` and `zeroPoint` of equal length -- one entry per tensor, or one
// per index along `quantizedDim` for per-channel weights.  Blockwise (the
// LLM weight format): `blockSize` > 0 with the scales (and optionally zero
// points) living in *other tensors* of the subgraph, named by index.
struct TfQuant
{
  std::vector<float> scale;
  std::vector<int64_t> zeroPoint;
  int32_t quantizedDim = 0;
  int32_t blockSize = 0;        // > 0: blockwise; scale/zeroPoint above unused
  int32_t scalesTensor = -1;    // blockwise: tensor index of the scales
  int32_t zeroPointsTensor = -1;

  bool empty() const { return scale.empty() && blockSize == 0; }
};

// Builtin options of one operator.  One struct for every kind, since each
// recipe sets two or three fields and a variant per op would be noise; the
// `kind` says which fields are read.  Everything not listed for a kind is
// ignored, and an op whose options table has no fields (Quantize,
// Dequantize, Transpose, Rsqrt ...) takes kind None.
struct TfOptions
{
  enum class Kind
  {
    None,
    FullyConnected,   // act, keepNumDims, asymmetricQuantizeInputs, quantizedBiasType
    BatchMatMul,      // adjX, adjY, asymmetricQuantizeInputs
    Conv2d,           // padding, strideW/H, act, dilationW/H, quantizedBiasType
    DepthwiseConv2d,  // padding, strideW/H, depthMultiplier, act, dilationW/H
    Softmax,          // beta
    Mul,              // act
    Add,              // act
    Sub,              // act
    Reducer,          // keepDims  (Mean, Sum, ReduceMax)
    Reshape,          // newShape
    Cast,             // castIn, castOut
    Gelu,             // approximate
    Concatenation,    // axis, act
    Composite,        // compositeName, decompositionSubgraph, compositeVersion
  };
  Kind kind = Kind::None;

  TfActivation act = TfActivation::None;
  bool keepNumDims = false;
  bool asymmetricQuantizeInputs = false;
  TfType quantizedBiasType = TfType::F32;   // F32 = "unset" in the schema (0)
  bool adjX = false, adjY = false;
  TfPadding padding = TfPadding::Same;
  int32_t strideW = 1, strideH = 1;
  int32_t dilationW = 1, dilationH = 1;
  int32_t depthMultiplier = 1;
  float beta = 1.0f;
  bool keepDims = false;
  std::vector<int32_t> newShape;
  TfType castIn = TfType::F32, castOut = TfType::F32;
  bool approximate = false;
  int32_t axis = 0;
  std::string compositeName;
  int32_t decompositionSubgraph = -1;
  int32_t compositeVersion = 1;
};

// The finished model.  The bytes live inside `storage` starting at `head`
// (a back-to-front builder fills from the end), so the model is
// `data()`/`size()` and never copied out; storage stays 16-byte aligned so a
// runtime that maps weights in place sees them aligned.
struct TfliteBytes
{
  std::vector<uint8_t> storage;
  size_t head = 0;
  std::string description;   // the model's own description string
  double buildUs = 0.0;      // what build() took: the fills are most of it
  const uint8_t *data() const { return storage.data() + head; }
  size_t size() const { return storage.size() - head; }
};

// One model under construction: tensors, buffers and operators of one or
// more subgraphs.  Subgraph 0 is the one that runs; further subgraphs exist
// only as composite-op decompositions.  Tensor and buffer indices are the
// integers the add* calls return.
class TfliteModel
{
public:
  TfliteModel();

  // Reserve room up front for the weight bytes the recipe will add, so the
  // builder never reallocates (and copies) a half-built model of hundreds of
  // megabytes.  Advisory; the builder grows without it.
  void reserveBytes(size_t bytes);

  // A weight buffer.  Index 0 is the empty buffer every non-constant tensor
  // refers to, so the first call returns 1.  Not copied: `data` must stay
  // valid until build() returns.
  int addBuffer(const void *data, size_t bytes);

  // A weight buffer generated straight into the model bytes: build() reserves
  // `bytes` inside the FlatBuffer and calls `fill` with the destination.  A
  // model's weights then exist once in memory instead of twice, which on a
  // phone is the difference between a 1 GB rung and a killed process.
  int addBufferFill(size_t bytes, void (*fill)(uint8_t *dst, size_t bytes, void *ctx), void *ctx);

  // Add a tensor to subgraph `sg`.  `buffer` 0 makes it an activation (an
  // input, output or intermediate); a positive index makes it a constant.
  int addTensor(int sg, const std::vector<int32_t> &shape, TfType type,
                const std::string &name, int buffer = 0, const TfQuant &q = TfQuant());

  // Add an operator to subgraph `sg`.  -1 in `inputs` is the schema's
  // "optional input absent" (a FULLY_CONNECTED with no bias).
  void addOp(int sg, TfOp op, int version, const std::vector<int32_t> &inputs,
             const std::vector<int32_t> &outputs, const TfOptions &opts = TfOptions());

  // Graph inputs / outputs of subgraph `sg`, in signature order.
  void setInputs(int sg, const std::vector<int32_t> &tensors);
  void setOutputs(int sg, const std::vector<int32_t> &tensors);

  // A further subgraph (returns its index); subgraph 0 always exists.
  int addSubgraph(const std::string &name);

  // Serialize.  Every recipe finishes with this exactly once.
  TfliteBytes build(const std::string &description) const;

  size_t numTensors(int sg = 0) const { return subgraphs_[sg].tensors.size(); }

private:
  struct Tensor
  {
    std::vector<int32_t> shape;
    TfType type;
    int buffer;
    std::string name;
    TfQuant quant;
  };
  struct Op
  {
    TfOp op;
    int version;
    std::vector<int32_t> inputs, outputs;
    TfOptions opts;
  };
  struct Subgraph
  {
    std::string name;
    std::vector<Tensor> tensors;
    std::vector<Op> ops;
    std::vector<int32_t> inputs, outputs;
  };
  struct Buffer
  {
    const uint8_t *data;   // null when `fill` generates the bytes
    size_t bytes;
    void (*fill)(uint8_t *, size_t, void *);
    void *ctx;
  };

  std::vector<Subgraph> subgraphs_;
  std::vector<Buffer> buffers_;
  size_t reserve_ = 0;
};

} // namespace clpeak_tflite

#endif // CLPEAK_TFLITE_MODEL_H
