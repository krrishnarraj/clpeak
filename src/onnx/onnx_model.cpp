#ifdef ENABLE_ONNX

#include "onnx_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

// ---------------------------------------------------------------------------
// Minimal protobuf wire-format writer.  Everything a GraphProto needs is
// varints (wire type 0) and length-delimited fields (wire type 2).
// ---------------------------------------------------------------------------

namespace
{

struct Pb
{
  std::string b;

  void varint(uint64_t v)
  {
    while (v >= 0x80)
    {
      b.push_back((char)(0x80 | (v & 0x7f)));
      v >>= 7;
    }
    b.push_back((char)v);
  }
  void tag(int field, int wire) { varint(((uint64_t)field << 3) | wire); }

  void vint(int field, uint64_t v)
  {
    tag(field, 0);
    varint(v);
  }
  void str(int field, const std::string &s)
  {
    tag(field, 2);
    varint(s.size());
    b += s;
  }
};

// TypeProto for a static-shape tensor.  A scalar (empty dims) still carries a
// TensorShapeProto, just an empty one -- that is how ONNX spells rank 0.
std::string tensorType(int dtype, const OnnxDims &dims)
{
  Pb shape;
  for (int64_t d : dims)
  {
    Pb dim;
    dim.vint(1, (uint64_t)d);       // Dimension.dim_value
    shape.str(1, dim.b);            // TensorShapeProto.dim
  }
  Pb tt;
  tt.vint(1, (uint64_t)dtype);      // Tensor.elem_type
  tt.str(2, shape.b);               // Tensor.shape
  Pb type;
  type.str(1, tt.b);                // TypeProto.tensor_type
  return type.b;
}

std::string valueInfo(const std::string &name, int dtype, const OnnxDims &dims)
{
  Pb vi;
  vi.str(1, name);                  // ValueInfoProto.name
  vi.str(2, tensorType(dtype, dims));
  return vi.b;
}

// AttributeProto.  `type` (field 20) must be set or ORT rejects the model:
// 2 = INT (field 3), 7 = INTS (field 8, written unpacked).
std::string attribute(const OnnxAttr &a)
{
  Pb at;
  at.str(1, a.name);                // name
  if (a.isList)
  {
    for (int64_t v : a.ints)
      at.vint(8, (uint64_t)v);      // ints
    at.vint(20, 7);                 // type = INTS
  }
  else
  {
    at.vint(3, (uint64_t)a.i);      // i
    at.vint(20, 2);                 // type = INT
  }
  return at.b;
}

} // namespace

// ---------------------------------------------------------------------------
// OnnxGraph
// ---------------------------------------------------------------------------

void OnnxGraph::input(const std::string &name, int dtype, const OnnxDims &dims)
{
  Pb g;
  g.str(11, valueInfo(name, dtype, dims));   // GraphProto.input
  m_inputs += g.b;
}

void OnnxGraph::output(const std::string &name, int dtype, const OnnxDims &dims)
{
  Pb g;
  g.str(12, valueInfo(name, dtype, dims));   // GraphProto.output
  m_outputs += g.b;
}

void OnnxGraph::initializer(const std::string &name, int dtype,
                            const OnnxDims &dims, const std::string &raw)
{
  Pb t;
  for (int64_t d : dims)
    t.vint(1, (uint64_t)d);         // TensorProto.dims
  t.vint(2, (uint64_t)dtype);       // data_type
  t.str(8, name);                   // name
  t.str(9, raw);                    // raw_data

  Pb g;
  g.str(5, t.b);                    // GraphProto.initializer
  m_inits += g.b;
}

void OnnxGraph::node(const std::string &opType,
                     const std::vector<std::string> &inputs,
                     const std::vector<std::string> &outputs,
                     const std::vector<OnnxAttr> &attrs)
{
  Pb n;
  for (const auto &i : inputs)
    n.str(1, i);                    // NodeProto.input
  for (const auto &o : outputs)
    n.str(2, o);                    // NodeProto.output
  n.str(3, "n" + std::to_string(m_nodeCount++));   // name
  n.str(4, opType);                 // op_type
  for (const auto &a : attrs)
    n.str(5, attribute(a));         // attribute

  Pb g;
  g.str(1, n.b);                    // GraphProto.node
  m_nodes += g.b;
}

void OnnxGraph::shapeInitializer(const std::string &name, const OnnxDims &shape)
{
  std::string raw(shape.size() * sizeof(int64_t), '\0');
  std::memcpy(&raw[0], shape.data(), raw.size());
  initializer(name, ONNX_DT_INT64, {(int64_t)shape.size()}, raw);
}

int onnxOpsetForDtype(int dtype)
{
  switch (dtype)
  {
  // Everything the backend measured before the low-precision work: expressible
  // in opset 17, which is where every recipe stays unless a type forces it up.
  case ONNX_DT_FLOAT:
  case ONNX_DT_FLOAT16:
  case ONNX_DT_BFLOAT16:
  case ONNX_DT_INT8:
  case ONNX_DT_UINT8:
  case ONNX_DT_INT32:
  case ONNX_DT_INT64:
  default:
    return 17;

  // Float8 arrived with ONNX 1.14, which is opset 19.  Anything older cannot
  // name the type at all, so the model will not load rather than run slowly.
  case ONNX_DT_FLOAT8E4M3FN:
  case ONNX_DT_FLOAT8E5M2:
    return 19;

  // Int4 arrived with ONNX 1.16, opset 21, which is also where
  // DequantizeLinear gained the block_size attribute the blocked scales need.
  case ONNX_DT_UINT4:
  case ONNX_DT_INT4:
    return 21;

  // Float4 arrived with ONNX 1.18, opset 23.
  case ONNX_DT_FLOAT4E2M1:
    return 23;
  }
}

uint32_t onnxMinOrtApiForOpset(int opset)
{
  // ORT numbers its API after its own minor version, so these are ORT minor
  // releases: 1.15 was the first to parse opset 19, 1.18 opset 21, 1.22
  // opset 23.  Anything at or below 17 predates the oldest runtime this
  // backend loads at all, so it never gates.
  if (opset >= 23) return 22;
  if (opset >= 21) return 18;
  if (opset >= 19) return 15;
  return 0;
}

void OnnxGraph::setOpset(int opset)
{
  m_opset = opset;
}

void OnnxGraph::reduceMax(const std::string &in, const std::string &out,
                          const OnnxDims &axes)
{
  // Opset 18 moved `axes` from an attribute to an optional input.  Both
  // spellings mean the same thing; which one is legal depends only on the
  // opset the model declares, so the choice belongs here rather than in every
  // recipe.  Without this, raising a recipe's opset to reach a new datatype
  // would break its reduction -- a node with nothing to do with the datatype.
  if (m_opset < 18)
  {
    node("ReduceMax", {in}, {out},
         {OnnxAttr::list("axes", axes), OnnxAttr::num("keepdims", 0)});
    return;
  }

  // The axes input is named after the output so several reductions can
  // coexist in one graph.
  const std::string axesName = out + "_axes";
  shapeInitializer(axesName, axes);
  node("ReduceMax", {in, axesName}, {out}, {OnnxAttr::num("keepdims", 0)});
}

std::string OnnxGraph::build() const
{
  // A block's initializers run to hundreds of megabytes, so the graph body is
  // never materialised as its own string: its length is computed up front and
  // the pieces are appended straight into the output.  Assembling graph-then-
  // model the obvious way would hold three copies of the weights at once.
  Pb gname;
  gname.str(2, "clpeak");           // GraphProto.name

  const size_t graphSize = m_nodes.size() + gname.b.size() + m_inits.size() +
                           m_inputs.size() + m_outputs.size();

  Pb opset;
  opset.vint(2, (uint64_t)m_opset); // OperatorSetIdProto.version (default domain)

  // The IR version gates which TensorProto datatypes may appear at all, and
  // it moves with the opset rather than independently: float8 needs IR 9,
  // int4 and the blocked quantization scales need 10, float4 needs 11.
  // Declaring a higher IR version than the types require costs nothing but
  // compatibility with older runtimes, so it tracks the opset exactly.
  const uint64_t irVersion = (m_opset >= 23) ? 11
                           : (m_opset >= 21) ? 10
                           : (m_opset >= 19) ? 9
                                             : 8;

  Pb m;
  m.b.reserve(graphSize + 64);
  m.vint(1, irVersion);             // ModelProto.ir_version
  m.str(2, "clpeak");               // producer_name
  m.tag(7, 2);                      // ModelProto.graph, length-delimited
  m.varint(graphSize);
  m.b += m_nodes;
  m.b += gname.b;
  m.b += m_inits;
  m.b += m_inputs;
  m.b += m_outputs;
  m.str(8, opset.b);                // opset_import
  return m.b;
}

// ---------------------------------------------------------------------------
// Model recipes
// ---------------------------------------------------------------------------

std::string onnxMatMulModel(int64_t M, int64_t K, int64_t N, int dtype,
                            const std::string &weightRaw)
{
  OnnxGraph g;
  g.setOpset(onnxOpsetForDtype(dtype));
  g.input("A", dtype, {M, K});
  g.initializer("B", dtype, {K, N}, weightRaw);
  g.node("MatMul", {"A", "B"}, {"C"});
  g.output("C", dtype, {M, N});
  return g.build();
}

// Zero point for a quantized element type, as its own one-byte encoding.
// int8 and the float8 types are symmetric so it is a literal zero; uint8
// centres on 128.  ONNX requires the float8 zero point to be zero, which 0x00
// is in both E4M3FN and E5M2.
static std::string quantZeroPoint(int dtype)
{
  if (dtype == ONNX_DT_UINT8)
    return std::string(1, (char)(unsigned char)128);
  return std::string(1, '\0');
}

std::string onnxQdqMatMulModel(int64_t M, int64_t K, int64_t N,
                               const std::string &weightRaw,
                               float aScale, float bScale, float cScale,
                               int actDtype, int wDtype, bool floatIo)
{
  auto f32 = [](float v) {
    std::string s(4, '\0');
    std::memcpy(&s[0], &v, 4);
    return s;
  };
  const std::string actZp = quantZeroPoint(actDtype);
  const std::string wZp   = quantZeroPoint(wDtype);

  OnnxGraph g;
  // The opset follows whichever operand type is newer: float8 cannot be named
  // below 19, int8 needs nothing beyond 17.
  g.setOpset(std::max(onnxOpsetForDtype(actDtype), onnxOpsetForDtype(wDtype)));
  if (floatIo)
    g.input("A", ONNX_DT_FLOAT, {M, K});
  else
    g.input("A_q", actDtype, {M, K});

  g.initializer("a_scale", ONNX_DT_FLOAT, {}, f32(aScale));
  g.initializer("a_zp",    actDtype, {}, actZp);
  g.initializer("B_q",     wDtype,   {K, N}, weightRaw);
  g.initializer("b_scale", ONNX_DT_FLOAT, {}, f32(bScale));
  g.initializer("b_zp",    wDtype,   {}, wZp);
  g.initializer("c_scale", ONNX_DT_FLOAT, {}, f32(cScale));
  g.initializer("c_zp",    actDtype, {}, actZp);

  // Quantize on device when the boundary must stay floating point.  Nothing
  // sits between the dequantize and the matmul either way, which is what ORT
  // needs to see to recognise a quantized matmul at all.
  if (floatIo)
    g.node("QuantizeLinear", {"A", "a_scale", "a_zp"}, {"A_q"});
  g.node("DequantizeLinear", {"A_q", "a_scale", "a_zp"}, {"A_f"});
  g.node("DequantizeLinear", {"B_q", "b_scale", "b_zp"}, {"B_f"});
  g.node("MatMul",           {"A_f", "B_f"},             {"C_f"});
  g.node("QuantizeLinear",   {"C_f", "c_scale", "c_zp"}, {"C_q"});

  if (floatIo)
  {
    g.node("DequantizeLinear", {"C_q", "c_scale", "c_zp"}, {"C"});
    g.output("C", ONNX_DT_FLOAT, {M, N});
  }
  else
  {
    g.output("C_q", actDtype, {M, N});
  }
  return g.build();
}

std::string onnxResidentNvfp4MatMulModel(int64_t M, int64_t K, int64_t N,
                                         int64_t blockSize,
                                         const std::string &aPacked,
                                         const std::string &aBlockScales,
                                         const std::string &bPacked,
                                         const std::string &bBlockScales,
                                         float globalScale)
{
  auto f32 = [](float v) {
    std::string s(4, '\0');
    std::memcpy(&s[0], &v, 4);
    return s;
  };

  OnnxGraph g;
  g.setOpset(onnxOpsetForDtype(ONNX_DT_FLOAT4E2M1));
  g.input("S", ONNX_DT_FLOAT, {});

  // A blocks along K, which is its second axis; B blocks along K, which is its
  // first.  The reduction axis is the one a group-quantized tensor groups on,
  // and it is a different axis number for each operand.
  g.initializer("A_q",  ONNX_DT_FLOAT4E2M1,   {M, K}, aPacked);
  g.initializer("A_bs", ONNX_DT_FLOAT8E4M3FN, {M, K / blockSize}, aBlockScales);
  g.initializer("B_q",  ONNX_DT_FLOAT4E2M1,   {K, N}, bPacked);
  g.initializer("B_bs", ONNX_DT_FLOAT8E4M3FN, {K / blockSize, N}, bBlockScales);
  g.initializer("gs",   ONNX_DT_FLOAT, {}, f32(globalScale));

  // Level one: the block scales are themselves quantized, and the global scale
  // dequantizes them.  Per-tensor, so no axis and no block size here.
  g.node("DequantizeLinear", {"A_bs", "gs"}, {"A_s"});
  g.node("DequantizeLinear", {"B_bs", "gs"}, {"B_s"});

  // Level two: the data, against the scales just recovered.
  g.node("DequantizeLinear", {"A_q", "A_s"}, {"A_f"},
         {OnnxAttr::num("axis", 1), OnnxAttr::num("block_size", blockSize)});
  g.node("DequantizeLinear", {"B_q", "B_s"}, {"B_f"},
         {OnnxAttr::num("axis", 0), OnnxAttr::num("block_size", blockSize)});

  g.node("MatMul", {"A_f", "B_f"}, {"C"});
  g.reduceMax("C", "R", {0});
  g.node("Mul", {"R", "S"}, {"Y"});
  g.output("Y", ONNX_DT_FLOAT, {N});
  return g.build();
}

std::string onnxResidentWeightOnlyMatMulModel(int64_t M, int64_t K, int64_t N,
                                              int wDtype, int64_t blockSize,
                                              const std::string &aRaw,
                                              const std::string &wPacked,
                                              const std::string &wScalesRaw)
{
  OnnxGraph g;
  g.setOpset(onnxOpsetForDtype(wDtype));

  // Activations are fp16 and resident, exactly as in the plain throughput
  // model: only the weights are narrow, and only the weights are what a
  // quantized model actually shrinks.
  g.input("S", ONNX_DT_FLOAT16, {});
  g.initializer("A",       ONNX_DT_FLOAT16, {M, K}, aRaw);
  g.initializer("B_q",     wDtype,          {K, N}, wPacked);
  g.initializer("b_scale", ONNX_DT_FLOAT16, {K / blockSize, N}, wScalesRaw);

  // Blocked dequantize: one scale per `blockSize` rows per column, which is
  // the axis the reduction runs along and therefore the axis a group-quantized
  // model groups on.  No zero point -- the quantization is symmetric.
  g.node("DequantizeLinear", {"B_q", "b_scale"}, {"B_f"},
         {OnnxAttr::num("axis", 0), OnnxAttr::num("block_size", blockSize)});
  g.node("MatMul", {"A", "B_f"}, {"C"});
  g.reduceMax("C", "R", {0});
  g.node("Mul", {"R", "S"}, {"Y"});
  g.output("Y", ONNX_DT_FLOAT16, {N});
  return g.build();
}

std::string onnxResidentMatMulModel(int64_t M, int64_t K, int64_t N, int dtype,
                                    const std::string &aRaw,
                                    const std::string &bRaw,
                                    bool reduceInFloat)
{
  OnnxGraph g;
  g.setOpset(onnxOpsetForDtype(dtype));
  // The scalar and the output follow the reduction, not the matmul: casting
  // the product to fp32 means everything downstream of it is fp32 too.
  const int tailDtype = reduceInFloat ? ONNX_DT_FLOAT : dtype;
  // The runtime scalar multiplies the *result*, leaving the matmul itself a
  // product of two constants -- so this graph runs correctly only while
  // constant folding stays disabled, and ONNX Runtime 1.17 accepts the
  // request to disable it and ignores it.  gemm.cpp guards against that by
  // checking the timings scale with the problem size.
  //
  // Scaling an operand instead would make the graph unfoldable outright, and
  // that was tried.  It cannot be used: the CPU provider has no fp16 kernel
  // for the multiply, so it inserts a Cast and carries out the whole matmul
  // in fp32 -- the half-precision row came back equal to the single-precision
  // one, measuring the wrong arithmetic entirely.  A guarded fold beats a
  // silent upcast.
  g.input("S", tailDtype, {});
  g.initializer("A", dtype, {M, K}, aRaw);
  g.initializer("B", dtype, {K, N}, bRaw);

  // ReduceMax, not ReduceSum: summing the rows of A*B equals multiplying the
  // summed rows of A, a rewrite an optimiser is free to make and which would
  // quietly turn this matrix multiply into a matrix-vector one.  Max does not
  // distribute over the product, so the full result has to be computed.
  g.node("MatMul", {"A", "B"}, {"C"});
  // The cast sits after the multiply, so it cannot change the arithmetic being
  // measured -- and the numeric-error row is the independent check on that: a
  // matmul quietly promoted to fp32 would report fp32's rate as well as fp32's
  // error.
  const char *reduceIn = "C";
  if (reduceInFloat)
  {
    g.node("Cast", {"C"}, {"Cf"}, {OnnxAttr::num("to", ONNX_DT_FLOAT)});
    reduceIn = "Cf";
  }
  g.reduceMax(reduceIn, "R", {0});
  g.node("Mul",    {"R", "S"}, {"Y"});
  g.output("Y", tailDtype, {N});
  return g.build();
}

std::string onnxResidentQdqMatMulModel(int64_t M, int64_t K, int64_t N,
                                       const std::string &aRaw,
                                       const std::string &bRaw,
                                       float aScale, float bScale, float cScale,
                                       int actDtype, int wDtype)
{
  // Signed activations are symmetric (zero point 0); unsigned ones centre on
  // 128.  TensorRT accepts only the former, x86 MLAS only fuses the latter.
  const std::string actZp = quantZeroPoint(actDtype);
  const std::string wZp   = quantZeroPoint(wDtype);
  auto f32 = [](float v) {
    std::string s(4, '\0');
    std::memcpy(&s[0], &v, 4);
    return s;
  };

  OnnxGraph g;
  // Every quantization scale is a build-time constant, and the runtime scalar
  // scales the reduced result instead -- exactly as in the floating-point
  // model.  A runtime scale looks tidier, since it keeps the dequantize out
  // of constant folding's reach without disabling anything, and ONNX Runtime
  // is happy with it: QLinearMatMul takes its scales as inputs.  TensorRT is
  // not.  It bakes quantization into the engine at build time, and a scale it
  // cannot see until the run means it cannot commit to integer arithmetic.
  // On an RTX 5060 that showed as 20 TOPS -- indistinguishable from the same
  // card's fp32 and a third of its fp16.  With constant scales the same test
  // reads 125 TOPS, 1.9x the fp16 rate, which is what int8 tensor cores are
  // supposed to do.  The cost is one disabled optimizer, see
  // `keepConstantsUnfolded`.
  g.setOpset(std::max(onnxOpsetForDtype(actDtype), onnxOpsetForDtype(wDtype)));
  g.input("S", ONNX_DT_FLOAT, {});
  g.initializer("A_q",     actDtype, {M, K}, aRaw);
  g.initializer("a_scale", ONNX_DT_FLOAT, {}, f32(aScale));
  g.initializer("a_zp",    actDtype, {}, actZp);
  g.initializer("B_q",     wDtype,   {K, N}, bRaw);
  g.initializer("b_scale", ONNX_DT_FLOAT, {}, f32(bScale));
  g.initializer("b_zp",    wDtype,   {}, wZp);
  g.initializer("c_scale", ONNX_DT_FLOAT, {}, f32(cScale));
  g.initializer("c_zp",    actDtype, {}, actZp);

  // Untouched DQ -> MatMul -> Q; the reduction hangs off the far side.
  g.node("DequantizeLinear", {"A_q", "a_scale", "a_zp"}, {"A_f"});
  g.node("DequantizeLinear", {"B_q", "b_scale", "b_zp"}, {"B_f"});
  g.node("MatMul",           {"A_f", "B_f"},             {"C_f"});
  g.node("QuantizeLinear",   {"C_f", "c_scale", "c_zp"}, {"C_q"});
  // Dequantize before reducing, not after.  Reducing the quantized result
  // directly saves a pass over it and ONNX Runtime accepts it, but it is not
  // the shape a quantized graph normally takes -- a QuantizeLinear is
  // followed by a DequantizeLinear -- and TensorRT rejects the short version
  // outright: "Node n4 cannot be quantized by n3.  You might want to add a DQ
  // node before n4."  The standard pattern costs one full-width pass, about a
  // fifth of the measured rate, and is what real quantized layers do anyway.
  g.node("DequantizeLinear", {"C_q", "c_scale", "c_zp"}, {"C_d"});
  g.reduceMax("C_d", "R", {0});
  g.node("Mul",              {"R", "S"}, {"Y"});
  g.output("Y", ONNX_DT_FLOAT, {N});
  return g.build();
}

std::string onnxResidentConvModel(int64_t channels, int64_t spatial,
                                  int64_t kernel, int64_t group, int dtype,
                                  const std::string &xRaw,
                                  const std::string &wRaw)
{
  const int64_t pad = (kernel - 1) / 2;   // keeps the output the same size
  const int64_t inPerGroup = channels / group;

  OnnxGraph g;
  g.input("S", dtype, {});
  g.initializer("X0", dtype, {1, channels, spatial, spatial}, xRaw);
  g.initializer("W",  dtype, {channels, inPerGroup, kernel, kernel}, wRaw);

  g.node("Mul", {"X0", "S"}, {"X"});
  g.node("Conv", {"X", "W"}, {"Y"},
         {OnnxAttr::list("kernel_shape", {kernel, kernel}),
          OnnxAttr::list("pads", {pad, pad, pad, pad}),
          OnnxAttr::list("strides", {1, 1}),
          OnnxAttr::list("dilations", {1, 1}),
          OnnxAttr::num("group", group)});

  // Down to one value per output channel: the result is otherwise as large as
  // the input and would be measured crossing the host boundary rather than
  // being computed.
  g.reduceMax("Y", "R", {0, 2, 3});
  g.output("R", dtype, {channels});
  return g.build();
}

std::string onnxResidentActivationModel(int64_t rows, int64_t cols,
                                        OnnxActivation act,
                                        const std::string &xRaw)
{
  OnnxGraph g;
  g.input("S", ONNX_DT_FLOAT16, {});
  g.initializer("X", ONNX_DT_FLOAT16, {rows, cols}, xRaw);

  std::string out = "X";
  switch (act)
  {
  case OnnxActivation::None:
    break;

  case OnnxActivation::Silu:
    // x * sigmoid(x): the gate in a SwiGLU feed-forward.
    g.node("Sigmoid", {"X"}, {"Sg"});
    g.node("Mul", {"X", "Sg"}, {"A"});
    out = "A";
    break;

  case OnnxActivation::Softmax:
    g.node("Softmax", {"X"}, {"A"}, {OnnxAttr::num("axis", -1)});
    out = "A";
    break;

  case OnnxActivation::LayerNorm:
  {
    // Opset 17 has LayerNormalization as one node, scale and bias supplied.
    std::string scale((size_t)cols * 2, '\0');
    std::string bias((size_t)cols * 2, '\0');
    uint16_t *sh = reinterpret_cast<uint16_t *>(&scale[0]);
    uint16_t *bh = reinterpret_cast<uint16_t *>(&bias[0]);
    for (int64_t i = 0; i < cols; i++)
    {
      sh[i] = floatToHalf(1.0f);
      bh[i] = floatToHalf(0.0f);
    }
    g.initializer("Ln_scale", ONNX_DT_FLOAT16, {cols}, scale);
    g.initializer("Ln_bias",  ONNX_DT_FLOAT16, {cols}, bias);
    g.node("LayerNormalization", {"X", "Ln_scale", "Ln_bias"}, {"A"},
           {OnnxAttr::num("axis", -1)});
    out = "A";
    break;
  }
  }

  g.reduceMax(out, "R", {0});
  g.node("Mul", {"R", "S"}, {"Y"});
  g.output("Y", ONNX_DT_FLOAT16, {cols});
  return g.build();
}

std::string onnxTransferModel(OnnxTransfer dir, int64_t elems)
{
  OnnxGraph g;
  switch (dir)
  {
  case OnnxTransfer::ToDevice:
    // Everything arrives; one element goes back.  Gather rather than a
    // reduction: a reduction reads the whole tensor on the device, and on a
    // provider with no real host transfer that read *is* the measurement --
    // the CPU EP reported 4 GB/s for a "transfer" that never happens, which
    // was its fp16 reduction rate and nothing else.  Picking one element
    // still forces the whole input across, because a graph input is
    // materialised in full before any kernel sees it.
    g.input("X", ONNX_DT_FLOAT16, {elems});
    {
      std::string idx(sizeof(int64_t), '\0');
      g.initializer("idx", ONNX_DT_INT64, {1}, idx);   // element 0
    }
    g.node("Gather", {"X", "idx"}, {"Y"}, {OnnxAttr::num("axis", 0)});
    g.output("Y", ONNX_DT_FLOAT16, {1});
    break;

  case OnnxTransfer::RoundTrip:
    // Squared rather than scaled: same shape on both operands avoids
    // broadcasting, which some providers decline, and needs no constant.
    g.input("X", ONNX_DT_FLOAT16, {elems});
    g.node("Mul", {"X", "X"}, {"Y"});
    g.output("Y", ONNX_DT_FLOAT16, {elems});
    break;

  case OnnxTransfer::ComputeOnly:
    // The round trip minus the return journey.  The result is picked from
    // with a Gather, not summarised with a reduction: a reduction would read
    // the whole result back on the device, and that extra pass would be
    // subtracted out of the return trip along with everything else, flattering
    // it.  Gathering one element leaves exactly the round trip's work minus
    // the journey home.
    g.input("X", ONNX_DT_FLOAT16, {elems});
    {
      std::string idx(sizeof(int64_t), '\0');
      g.initializer("idx", ONNX_DT_INT64, {1}, idx);
    }
    g.node("Mul", {"X", "X"}, {"T"});
    g.node("Gather", {"T", "idx"}, {"Y"}, {OnnxAttr::num("axis", 0)});
    g.output("Y", ONNX_DT_FLOAT16, {1});
    break;
  }
  return g.build();
}

// ---------------------------------------------------------------------------
// Transformer decoder block
// ---------------------------------------------------------------------------

namespace
{

// Deterministic weights in [-0.5, 0.5), in `dtype`.  Small magnitudes keep
// accumulation over thousands of terms away from the NaN/denormal slow paths
// raw random bit patterns would hit.
//
// Position-hashed rather than a running sequence, because onnxFillBlockedWeights
// has to visit each block twice -- once for its maximum, once to quantize
// against it -- and drawing from the same generator is what makes every
// precision row multiply the *same* matrix, differing only in how it is stored.
std::string blockWeights(int64_t rows, int64_t cols, uint32_t seed, int dtype)
{
  std::string raw((size_t)onnxElemBytes(dtype, rows * cols), '\0');
  float    *f = reinterpret_cast<float *>(&raw[0]);
  uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
  for (int64_t i = 0; i < rows; i++)
    for (int64_t j = 0; j < cols; j++)
    {
      const float   v = onnxWeightAt(i, j, seed);
      const int64_t k = i * cols + j;
      switch (dtype)
      {
      case ONNX_DT_FLOAT:    f[k] = v; break;
      case ONNX_DT_BFLOAT16: h[k] = floatToBf16(v); break;
      default:               h[k] = floatToHalf(v); break;
      }
    }
  return raw;
}

// One scalar of `dtype` (fp32, fp16 or bf16), as raw bytes.
std::string floatScalar(float v, int dtype)
{
  if (dtype == ONNX_DT_FLOAT)
  {
    std::string s(4, '\0');
    std::memcpy(&s[0], &v, 4);
    return s;
  }
  std::string s(2, '\0');
  uint16_t h = (dtype == ONNX_DT_BFLOAT16) ? floatToBf16(v) : floatToHalf(v);
  std::memcpy(&s[0], &h, 2);
  return s;
}

// The KV cache, quantized per tensor.  Values are stored spending the whole
// range and the scale halves them back, so the dequantized cache holds the
// same [-0.5, 0.5) the floating-point one does -- the row then differs from
// its fp16 sibling in how the cache was *stored* and in nothing else.
std::string quantBlockWeights(int64_t rows, int64_t cols, uint32_t seed,
                              int dtype)
{
  std::string raw((size_t)onnxElemBytes(dtype, rows * cols), '\0');
  for (int64_t i = 0; i < rows; i++)
    for (int64_t j = 0; j < cols; j++)
      onnxStoreQuantElem(&raw[0], i * cols + j, dtype,
                         onnxWeightAt(i, j, seed) * 2.0f);
  return raw;
}

// Quantization scale for a K-deep dot product's result: four sigma of the sum
// mapped onto the widest 8-bit code, exactly as gemm.cpp's qdqOutputScale
// does.  Each projection has its own K, so each gets its own.
float blockQdqScale(int64_t K)
{
  return (float)(4.0 * std::sqrt((double)K) / 3.0 / 127.0);
}

} // namespace

std::string onnxBlockModel(const OnnxBlockShape &sh)
{
  const int64_t d    = sh.dModel;
  const int64_t H    = sh.heads;
  const int64_t Dh   = sh.headDim;
  const int64_t ffn  = sh.ffnHidden;
  const int64_t S    = sh.seq;
  const bool    decode = sh.kvLen > 0;
  const int64_t ctx  = decode ? sh.kvLen : S;   // keys/values attended over
  const int     act  = sh.actDtype;

  OnnxGraph g;

  // The opset is the highest any part of this graph needs, and a narrow weight
  // type or DequantizeLinear's block_size attribute (opset 21) can raise it
  // independently of the arithmetic being measured.
  {
    int opset = std::max(onnxOpsetForDtype(act), onnxOpsetForDtype(sh.wDtype));
    if (sh.qdq)
      opset = std::max(opset, onnxOpsetForDtype(sh.qActDtype));
    // A quantized cache dequantized by a *half-precision* scale needs opset 19
    // on the strength of the scale alone.  int8 is expressible at 17 and
    // onnxOpsetForDtype says so, but DequantizeLinear did not accept anything
    // but an fp32 scale until 19, and a graph that declares 17 is refused with
    // "Type 'tensor(float16)' of input parameter (kv_scale) ... is invalid" --
    // a message about the scale, in a graph whose point is the cache.  The
    // weight-only path never hit this because its own block_size attribute
    // already forces 21.
    if (sh.kvDtype)
    {
      opset = std::max(opset, onnxOpsetForDtype(sh.kvDtype));
      if (act != ONNX_DT_FLOAT) opset = std::max(opset, 19);
    }
    if (sh.wBlock > 0) opset = std::max(opset, 21);
    g.setOpset(opset);
  }

  // The activations are a constant scaled by a runtime scalar, and the result
  // leaves as one reduced row.  Passing a [S, d] tensor in and out each run
  // costs a discrete GPU two host transfers it does not otherwise need -- 4 MB
  // against a 1.9 ms layer on an RTX 5060, about 15% -- while the weights, the
  // thing the layer is actually made of, are resident either way.  Scaling by
  // a runtime value also keeps every node downstream of it non-constant, which
  // is what the floating-point variants rely on.  The quantized ones need more
  // than that -- a weight dequantize has nothing but constants on its inputs --
  // so block.cpp disables constant folding for all of them alike.
  g.input("S", act, {});
  g.initializer("X0", act, {S, d}, blockWeights(S, d, 0xa5a5a5a5u, act));
  g.node("Mul", {"X0", "S"}, {"X"});

  // ---- Quantization constants, shared by every projection ----------------
  //
  // Present only in the W8A8 form.  Every scale is a build-time constant: a
  // runtime one keeps the dequantize out of constant folding's reach without
  // disabling an optimizer, and ONNX Runtime accepts it, but TensorRT bakes
  // quantization into the engine when it builds and a scale it cannot see
  // until the run leaves it unable to commit to integer arithmetic.  See
  // onnxResidentQdqMatMulModel, where the same choice cost 6x.
  if (sh.qdq)
  {
    // One activation scale for all seven projections, sized for the tensors
    // that actually reach them: a d-deep dot product of [-0.5, 0.5) operands.
    // The feed-forward's second input runs larger and saturates, which costs
    // nothing here -- int8 saturation is finite and takes no slow path, and
    // this row measures rate, not accuracy.  What a real deployment does
    // instead is calibrate per tensor, which no fixed graph can do.
    //
    // **The scales are fp32 and cannot be anything else.**  QuantizeLinear has
    // taken half-precision scales since opset 19, and spelling them that way
    // in a half-precision block is the obvious move -- it keeps the whole graph
    // at one width.  It produces a model that is valid right up until ORT
    // rewrites it: the QDQ selector fuses DequantizeLinear/MatMul/Quantize into
    // QLinearMatMul, which carries fp32 scales and nothing else, and the
    // rewritten graph then fails its own type check with "Type
    // 'tensor(float16)' of input parameter (U_cs) of operator (QLinearMatMul)
    // is invalid".  The message reads as though the emitted graph were broken
    // and it was not -- the same trap float8 hit from the other direction, and
    // the same lesson: check what ORT rewrote the graph into.
    //
    // Holding the fusion off with keepQdqUnfused is the wrong answer here.
    // Fusing is the entire point of an int8 row; a graph that stays unfused
    // dequantizes and multiplies in float, which the caller's fusion check
    // rejects anyway.  So the scales are fp32 and `projection` casts across
    // the boundary instead.
    g.initializer("qa_scale", ONNX_DT_FLOAT, {},
                  floatScalar(blockQdqScale(d), ONNX_DT_FLOAT));
    g.initializer("qa_zp", sh.qActDtype, {}, quantZeroPoint(sh.qActDtype));
    g.initializer("qw_scale", ONNX_DT_FLOAT, {},
                  floatScalar(onnxQuantScaleFor(sh.wDtype), ONNX_DT_FLOAT));
    g.initializer("qw_zp", sh.wDtype, {}, quantZeroPoint(sh.wDtype));
  }

  // ---- One projection, in whichever precision the shape asks for ---------
  //
  // The three forms differ here and nowhere else.  Attention, softmax, the
  // SwiGLU and the residuals stay in `actDtype` in all of them, which is both
  // what quantized inference does in practice and what keeps the rows
  // comparable: whatever separates two of them is the projection precision,
  // because nothing else moved.
  //
  // Distinct seeds so no two projections share a matrix; a repeated weight
  // would let a runtime cache or fold work that a real model cannot.
  auto projection = [&](const std::string &out, const std::string &in,
                        const std::string &w, int64_t K, int64_t N,
                        uint32_t seed)
  {
    if (sh.qdq)
    {
      // Canonical QDQ: quantize the activations, dequantize both operands
      // into the multiply, quantize the result and dequantize it back out.
      // Nothing may sit between a DequantizeLinear and the MatMul or ORT
      // stops recognising a quantized matmul, and the closing Q must be
      // followed by a DQ or TensorRT refuses to build the engine -- both
      // learned the hard way in onnxResidentQdqMatMulModel.
      std::string packed;
      packed.assign((size_t)K * (size_t)N, '\0');
      for (int64_t i = 0; i < K; i++)
        for (int64_t j = 0; j < N; j++)
          onnxStoreQuantElem(&packed[0], i * N + j, sh.wDtype,
                             onnxWeightAt(i, j, seed) * 2.0f);
      g.initializer(w + "_q", sh.wDtype, {K, N}, packed);

      const std::string cs = out + "_cs";
      g.initializer(cs, ONNX_DT_FLOAT, {},
                    floatScalar(blockQdqScale(K), ONNX_DT_FLOAT));

      // The quantization boundary is fp32 because QLinearMatMul's scales are
      // (see above), so a half-precision block casts into it and back out.
      // The casts sit *outside* the Q/DQ pattern -- before the first
      // QuantizeLinear and after the last DequantizeLinear -- so the
      // DequantizeLinear-to-MatMul adjacency ORT matches on is untouched.
      //
      // They are two passes over an activation tensor per projection, which is
      // 2 MB at the 512-token prompt against 54 GFLOP of layer, and 4 KB while
      // decoding.  Neither is measurable.  Casting the *weights* would be, and
      // is why they are quantized offline into the initializer instead.
      const std::string src = (act == ONNX_DT_FLOAT) ? in : out + "_i32";
      if (act != ONNX_DT_FLOAT)
        g.node("Cast", {in}, {src}, {OnnxAttr::num("to", ONNX_DT_FLOAT)});

      const std::string dst = (act == ONNX_DT_FLOAT) ? out : out + "_o32";
      g.node("QuantizeLinear",   {src, "qa_scale", "qa_zp"},         {out + "_aq"});
      g.node("DequantizeLinear", {out + "_aq", "qa_scale", "qa_zp"}, {out + "_af"});
      g.node("DequantizeLinear", {w + "_q", "qw_scale", "qw_zp"},    {w + "_f"});
      g.node("MatMul",           {out + "_af", w + "_f"},            {out + "_mm"});
      g.node("QuantizeLinear",   {out + "_mm", cs, "qa_zp"},         {out + "_cq"});
      g.node("DequantizeLinear", {out + "_cq", cs, "qa_zp"},         {dst});
      if (act != ONNX_DT_FLOAT)
        g.node("Cast", {dst}, {out}, {OnnxAttr::num("to", act)});
      return;
    }

    if (sh.wBlock > 0)
    {
      // Blocked weight-only: one scale per `wBlock` rows per column, along the
      // reduction axis, which is the form AWQ, GPTQ and ORT's own MatMulNBits
      // all have.  The arithmetic stays `actDtype` -- the weights are unpacked
      // on the way into the multiply -- so what the narrow type buys is weight
      // traffic, which is why the decode rows are where it shows and the
      // compute-bound prefill rows mostly are not.
      std::string packed, scales;
      onnxFillBlockedWeights(packed, scales, K, N, sh.wBlock, seed, sh.wDtype);
      g.initializer(w + "_q", sh.wDtype, {K, N}, packed);
      g.initializer(w + "_s", act, {K / sh.wBlock, N}, scales);
      g.node("DequantizeLinear", {w + "_q", w + "_s"}, {w + "_f"},
             {OnnxAttr::num("axis", 0), OnnxAttr::num("block_size", sh.wBlock)});
      g.node("MatMul", {in, w + "_f"}, {out});
      return;
    }

    g.initializer(w, act, {K, N}, blockWeights(K, N, seed, act));
    g.node("MatMul", {in, w}, {out});
  };

  // 1/sqrt(head_dim), the standard attention scale.
  g.initializer("scale", act, {},
                floatScalar(1.0f / std::sqrt((float)Dh), act));

  g.shapeInitializer("sh_heads", {S, H, Dh});
  g.shapeInitializer("sh_flat",  {S, d});

  // ---- QKV projection ----------------------------------------------------
  projection("Q",    "X", "Wq", d, d, 0x11111111u);
  projection("Knew", "X", "Wk", d, d, 0x22222222u);
  projection("Vnew", "X", "Wv", d, d, 0x33333333u);

  g.node("Reshape",   {"Q", "sh_heads"}, {"Qr"});
  g.node("Transpose", {"Qr"}, {"Qh"}, {OnnxAttr::list("perm", {1, 0, 2})});

  // ---- Attention ---------------------------------------------------------
  // Decode reads a constant cache; prefill builds K/V from this pass.  The
  // K side is stored/produced already transposed to [H, Dh, ctx] so the
  // score matmul needs no extra transpose at run time.
  //
  // The cache stays in `actDtype` in every variant.  Quantizing it is a third
  // axis and a different measurement -- an unfused dequantize would read the
  // whole cache back at full width on every token, which is the shape nobody
  // deploys.
  if (decode && sh.kvDtype && sh.kvDtype != act)
  {
    // A quantized cache, dequantized on the way into attention.  Nothing sits
    // between the DequantizeLinear and the MatMul, the arrangement ORT needs
    // to recognise a quantized matmul at all -- but there is no quantized
    // *batched* matmul for it to recognise, which is the question this row
    // exists to ask rather than an oversight.  Whether the provider folds the
    // dequantize into its attention kernel or reads the whole cache back at
    // full width every token is what the caller's fusion check reports.
    g.initializer("kv_scale", act, {},
                  floatScalar(onnxQuantScaleFor(sh.kvDtype) / 2.0f, act));
    g.initializer("kv_zp", sh.kvDtype, {}, quantZeroPoint(sh.kvDtype));
    g.initializer("Kc_q", sh.kvDtype, {H, Dh, ctx},
                  quantBlockWeights(H * Dh, ctx, 0x88888888u, sh.kvDtype));
    g.initializer("Vc_q", sh.kvDtype, {H, ctx, Dh},
                  quantBlockWeights(H * ctx, Dh, 0x99999999u, sh.kvDtype));
    g.node("DequantizeLinear", {"Kc_q", "kv_scale", "kv_zp"}, {"Kc"});
    g.node("DequantizeLinear", {"Vc_q", "kv_scale", "kv_zp"}, {"Vc"});
  }
  else if (decode)
  {
    g.initializer("Kc", act, {H, Dh, ctx},
                  blockWeights(H * Dh, ctx, 0x88888888u, act));
    g.initializer("Vc", act, {H, ctx, Dh},
                  blockWeights(H * ctx, Dh, 0x99999999u, act));
  }
  else
  {
    g.node("Reshape",   {"Knew", "sh_heads"}, {"Kr"});
    g.node("Transpose", {"Kr"}, {"Kc"}, {OnnxAttr::list("perm", {1, 2, 0})});
    g.node("Reshape",   {"Vnew", "sh_heads"}, {"Vr"});
    g.node("Transpose", {"Vr"}, {"Vc"}, {OnnxAttr::list("perm", {1, 0, 2})});
  }

  g.node("MatMul",  {"Qh", "Kc"},      {"Scores"});
  g.node("Mul",     {"Scores", "scale"}, {"ScoresS"});
  g.node("Softmax", {"ScoresS"}, {"P"}, {OnnxAttr::num("axis", -1)});
  g.node("MatMul",  {"P", "Vc"}, {"Ctx"});

  g.node("Transpose", {"Ctx"}, {"CtxT"}, {OnnxAttr::list("perm", {1, 0, 2})});
  g.node("Reshape",   {"CtxT", "sh_flat"}, {"CtxF"});
  projection("AttnOut", "CtxF", "Wo", d, d, 0x44444444u);
  g.node("Add",       {"X", "AttnOut"}, {"R1"});

  // ---- SwiGLU feed-forward ----------------------------------------------
  projection("G", "R1", "Wg", d, ffn, 0x55555555u);
  projection("U", "R1", "Wu", d, ffn, 0x66666666u);
  g.node("Sigmoid", {"G"}, {"Gs"});
  g.node("Mul",     {"G", "Gs"}, {"Act"});      // SiLU
  g.node("Mul",     {"Act", "U"}, {"Hh"});
  projection("Down", "Hh", "Wd", ffn, d, 0x77777777u);
  g.node("Add",     {"R1", "Down"}, {"Y"});

  // ReduceMax rather than ReduceSum: summing rows of a product equals
  // multiplying the summed rows, a rewrite that would let an optimiser shrink
  // the work.  See onnxResidentMatMulModel.
  g.reduceMax("Y", "Yr", {0});
  g.output("Yr", act, {d});
  if (decode)
  {
    // Keeps the K/V projections live, and mirrors the cache write a real
    // decode step performs.  Both are one row, so they cost nothing to return.
    g.output("Knew", act, {S, d});
    g.output("Vnew", act, {S, d});
  }
  return g.build();
}

// ---------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------

uint16_t floatToHalf(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t man  = x & 0x7fffffu;

  if (exp <= 0)
    return (uint16_t)sign;                      // flush to zero (inputs are ~1)
  if (exp >= 31)
    return (uint16_t)(sign | 0x7c00u);          // inf
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
}

float halfToFloat(uint16_t h)
{
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp  = (h >> 10) & 0x1fu;
  uint32_t man  = h & 0x3ffu;
  uint32_t x;

  if (exp == 0)
    x = sign;                                   // zero / denormal -> zero
  else if (exp == 31)
    x = sign | 0x7f800000u | (man << 13);       // inf / NaN
  else
    x = sign | ((exp - 15 + 127) << 23) | (man << 13);

  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

void onnxStoreNibble(void *dst, int64_t index, uint8_t nib)
{
  uint8_t *b = static_cast<uint8_t *>(dst) + (index >> 1);
  nib &= 0x0F;
  if (index & 1)
    *b = (uint8_t)((*b & 0x0F) | (nib << 4));   // odd elements: high nibble
  else
    *b = (uint8_t)((*b & 0xF0) | nib);          // even elements: low nibble
}

uint8_t onnxLoadNibble(const void *src, int64_t index)
{
  const uint8_t b = static_cast<const uint8_t *>(src)[index >> 1];
  return (index & 1) ? (uint8_t)(b >> 4) : (uint8_t)(b & 0x0F);
}

void onnxStoreInt4(void *dst, int64_t index, int q)
{
  if (q < -8) q = -8;
  if (q >  7) q =  7;
  const uint8_t nib = (uint8_t)(q & 0x0F);
  onnxStoreNibble(dst, index, nib);
}

int onnxLoadInt4(const void *src, int64_t index)
{
  const uint8_t nib = onnxLoadNibble(src, index);
  return (nib & 0x8) ? (int)nib - 16 : (int)nib;   // sign-extend from 4 bits
}

uint64_t onnxElemBytes(int dtype, int64_t count)
{
  switch (dtype)
  {
  case ONNX_DT_FLOAT:                          return (uint64_t)count * 4;
  case ONNX_DT_FLOAT16: case ONNX_DT_BFLOAT16: return (uint64_t)count * 2;
  case ONNX_DT_UINT4:   case ONNX_DT_INT4:   case ONNX_DT_FLOAT4E2M1:
    return (uint64_t)((count + 1) / 2);        // two to a byte, last one padded
  default:                                     return (uint64_t)count;
  }
}

float onnxWeightAt(int64_t i, int64_t j, uint32_t seed)
{
  uint32_t s = seed ^ (uint32_t)(i * 0x9E3779B1u) ^ (uint32_t)(j * 0x85EBCA77u);
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  return (float)(s >> 8) / 16777216.0f - 0.5f;
}

void onnxFillBlockedWeights(std::string &packed, std::string &scales,
                            int64_t K, int64_t N, int64_t blockSize,
                            uint32_t seed, int wDtype)
{
  // Each format spends its block on its own widest magnitude: int4 reaches 7,
  // int8 reaches 127, and float4's largest code is 6.
  const bool  isFp4 = (wDtype == ONNX_DT_FLOAT4E2M1);
  const bool  isI4  = (wDtype == ONNX_DT_INT4);
  const float top   = isFp4 ? 6.0f : (isI4 ? 7.0f : 127.0f);

  packed.assign((size_t)onnxElemBytes(wDtype, K * N), '\0');
  const int64_t blocks = K / blockSize;
  scales.assign((size_t)blocks * (size_t)N * 2, '\0');
  uint16_t *sc = reinterpret_cast<uint16_t *>(&scales[0]);

  for (int64_t b = 0; b < blocks; b++)
  {
    for (int64_t j = 0; j < N; j++)
    {
      float maxAbs = 0.0f;
      for (int64_t r = 0; r < blockSize; r++)
      {
        const float w = onnxWeightAt(b * blockSize + r, j, seed);
        const float a = w < 0.0f ? -w : w;
        if (a > maxAbs) maxAbs = a;
      }
      // A block of identical zeros cannot happen with this generator, but a
      // zero scale would produce NaNs on dequantize rather than a bad number.
      const float scale = (maxAbs > 0.0f) ? maxAbs / top : 1.0f;
      sc[b * N + j] = floatToHalf(scale);

      for (int64_t r = 0; r < blockSize; r++)
      {
        const int64_t i = b * blockSize + r;
        const float   w = onnxWeightAt(i, j, seed);
        const float   t = w / scale;
        const int     q = (int)(t + (t < 0.0f ? -0.5f : 0.5f));
        if (isFp4)
          onnxStoreNibble(&packed[0], i * N + j, floatToFp4E2M1(t));
        else if (isI4)
          onnxStoreInt4(&packed[0], i * N + j, q);
        else
          packed[(size_t)(i * N + j)] =
              (char)(signed char)(q < -127 ? -127 : (q > 127 ? 127 : q));
      }
    }
  }
}

void onnxFillNvfp4(std::string &packed, std::string &blockScales,
                   int64_t rows, int64_t cols, int blockAxis, int64_t blockSize,
                   float globalScale, uint32_t seed)
{
  const int64_t sRows = (blockAxis == 0) ? rows / blockSize : rows;
  const int64_t sCols = (blockAxis == 0) ? cols : cols / blockSize;
  packed.assign((size_t)onnxElemBytes(ONNX_DT_FLOAT4E2M1, rows * cols), '\0');
  blockScales.assign((size_t)sRows * (size_t)sCols, '\0');
  uint8_t *bs = reinterpret_cast<uint8_t *>(&blockScales[0]);

  for (int64_t sr = 0; sr < sRows; sr++)
  {
    for (int64_t sc = 0; sc < sCols; sc++)
    {
      const int64_t i0 = (blockAxis == 0) ? sr * blockSize : sr;
      const int64_t j0 = (blockAxis == 0) ? sc : sc * blockSize;

      float maxAbs = 0.0f;
      for (int64_t b = 0; b < blockSize; b++)
      {
        const int64_t i = (blockAxis == 0) ? i0 + b : i0;
        const int64_t j = (blockAxis == 0) ? j0 : j0 + b;
        const float   w = onnxWeightAt(i, j, seed) * 2.0f;   // over [-1, 1]
        const float   a = w < 0.0f ? -w : w;
        if (a > maxAbs) maxAbs = a;
      }

      // The block scale maps this block's peak onto float4's largest
      // magnitude, and is stored in E4M3 with the global scale divided out.
      // Rounding it there is part of the format, so the data below is
      // quantized against the value that will actually be recovered rather
      // than the one that was intended.
      const float wanted = (maxAbs > 0.0f) ? maxAbs / 6.0f : 1.0f;
      const uint8_t stored = floatToFp8E4M3(wanted / globalScale);
      bs[sr * sCols + sc] = stored;
      const float scale = fp8E4M3ToFloat(stored) * globalScale;
      const float inv   = (scale > 0.0f) ? 1.0f / scale : 0.0f;

      for (int64_t b = 0; b < blockSize; b++)
      {
        const int64_t i = (blockAxis == 0) ? i0 + b : i0;
        const int64_t j = (blockAxis == 0) ? j0 : j0 + b;
        const float   w = onnxWeightAt(i, j, seed) * 2.0f;
        onnxStoreNibble(&packed[0], i * cols + j, floatToFp4E2M1(w * inv));
      }
    }
  }
}

bool onnxIsQuantElem(int dtype)
{
  return dtype == ONNX_DT_INT8 || dtype == ONNX_DT_UINT8 ||
         dtype == ONNX_DT_FLOAT8E4M3FN || dtype == ONNX_DT_FLOAT8E5M2 ||
         dtype == ONNX_DT_FLOAT4E2M1;
}

bool onnxQdqFusionIsLegal(int dtype)
{
  return dtype == ONNX_DT_INT8 || dtype == ONNX_DT_UINT8;
}

float onnxQuantScaleFor(int dtype)
{
  // int8 stores [-127, 127] and needs 1/127 to come back to [-1, 1]; the
  // float8 types store the value itself, so their scale is one.  Both land in
  // the same dequantized range on purpose -- the accuracy rows then differ
  // only by how each format rounded, which is the comparison worth having.
  switch (dtype)
  {
  case ONNX_DT_FLOAT8E4M3FN:
  case ONNX_DT_FLOAT8E5M2:
    return 1.0f;
  // Float4's largest magnitude is 6, and it has eight of them in total.  A
  // scale of one would leave [-1, 1] using five codes out of sixteen; spending
  // the whole range and scaling back by a sixth is what makes the row a
  // measurement of the format rather than of a badly chosen scale.
  case ONNX_DT_FLOAT4E2M1:
    return 1.0f / 6.0f;
  default:
    return 1.0f / 127.0f;
  }
}

void onnxStoreQuantElem(void *dst, int64_t index, int dtype, float v)
{
  switch (dtype)
  {
  case ONNX_DT_UINT8:
    static_cast<uint8_t *>(dst)[index] = (uint8_t)(v * 127.0f + 128.0f);
    break;
  case ONNX_DT_FLOAT8E4M3FN:
    static_cast<uint8_t *>(dst)[index] = floatToFp8E4M3(v);
    break;
  case ONNX_DT_FLOAT8E5M2:
    static_cast<uint8_t *>(dst)[index] = floatToFp8E5M2(v);
    break;
  case ONNX_DT_FLOAT4E2M1:
    onnxStoreNibble(dst, index, floatToFp4E2M1(v * 6.0f));
    break;
  default:
    static_cast<int8_t *>(dst)[index] = (int8_t)(v * 127.0f);
    break;
  }
}

// E4M3FN: sign, 4 exponent bits biased by 7, 3 mantissa bits, no infinity.
// Exponent 15 with mantissa 7 is the only NaN, so the largest finite value is
// 2^8 * 1.75 = 448.
// Float4 E2M1 has eight magnitudes and nothing else, so rounding is a search
// over them rather than bit surgery.  Ties go to the even code, as IEEE
// rounding does: 0.25 lands on 0 rather than 0.5, and 5.0 on 4 rather than 6.
static const float kFp4Magnitudes[8] = {
  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
};

uint8_t floatToFp4E2M1(float f)
{
  const uint32_t sign = (f < 0.0f) ? 8u : 0u;
  float a = f < 0.0f ? -f : f;
  if (!(a == a))                       // NaN: E2M1 has none, so saturate
    a = kFp4Magnitudes[7];

  uint32_t best = 7;                   // saturating, since there is no infinity
  for (uint32_t i = 0; i < 7; i++)
  {
    const float mid = 0.5f * (kFp4Magnitudes[i] + kFp4Magnitudes[i + 1]);
    if (a < mid || (a == mid && (i & 1u) == 0u))
    {
      best = i;
      break;
    }
    if (a < kFp4Magnitudes[i + 1])
    {
      best = i + 1;
      break;
    }
  }
  return (uint8_t)(sign | best);
}

float fp4E2M1ToFloat(uint8_t code)
{
  const float mag = kFp4Magnitudes[code & 0x7u];
  return (code & 0x8u) ? -mag : mag;
}

uint8_t floatToFp8E4M3(float f)
{
  uint32_t b;
  std::memcpy(&b, &f, 4);
  const uint32_t sign = (b >> 31) & 1u;
  const uint32_t rawE = (b >> 23) & 0xFFu;
  const uint32_t man  = b & 0x7FFFFFu;

  if (rawE == 0xFFu)                       // inf or NaN: E4M3FN has only NaN
    return (uint8_t)((sign << 7) | 0x7Fu);

  int32_t  e = (int32_t)rawE - 127 + 7;
  uint32_t m;
  if (e >= 1)
  {
    m = man >> 20;                         // keep 3 mantissa bits
    const uint32_t rem = man & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u)))
    {
      if (++m == 8u) { m = 0; e++; }
    }
  }
  else
  {
    // Subnormal: shift the implicit one down into the stored mantissa.
    const int32_t shift = 1 - e;
    if (shift > 24)
      return (uint8_t)(sign << 7);         // underflows to a signed zero
    const uint32_t full  = (1u << 23) | man;
    const uint32_t drop  = (uint32_t)(20 + shift);
    m = (drop >= 32) ? 0u : (full >> drop);
    const uint32_t rem   = (drop >= 32) ? full : (full & ((1u << drop) - 1u));
    const uint32_t half  = (drop >= 32) ? 0u : (1u << (drop - 1));
    if (rem > half || (rem == half && (m & 1u)))
      m++;
    e = 0;
    if (m == 8u) { m = 0; e = 1; }
  }

  if (e > 15 || (e == 15 && m > 6))        // saturate rather than signal
    return (uint8_t)((sign << 7) | 0x7Eu); // 448, the largest finite
  return (uint8_t)((sign << 7) | ((uint32_t)e << 3) | m);
}

float fp8E4M3ToFloat(uint8_t v)
{
  const uint32_t sign = (v >> 7) & 1u;
  const uint32_t e    = (v >> 3) & 0xFu;
  const uint32_t m    = v & 0x7u;
  float mag;
  if (e == 0)
    mag = std::ldexp((float)m, -9);        // 2^-6 * m/8
  else if (e == 15u && m == 7u)
    mag = std::numeric_limits<float>::quiet_NaN();
  else
    mag = std::ldexp((float)(8u + m), (int)e - 10);   // 2^(e-7) * (8+m)/8
  return sign ? -mag : mag;
}

// E5M2 has fp16's exponent field exactly, so it is fp16 with eight mantissa
// bits rounded away -- no separate exponent handling needed.
uint8_t floatToFp8E5M2(float f)
{
  const uint16_t h  = floatToHalf(f);
  const uint16_t lo = h & 0xFFu;
  uint8_t        hi = (uint8_t)(h >> 8);
  const bool     isNaNOrInf = ((h & 0x7C00u) == 0x7C00u);
  if (!isNaNOrInf && (lo > 0x80u || (lo == 0x80u && (hi & 1u))))
    hi++;                                  // may carry into the exponent, correctly
  return hi;
}

float fp8E5M2ToFloat(uint8_t v)
{
  return halfToFloat((uint16_t)((uint16_t)v << 8));
}

float bf16ToFloat(uint16_t h)
{
  // bfloat16 is the top half of an fp32, so widening is a shift -- exact for
  // every value including the NaNs and infinities.
  const uint32_t bits = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

uint16_t floatToBf16(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  // Round to nearest even on the truncated 16 bits.
  x += 0x7fffu + ((x >> 16) & 1u);
  return (uint16_t)(x >> 16);
}

#endif // ENABLE_ONNX
