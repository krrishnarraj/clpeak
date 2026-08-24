#ifdef ENABLE_ONNX

#include "onnx_model.h"

#include <cmath>
#include <cstring>

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
  opset.vint(2, 17);                // OperatorSetIdProto.version (default domain)

  Pb m;
  m.b.reserve(graphSize + 64);
  m.vint(1, 8);                     // ModelProto.ir_version
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
  g.input("A", dtype, {M, K});
  g.initializer("B", dtype, {K, N}, weightRaw);
  g.node("MatMul", {"A", "B"}, {"C"});
  g.output("C", dtype, {M, N});
  return g.build();
}

std::string onnxQdqMatMulModel(int64_t M, int64_t K, int64_t N,
                               const std::string &weightRawInt8,
                               float aScale, float bScale, float cScale,
                               int actDtype)
{
  const std::string zeroI8(1, '\0');
  const std::string zp128(1, (char)(unsigned char)128);
  const std::string &actZp =
      (actDtype == ONNX_DT_UINT8) ? zp128 : zeroI8;
  auto f32 = [](float v) {
    std::string s(4, '\0');
    std::memcpy(&s[0], &v, 4);
    return s;
  };

  OnnxGraph g;
  g.input("A_q", actDtype, {M, K});

  g.initializer("a_scale", ONNX_DT_FLOAT, {}, f32(aScale));
  g.initializer("a_zp",    actDtype, {}, actZp);
  g.initializer("B_q",     ONNX_DT_INT8,  {K, N}, weightRawInt8);
  g.initializer("b_scale", ONNX_DT_FLOAT, {}, f32(bScale));
  g.initializer("b_zp",    ONNX_DT_INT8,  {}, zeroI8);
  g.initializer("c_scale", ONNX_DT_FLOAT, {}, f32(cScale));
  g.initializer("c_zp",    actDtype, {}, actZp);

  g.node("DequantizeLinear", {"A_q", "a_scale", "a_zp"}, {"A_f"});
  g.node("DequantizeLinear", {"B_q", "b_scale", "b_zp"}, {"B_f"});
  g.node("MatMul",           {"A_f", "B_f"},             {"C_f"});
  g.node("QuantizeLinear",   {"C_f", "c_scale", "c_zp"}, {"C_q"});

  g.output("C_q", actDtype, {M, N});
  return g.build();
}

std::string onnxResidentMatMulModel(int64_t M, int64_t K, int64_t N, int dtype,
                                    const std::string &aRaw,
                                    const std::string &bRaw)
{
  OnnxGraph g;
  g.input("S", dtype, {});                       // scalar: keeps the graph live
  g.initializer("A", dtype, {M, K}, aRaw);
  g.initializer("B", dtype, {K, N}, bRaw);

  // ReduceMax, not ReduceSum: summing the rows of A*B equals multiplying the
  // summed rows of A, a rewrite an optimiser is free to make and which would
  // quietly turn this matrix multiply into a matrix-vector one.  Max does not
  // distribute over the product, so the full result has to be computed.
  // At opset 17 its axes are an attribute; ReduceSum takes them as an input.
  g.node("MatMul",    {"A", "B"}, {"C"});
  g.node("ReduceMax", {"C"}, {"R"},
         {OnnxAttr::list("axes", {0}), OnnxAttr::num("keepdims", 0)});
  g.node("Mul",       {"R", "S"}, {"Y"});
  g.output("Y", dtype, {N});
  return g.build();
}

std::string onnxResidentQdqMatMulModel(int64_t M, int64_t K, int64_t N,
                                       const std::string &aRaw,
                                       const std::string &bRawInt8,
                                       float aScale, float bScale, float cScale,
                                       int actDtype)
{
  (void)aScale;   // supplied at run time, see below
  const std::string zeroI8(1, '\0');
  const std::string zp128(1, (char)(unsigned char)128);
  // Signed activations are symmetric (zero point 0); unsigned ones centre on
  // 128.  TensorRT accepts only the former, x86 MLAS only fuses the latter.
  const bool unsignedAct = (actDtype == ONNX_DT_UINT8);
  const std::string &actZp = unsignedAct ? zp128 : zeroI8;
  auto f32 = [](float v) {
    std::string s(4, '\0');
    std::memcpy(&s[0], &v, 4);
    return s;
  };

  OnnxGraph g;
  // The activation scale arrives at run time rather than as a constant.  That
  // is what keeps the dequantize -- and therefore the matmul below it -- out
  // of reach of constant folding, so this model needs no optimizer disabled,
  // unlike its floating-point counterpart.  QLinearMatMul takes scales as
  // inputs, so the quantized fusion is unaffected.
  g.input("S", ONNX_DT_FLOAT, {});
  g.initializer("A_q",     actDtype, {M, K}, aRaw);
  g.initializer("a_zp",    actDtype, {}, actZp);
  g.initializer("B_q",     ONNX_DT_INT8,  {K, N}, bRawInt8);
  g.initializer("b_scale", ONNX_DT_FLOAT, {}, f32(bScale));
  g.initializer("b_zp",    ONNX_DT_INT8,  {}, zeroI8);
  g.initializer("c_scale", ONNX_DT_FLOAT, {}, f32(cScale));
  g.initializer("c_zp",    actDtype, {}, actZp);

  // Untouched DQ -> MatMul -> Q; the reduction hangs off the far side.
  g.node("DequantizeLinear", {"A_q", "S", "a_zp"}, {"A_f"});
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
  g.node("ReduceMax",        {"C_d"}, {"Y"},
         {OnnxAttr::list("axes", {0}), OnnxAttr::num("keepdims", 0)});
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
  g.node("ReduceMax", {"Y"}, {"R"},
         {OnnxAttr::list("axes", {0, 2, 3}), OnnxAttr::num("keepdims", 0)});
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

  g.node("ReduceMax", {out}, {"R"},
         {OnnxAttr::list("axes", {0}), OnnxAttr::num("keepdims", 0)});
  g.node("Mul", {"R", "S"}, {"Y"});
  g.output("Y", ONNX_DT_FLOAT16, {cols});
  return g.build();
}

std::string onnxTransferModel(OnnxTransfer dir, int64_t elems,
                              const std::string &constRaw)
{
  (void)constRaw;   // no longer needed: see FromDevice below
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

  case OnnxTransfer::FromDevice:
    // A scalar arrives and is grown on the device, so the tensor is never
    // shipped in and only the return trip is timed.  Expand rather than a
    // large constant: an initializer this size makes session creation, which
    // compiles ahead of time on the NPU providers, take longer than the
    // measurement.
    g.input("S", ONNX_DT_FLOAT16, {});
    g.shapeInitializer("shape", {elems});
    g.node("Expand", {"S", "shape"}, {"Y"});
    g.output("Y", ONNX_DT_FLOAT16, {elems});
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

// Deterministic fp16 weights in [-0.5, 0.5).  Small magnitudes keep fp16
// accumulation over thousands of terms far from overflow, and avoid the
// NaN/denormal slow paths raw random bit patterns would hit.
std::string blockWeights(int64_t count, uint32_t seed)
{
  std::string raw((size_t)count * 2, '\0');
  uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
  uint32_t s = seed;
  for (int64_t i = 0; i < count; i++)
  {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    h[i] = floatToHalf((float)(s >> 8) / 16777216.0f - 0.5f);
  }
  return raw;
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

  OnnxGraph g;

  // The activations are a constant scaled by a runtime scalar, and the result
  // leaves as one reduced row.  Passing a [S, d] tensor in and out each run
  // costs a discrete GPU two host transfers it does not otherwise need -- 4 MB
  // against a 1.9 ms layer on an RTX 5060, about 15% -- while the weights, the
  // thing the layer is actually made of, are resident either way.  Scaling by
  // a runtime value also keeps every node downstream non-constant, so nothing
  // here depends on constant folding being disabled.
  g.input("S", ONNX_DT_FLOAT16, {});
  g.initializer("X0", ONNX_DT_FLOAT16, {S, d},
                blockWeights(S * d, 0xa5a5a5a5u));
  g.node("Mul", {"X0", "S"}, {"X"});

  // ---- Weights ----------------------------------------------------------
  // Distinct seeds so no two projections share a matrix; a repeated weight
  // matrix would let a runtime cache or fold work that a real model cannot.
  g.initializer("Wq", ONNX_DT_FLOAT16, {d, d},     blockWeights(d * d, 0x11111111u));
  g.initializer("Wk", ONNX_DT_FLOAT16, {d, d},     blockWeights(d * d, 0x22222222u));
  g.initializer("Wv", ONNX_DT_FLOAT16, {d, d},     blockWeights(d * d, 0x33333333u));
  g.initializer("Wo", ONNX_DT_FLOAT16, {d, d},     blockWeights(d * d, 0x44444444u));
  g.initializer("Wg", ONNX_DT_FLOAT16, {d, ffn},   blockWeights(d * ffn, 0x55555555u));
  g.initializer("Wu", ONNX_DT_FLOAT16, {d, ffn},   blockWeights(d * ffn, 0x66666666u));
  g.initializer("Wd", ONNX_DT_FLOAT16, {ffn, d},   blockWeights(ffn * d, 0x77777777u));

  // 1/sqrt(head_dim), the standard attention scale.
  {
    std::string sc(2, '\0');
    uint16_t v = floatToHalf(1.0f / std::sqrt((float)Dh));
    std::memcpy(&sc[0], &v, 2);
    g.initializer("scale", ONNX_DT_FLOAT16, {}, sc);
  }

  g.shapeInitializer("sh_heads", {S, H, Dh});
  g.shapeInitializer("sh_flat",  {S, d});

  // ---- QKV projection ----------------------------------------------------
  g.node("MatMul", {"X", "Wq"}, {"Q"});
  g.node("MatMul", {"X", "Wk"}, {"Knew"});
  g.node("MatMul", {"X", "Wv"}, {"Vnew"});

  g.node("Reshape",   {"Q", "sh_heads"}, {"Qr"});
  g.node("Transpose", {"Qr"}, {"Qh"}, {OnnxAttr::list("perm", {1, 0, 2})});

  // ---- Attention ---------------------------------------------------------
  // Decode reads a constant cache; prefill builds K/V from this pass.  The
  // K side is stored/produced already transposed to [H, Dh, ctx] so the
  // score matmul needs no extra transpose at run time.
  if (decode)
  {
    g.initializer("Kc", ONNX_DT_FLOAT16, {H, Dh, ctx},
                  blockWeights(H * Dh * ctx, 0x88888888u));
    g.initializer("Vc", ONNX_DT_FLOAT16, {H, ctx, Dh},
                  blockWeights(H * ctx * Dh, 0x99999999u));
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
  g.node("MatMul",    {"CtxF", "Wo"}, {"AttnOut"});
  g.node("Add",       {"X", "AttnOut"}, {"R1"});

  // ---- SwiGLU feed-forward ----------------------------------------------
  g.node("MatMul",  {"R1", "Wg"}, {"G"});
  g.node("MatMul",  {"R1", "Wu"}, {"U"});
  g.node("Sigmoid", {"G"}, {"Gs"});
  g.node("Mul",     {"G", "Gs"}, {"Act"});      // SiLU
  g.node("Mul",     {"Act", "U"}, {"Hh"});
  g.node("MatMul",  {"Hh", "Wd"}, {"Down"});
  g.node("Add",     {"R1", "Down"}, {"Y"});

  // ReduceMax rather than ReduceSum: summing rows of a product equals
  // multiplying the summed rows, a rewrite that would let an optimiser shrink
  // the work.  See onnxResidentMatMulModel.
  g.node("ReduceMax", {"Y"}, {"Yr"},
         {OnnxAttr::list("axes", {0}), OnnxAttr::num("keepdims", 0)});
  g.output("Yr", ONNX_DT_FLOAT16, {d});
  if (decode)
  {
    // Keeps the K/V projections live, and mirrors the cache write a real
    // decode step performs.  Both are one row, so they cost nothing to return.
    g.output("Knew", ONNX_DT_FLOAT16, {S, d});
    g.output("Vnew", ONNX_DT_FLOAT16, {S, d});
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

uint16_t floatToBf16(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  // Round to nearest even on the truncated 16 bits.
  x += 0x7fffu + ((x >> 16) & 1u);
  return (uint16_t)(x >> 16);
}

#endif // ENABLE_ONNX
