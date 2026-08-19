#ifdef ENABLE_ONNX

#include "onnx_model.h"

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
                     const std::vector<std::string> &outputs)
{
  Pb n;
  for (const auto &i : inputs)
    n.str(1, i);                    // NodeProto.input
  for (const auto &o : outputs)
    n.str(2, o);                    // NodeProto.output
  n.str(3, "n" + std::to_string(m_nodeCount++));   // name
  n.str(4, opType);                 // op_type

  Pb g;
  g.str(1, n.b);                    // GraphProto.node
  m_nodes += g.b;
}

std::string OnnxGraph::build() const
{
  Pb graph;
  graph.b += m_nodes;
  graph.str(2, "clpeak");           // GraphProto.name
  graph.b += m_inits;
  graph.b += m_inputs;
  graph.b += m_outputs;

  Pb opset;
  opset.vint(2, 17);                // OperatorSetIdProto.version (default domain)

  Pb m;
  m.vint(1, 8);                     // ModelProto.ir_version
  m.str(2, "clpeak");               // producer_name
  m.str(7, graph.b);                // graph
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
                               float aScale, float bScale, float cScale)
{
  const std::string zeroI8(1, '\0');   // symmetric quantization: zero_point 0
  auto f32 = [](float v) {
    std::string s(4, '\0');
    std::memcpy(&s[0], &v, 4);
    return s;
  };

  OnnxGraph g;
  g.input("A_q", ONNX_DT_INT8, {M, K});

  g.initializer("a_scale", ONNX_DT_FLOAT, {}, f32(aScale));
  g.initializer("a_zp",    ONNX_DT_INT8,  {}, zeroI8);
  g.initializer("B_q",     ONNX_DT_INT8,  {K, N}, weightRawInt8);
  g.initializer("b_scale", ONNX_DT_FLOAT, {}, f32(bScale));
  g.initializer("b_zp",    ONNX_DT_INT8,  {}, zeroI8);
  g.initializer("c_scale", ONNX_DT_FLOAT, {}, f32(cScale));
  g.initializer("c_zp",    ONNX_DT_INT8,  {}, zeroI8);

  g.node("DequantizeLinear", {"A_q", "a_scale", "a_zp"}, {"A_f"});
  g.node("DequantizeLinear", {"B_q", "b_scale", "b_zp"}, {"B_f"});
  g.node("MatMul",           {"A_f", "B_f"},             {"C_f"});
  g.node("QuantizeLinear",   {"C_f", "c_scale", "c_zp"}, {"C_q"});

  g.output("C_q", ONNX_DT_INT8, {M, N});
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
