#ifdef ENABLE_ONNX

#include "onnx_model.h"

#include <cstring>

// ---------------------------------------------------------------------------
// Minimal protobuf wire-format writer.  Everything an ONNX GraphProto needs
// is varints (wire type 0) and length-delimited fields (wire type 2).
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

// TypeProto for a static-shape tensor: elem_type + dims.
std::string tensorType(int dtype, std::initializer_list<int64_t> dims)
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

std::string valueInfo(const std::string &name, int dtype,
                      std::initializer_list<int64_t> dims)
{
  Pb vi;
  vi.str(1, name);                  // ValueInfoProto.name
  vi.str(2, tensorType(dtype, dims));
  return vi.b;
}

// Wrap a finished GraphProto into a ModelProto.
std::string model(const std::string &graph)
{
  Pb opset;
  opset.vint(2, 17);                // OperatorSetIdProto.version (default domain)

  Pb m;
  m.vint(1, 8);                     // ModelProto.ir_version
  m.str(2, "clpeak");               // producer_name
  m.str(7, graph);                  // graph
  m.str(8, opset.b);                // opset_import
  return m.b;
}

} // namespace

std::string onnxMatMulModel(int64_t M, int64_t K, int64_t N, int dtype,
                            const std::string &weightRaw)
{
  Pb init;                          // TensorProto B[K,N]
  init.vint(1, (uint64_t)K);        // dims
  init.vint(1, (uint64_t)N);
  init.vint(2, (uint64_t)dtype);    // data_type
  init.str(8, "B");                 // name
  init.str(9, weightRaw);           // raw_data

  Pb node;                          // NodeProto: C = MatMul(A, B)
  node.str(1, "A");
  node.str(1, "B");
  node.str(2, "C");
  node.str(3, "mm");
  node.str(4, "MatMul");

  Pb g;                             // GraphProto
  g.str(1, node.b);                 // node
  g.str(2, "clpeak_gemm");          // name
  g.str(5, init.b);                 // initializer
  g.str(11, valueInfo("A", dtype, {M, K}));   // input
  g.str(12, valueInfo("C", dtype, {M, N}));   // output

  return model(g.b);
}

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

uint16_t floatToBf16(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  // Round to nearest even on the truncated 16 bits.
  x += 0x7fffu + ((x >> 16) & 1u);
  return (uint16_t)(x >> 16);
}

#endif // ENABLE_ONNX
