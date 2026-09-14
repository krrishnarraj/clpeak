#ifdef ENABLE_COREML

#include "coreml_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// Minimal protobuf wire-format writer.  Everything a Model / MILSpec.Program
// needs is varints (wire type 0) and length-delimited fields (wire type 2).
// Map fields are repeated entry messages with the key in field 1 and the
// value in field 2, which is how protobuf spells map<string, X>.
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
  void raw(int field, const void *p, size_t n)
  {
    tag(field, 2);
    varint(n);
    b.append(static_cast<const char *>(p), n);
  }
};

std::string mapEntry(const std::string &key, const std::string &value)
{
  Pb e;
  e.str(1, key);
  e.str(2, value);
  return e.b;
}

// MILSpec.TensorType: dataType, rank, and one ConstantDimension per dim.  A
// scalar is rank 0 with no dimensions.
std::string tensorType(int dtype, const CoremlDims &dims)
{
  Pb tt;
  tt.vint(1, (uint64_t)dtype);
  tt.vint(2, dims.size());
  for (int64_t d : dims)
  {
    Pb cd;
    cd.vint(1, (uint64_t)d);        // ConstantDimension.size
    Pb dim;
    dim.str(1, cd.b);               // Dimension.constant
    tt.str(3, dim.b);               // TensorType.dimensions
  }
  return tt.b;
}

std::string valueTypeTensor(int dtype, const CoremlDims &dims)
{
  Pb vt;
  vt.str(1, tensorType(dtype, dims)); // ValueType.tensorType
  return vt.b;
}

std::string namedValueType(const std::string &name, int dtype, const CoremlDims &dims)
{
  Pb n;
  n.str(1, name);
  n.str(2, valueTypeTensor(dtype, dims));
  return n.b;
}

// MILSpec.TensorValue in its several spellings.  Which one a dtype uses is
// coremltools' convention and the runtime's expectation: fp32 in `floats`,
// int32 in `ints`, bool in `bools`, strings in `strings`, and fp16 plus every
// integer narrower than 32 bits as raw `bytes`.
std::string tvFloats(const float *v, size_t n)
{
  Pb rf;
  rf.raw(1, v, n * 4);              // RepeatedFloats.values (packed)
  Pb tv;
  tv.str(1, rf.b);
  return tv.b;
}
std::string tvInts(const std::vector<int32_t> &v)
{
  Pb inner;
  for (int32_t x : v)
    inner.varint((uint64_t)(int64_t)x);
  Pb ri;
  ri.str(1, inner.b);               // RepeatedInts.values (packed)
  Pb tv;
  tv.str(2, ri.b);
  return tv.b;
}
std::string tvBools(bool v)
{
  Pb inner;
  inner.varint(v ? 1 : 0);
  Pb rb;
  rb.str(1, inner.b);
  Pb tv;
  tv.str(3, rb.b);
  return tv.b;
}
std::string tvStrings(const std::string &s)
{
  Pb rs;
  rs.str(1, s);
  Pb tv;
  tv.str(4, rs.b);
  return tv.b;
}
std::string tvBytes(const std::string &bytes)
{
  Pb rb;
  rb.str(1, bytes);
  Pb tv;
  tv.str(7, rb.b);
  return tv.b;
}

std::string immediateValue(int dtype, const CoremlDims &dims,
                           const std::string &tensorValue)
{
  Pb iv;
  iv.str(1, tensorValue);           // ImmediateValue.tensor
  Pb v;
  v.str(2, valueTypeTensor(dtype, dims));
  v.str(3, iv.b);                   // Value.immediateValue
  return v.b;
}

std::string blobFileValue(int dtype, const CoremlDims &dims, uint64_t offset)
{
  Pb bf;
  bf.str(1, "@model_path/weights/weight.bin");
  bf.vint(2, offset);
  Pb v;
  v.str(2, valueTypeTensor(dtype, dims));
  v.str(5, bf.b);                   // Value.blobFileValue
  return v.b;
}

std::string stringValue(const std::string &s)
{
  return immediateValue(CML_STRING, {}, tvStrings(s));
}

// A scalar or small tensor as an immediate Value in `dtype`.
std::string immediateFor(int dtype, const CoremlDims &dims, const std::string &raw)
{
  switch (dtype)
  {
  case CML_FP32:
    return immediateValue(dtype, dims, tvFloats(reinterpret_cast<const float *>(raw.data()), raw.size() / 4));
  case CML_INT32:
  {
    std::vector<int32_t> v(raw.size() / 4);
    std::memcpy(v.data(), raw.data(), raw.size());
    return immediateValue(dtype, dims, tvInts(v));
  }
  default:
    return immediateValue(dtype, dims, tvBytes(raw));
  }
}

// MIL storage-format blob dtype codes (MILBlob/Blob/BlobDataType.hpp).
uint32_t blobDtype(int dtype)
{
  switch (dtype)
  {
  case CML_FP16:    return 1;
  case CML_FP32:    return 2;
  case CML_UINT8:   return 3;
  case CML_INT8:    return 4;
  case CML_BF16:    return 5;
  case CML_INT16:   return 6;
  case CML_INT4:    return 8;
  case CML_UINT4:   return 11;
  case CML_INT32:   return 14;
  case CML_FP8E4M3: return 16;
  case CML_FP8E5M2: return 17;
  default:          return 0;
  }
}

bool isNibble(int dtype) { return dtype == CML_INT4 || dtype == CML_UINT4; }

// ArrayFeatureType.ArrayDataType for a model input / output.
uint64_t featureDtype(int dtype)
{
  switch (dtype)
  {
  case CML_FP16:  return 65552;   // 0x10000 | 16
  case CML_FP32:  return 65568;   // 0x10000 | 32
  case CML_INT32: return 131104;  // 0x20000 | 32
  default:        return 0;
  }
}

uint32_t hash32(uint32_t h)
{
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return h;
}

void storeNibble(std::string &dst, int64_t index, uint8_t nib)
{
  char &b = dst[(size_t)(index >> 1)];
  nib &= 0x0f;
  if (index & 1)
    b = (char)(((uint8_t)b & 0x0f) | (nib << 4));   // odd: high nibble
  else
    b = (char)(((uint8_t)b & 0xf0) | nib);          // even: low nibble
}

} // namespace

// ---------------------------------------------------------------------------
// Small helpers declared in the header
// ---------------------------------------------------------------------------

uint64_t coremlElemBytes(int dtype, int64_t count)
{
  switch (dtype)
  {
  case CML_FP32: case CML_INT32:              return (uint64_t)count * 4;
  case CML_FP16: case CML_BF16: case CML_INT16: return (uint64_t)count * 2;
  case CML_INT4: case CML_UINT4:              return ((uint64_t)count + 1) / 2;
  default:                                    return (uint64_t)count;
  }
}

std::string coremlOpsetName(int specVersion)
{
  // coremltools: spec 6 (iOS 15) is "CoreML5", and so on.
  return "CoreML" + std::to_string(specVersion - 1);
}

int coremlSpecForDtype(int dtype)
{
  switch (dtype)
  {
  case CML_INT4: case CML_UINT4:       return 9;   // blockwise / LUT ops, iOS 18
  case CML_FP8E4M3: case CML_FP8E5M2:  return 10;  // the type arrived with iOS 26
  default:                             return 8;   // fp16 / fp32 / int8: iOS 17 floor
  }
}

std::string coremlOsForSpec(int specVersion)
{
  switch (specVersion)
  {
  case 10: return "macOS 26 / iOS 26";
  case 9:  return "macOS 15 / iOS 18";
  default: return "macOS 14 / iOS 17";
  }
}

// ---------------------------------------------------------------------------
// CoremlProgram
// ---------------------------------------------------------------------------

CoremlProgram::CoremlProgram(int specVersion) : m_spec(specVersion)
{
  // Storage-format header: count (patched as blobs are added), version 2.
  m_blob.assign(64, '\0');
  const uint32_t version = 2;
  std::memcpy(&m_blob[4], &version, 4);
}

void CoremlProgram::input(const std::string &name, int dtype, const CoremlDims &dims)
{
  m_inputs.push_back({name, dtype, dims});
}

void CoremlProgram::output(const std::string &name, int dtype, const CoremlDims &dims)
{
  m_outputs.push_back({name, dtype, dims});
}

std::string CoremlProgram::opName(const std::string &type)
{
  return type + "_" + std::to_string(m_opCount++);
}

std::string CoremlProgram::constValue(const std::string &name, int dtype,
                                      const CoremlDims &dims, const std::string &valueMsg)
{
  Pb o;
  o.str(1, "const");
  o.str(5, mapEntry("name", stringValue(name)));
  o.str(5, mapEntry("val", valueMsg));
  o.str(3, namedValueType(name, dtype, dims));
  Pb blk;
  blk.str(3, o.b);                  // Block.operations
  m_ops += blk.b;
  return name;
}

std::string CoremlProgram::blobValue(int dtype, const CoremlDims &dims, const std::string &raw)
{
  // Each blob: 64-byte metadata at a 64-byte-aligned offset, data right
  // after it (so the data is aligned too), and the header's count bumped.
  // BlobFileValue.offset is the metadata's offset.
  while (m_blob.size() % 64)
    m_blob.push_back('\0');
  const uint64_t metaOff = m_blob.size();
  const uint64_t dataOff = metaOff + 64;

  std::string meta(64, '\0');
  const uint32_t sentinel = 0xDEADBEEFu;
  const uint32_t dt = blobDtype(dtype);
  const uint64_t size = raw.size();
  uint64_t paddingBits = 0;
  if (isNibble(dtype))
  {
    int64_t count = 1;
    for (int64_t d : dims)
      count *= d;
    if (count % 2)
      paddingBits = 4;
  }
  std::memcpy(&meta[0], &sentinel, 4);
  std::memcpy(&meta[4], &dt, 4);
  std::memcpy(&meta[8], &size, 8);
  std::memcpy(&meta[16], &dataOff, 8);
  std::memcpy(&meta[24], &paddingBits, 8);
  m_blob += meta;
  m_blob += raw;

  uint32_t count;
  std::memcpy(&count, &m_blob[0], 4);
  count++;
  std::memcpy(&m_blob[0], &count, 4);

  return blobFileValue(dtype, dims, metaOff);
}

std::string CoremlProgram::constBool(const std::string &name, bool v)
{
  return constValue(name, CML_BOOL, {}, immediateValue(CML_BOOL, {}, tvBools(v)));
}

std::string CoremlProgram::constInt(const std::string &name, int32_t v)
{
  return constValue(name, CML_INT32, {}, immediateValue(CML_INT32, {}, tvInts({v})));
}

std::string CoremlProgram::constInts(const std::string &name, const std::vector<int32_t> &v)
{
  const CoremlDims dims = {(int64_t)v.size()};
  return constValue(name, CML_INT32, dims, immediateValue(CML_INT32, dims, tvInts(v)));
}

std::string CoremlProgram::constString(const std::string &name, const std::string &s)
{
  return constValue(name, CML_STRING, {}, stringValue(s));
}

std::string CoremlProgram::constFloat(const std::string &name, int dtype, float v)
{
  return constValue(name, dtype, {}, immediateFor(dtype, {}, coremlFloatScalar(v, dtype)));
}

std::string CoremlProgram::constInt8(const std::string &name, int8_t v)
{
  return constValue(name, CML_INT8, {}, immediateValue(CML_INT8, {}, tvBytes(std::string(1, (char)v))));
}

std::string CoremlProgram::constTensor(const std::string &name, int dtype,
                                       const CoremlDims &dims, const std::string &raw)
{
  return constValue(name, dtype, dims, blobValue(dtype, dims, raw));
}

void CoremlProgram::op(const std::string &type, const Inputs &ins,
                       const std::vector<Out> &outs)
{
  Pb o;
  o.str(1, type);
  for (const auto &kv : ins)
  {
    Pb bind;
    bind.str(1, kv.second);         // Argument.Binding.name
    Pb arg;
    arg.str(1, bind.b);             // Argument.arguments
    o.str(2, mapEntry(kv.first, arg.b));
  }
  for (const auto &out : outs)
    o.str(3, namedValueType(out.name, out.dtype, out.dims));
  o.str(5, mapEntry("name", stringValue(opName(type))));
  Pb blk;
  blk.str(3, o.b);
  m_ops += blk.b;
}

void CoremlProgram::affineDequantize(const Out &out, int srcDtype,
                                     const std::string &packed,
                                     const std::string &scalesRaw, int64_t scaleCount,
                                     int scaleDtype, int32_t axis)
{
  // iOS 16 op: every parameter is an attribute holding a Value.
  const CoremlDims scaleDims = scaleCount > 1 ? CoremlDims{scaleCount} : CoremlDims{};
  Pb o;
  o.str(1, "constexpr_affine_dequantize");
  o.str(5, mapEntry("name", stringValue(opName("constexpr_affine_dequantize"))));
  o.str(5, mapEntry("quantized_data", blobValue(srcDtype, out.dims, packed)));
  o.str(5, mapEntry("zero_point", immediateValue(srcDtype, {}, tvBytes(std::string(1, '\0')))));
  o.str(5, mapEntry("scale", scaleCount > 1 ? blobValue(scaleDtype, scaleDims, scalesRaw)
                                            : immediateFor(scaleDtype, {}, scalesRaw)));
  o.str(5, mapEntry("axis", immediateValue(CML_INT32, {}, tvInts({axis}))));
  o.str(3, namedValueType(out.name, out.dtype, out.dims));
  Pb blk;
  blk.str(3, o.b);
  m_ops += blk.b;
}

void CoremlProgram::blockwiseDequantize(const Out &out, int srcDtype,
                                        const CoremlDims &dataDims,
                                        const std::string &packedData, int scaleDtype,
                                        const CoremlDims &scaleDims,
                                        const std::string &scalesRaw)
{
  // iOS 18 op: parameters are inputs bound to Values.
  auto valueInput = [](Pb &o, const char *param, const std::string &value)
  {
    Pb bind;
    bind.str(2, value);             // Argument.Binding.value
    Pb arg;
    arg.str(1, bind.b);
    o.str(2, mapEntry(param, arg.b));
  };
  Pb o;
  o.str(1, "constexpr_blockwise_shift_scale");
  valueInput(o, "data", blobValue(srcDtype, dataDims, packedData));
  valueInput(o, "scale", blobValue(scaleDtype, scaleDims, scalesRaw));
  o.str(3, namedValueType(out.name, out.dtype, out.dims));
  o.str(5, mapEntry("name", stringValue(opName("constexpr_blockwise_shift_scale"))));
  Pb blk;
  blk.str(3, o.b);
  m_ops += blk.b;
}

void CoremlProgram::lutToDense(const Out &out, int nbits, const std::string &packedIndices,
                               const std::string &lutRaw)
{
  auto valueInput = [](Pb &o, const char *param, const std::string &value)
  {
    Pb bind;
    bind.str(2, value);
    Pb arg;
    arg.str(1, bind.b);
    o.str(2, mapEntry(param, arg.b));
  };
  // Per-tensor scalar palettization: the table's rank is the data's plus
  // two -- one group along every data axis, 2^nbits entries, vectors of one.
  CoremlDims lutDims(out.dims.size(), 1);
  lutDims.push_back((int64_t)1 << nbits);
  lutDims.push_back(1);
  const int idxDtype = (nbits == 4) ? CML_UINT4 : CML_UINT8;
  Pb o;
  o.str(1, "constexpr_lut_to_dense");
  valueInput(o, "indices", blobValue(idxDtype, out.dims, packedIndices));
  valueInput(o, "lut", blobValue(out.dtype, lutDims, lutRaw));
  o.str(3, namedValueType(out.name, out.dtype, out.dims));
  o.str(5, mapEntry("name", stringValue(opName("constexpr_lut_to_dense"))));
  Pb blk;
  blk.str(3, o.b);
  m_ops += blk.b;
}

std::string CoremlProgram::buildModel() const
{
  const std::string opset = coremlOpsetName(m_spec);

  // Block: outputs by name, then the operations.
  Pb block;
  for (const auto &o : m_outputs)
    block.str(2, o.name);
  block.b += m_ops;

  // Function: typed inputs, opset, one block specialization under it.
  Pb fn;
  for (const auto &i : m_inputs)
    fn.str(1, namedValueType(i.name, i.dtype, i.dims));
  fn.str(2, opset);
  fn.str(3, mapEntry(opset, block.b));

  // Program: version 1, functions {"main"}.
  Pb prog;
  prog.vint(1, 1);
  prog.str(2, mapEntry("main", fn.b));

  // ModelDescription: feature descriptions with multi-array types.
  auto feature = [](const Feature &f)
  {
    Pb arr;
    for (int64_t d : f.dims)
      arr.vint(1, (uint64_t)d);     // ArrayFeatureType.shape
    arr.vint(2, featureDtype(f.dtype));
    Pb ft;
    ft.str(5, arr.b);               // FeatureType.multiArrayType
    Pb fd;
    fd.str(1, f.name);
    fd.str(3, ft.b);
    return fd.b;
  };
  Pb desc;
  for (const auto &i : m_inputs)
    desc.str(1, feature(i));        // ModelDescription.input
  for (const auto &o : m_outputs)
    desc.str(10, feature(o));       // ModelDescription.output
  {
    Pb md;
    md.str(1, "clpeak micro-graph");
    md.str(2, "1");
    md.str(3, "clpeak");
    desc.str(100, md.b);            // ModelDescription.metadata
  }

  Pb model;
  model.vint(1, (uint64_t)m_spec);  // Model.specificationVersion
  model.str(2, desc.b);             // Model.description
  model.str(502, prog.b);           // Model.mlProgram
  return model.b;
}

// ---------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------

uint16_t coremlFloatToHalf(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  const uint32_t man = x & 0x7fffffu;
  if (exp <= 0)
    return (uint16_t)sign;                     // flush to zero
  if (exp >= 31)
    return (uint16_t)(sign | 0x7c00u);         // inf
  // Round to nearest even on the 13 dropped bits.
  uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13);
  const uint32_t rem = man & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (h & 1)))
    h++;
  return (uint16_t)h;
}

float coremlHalfToFloat(uint16_t h)
{
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1fu;
  const uint32_t man = h & 0x3ffu;
  uint32_t x;
  if (exp == 0)
  {
    if (man == 0)
      x = sign;
    else
    {
      // Subnormal: normalise.
      uint32_t m = man;
      int32_t e = 127 - 15 + 1;
      while (!(m & 0x400u))
      {
        m <<= 1;
        e--;
      }
      m &= 0x3ffu;
      x = sign | ((uint32_t)e << 23) | (m << 13);
    }
  }
  else if (exp == 31)
    x = sign | 0x7f800000u | (man << 13);
  else
    x = sign | ((exp - 15 + 127) << 23) | (man << 13);
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

uint16_t coremlFloatToBf16(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t lsb = (x >> 16) & 1u;
  x += 0x7fffu + lsb;                          // round to nearest even
  return (uint16_t)(x >> 16);
}

// Float8 E4M3FN: bias 7, no infinity, 448 the largest finite value.
uint8_t coremlFloatToFp8E4M3(float f)
{
  if (std::isnan(f))
    return 0x7f;
  const uint8_t sign = f < 0 ? 0x80 : 0;
  float a = std::fabs(f);
  if (a > 448.0f)
    a = 448.0f;                                // saturate rather than overflow
  if (a < 0.0009765625f)                       // below the smallest subnormal / 2
    return sign;
  int e;
  const float m = std::frexp(a, &e);           // a = m * 2^e, m in [0.5, 1)
  // Normal range: exponent field 1..15 covers 2^-6 .. 2^8.
  int expField = e - 1 + 7;
  uint8_t code;
  if (expField >= 1)
  {
    // Three mantissa bits of (2m - 1), rounded to nearest even.
    float frac = (m * 2.0f - 1.0f) * 8.0f;
    int mant = (int)std::floor(frac);
    const float rem = frac - (float)mant;
    if (rem > 0.5f || (rem == 0.5f && (mant & 1)))
      mant++;
    if (mant == 8)
    {
      mant = 0;
      expField++;
    }
    if (expField > 15 || (expField == 15 && mant == 7))
    {
      expField = 15;
      mant = 6;                                // 448
    }
    code = (uint8_t)((expField << 3) | mant);
  }
  else
  {
    // Subnormal: value = mant * 2^-9.
    int mant = (int)std::lround(a / 0.001953125f);
    if (mant > 7)
      mant = 7;
    code = (uint8_t)mant;
  }
  return (uint8_t)(sign | code);
}

float coremlFp8E4M3ToFloat(uint8_t v)
{
  const float sign = (v & 0x80) ? -1.0f : 1.0f;
  const int expField = (v >> 3) & 0xf;
  const int mant = v & 7;
  if (expField == 0)
    return sign * (float)mant * 0.001953125f;  // 2^-9
  if (expField == 15 && mant == 7)
    return std::nanf("");
  return sign * std::ldexp(1.0f + (float)mant / 8.0f, expField - 7);
}

std::string coremlFloatScalar(float v, int dtype)
{
  std::string s;
  if (dtype == CML_FP32)
  {
    s.resize(4);
    std::memcpy(&s[0], &v, 4);
  }
  else
  {
    const uint16_t h = (dtype == CML_BF16) ? coremlFloatToBf16(v) : coremlFloatToHalf(v);
    s.resize(2);
    std::memcpy(&s[0], &h, 2);
  }
  return s;
}

std::string coremlFillFloats(int dtype, int64_t count, uint32_t seed, float magnitude)
{
  std::string raw((size_t)coremlElemBytes(dtype, count), '\0');
  float *f = reinterpret_cast<float *>(&raw[0]);
  uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
  uint32_t s = seed ? seed : 0x9e3779b9u;
  for (int64_t i = 0; i < count; i++)
  {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    const float v = ((float)(s >> 8) / 16777216.0f - 0.5f) * magnitude;
    if (dtype == CML_FP32)
      f[i] = v;
    else if (dtype == CML_BF16)
      h[i] = coremlFloatToBf16(v);
    else
      h[i] = coremlFloatToHalf(v);
  }
  return raw;
}

float coremlWeightAt(int64_t i, int64_t j, uint32_t seed)
{
  const uint32_t h = hash32((uint32_t)i * 2654435761u ^ hash32((uint32_t)j * 2246822519u ^ seed));
  return (float)(h & 0xffffffu) / 16777216.0f - 0.5f;
}

// ---------------------------------------------------------------------------
// Weight formats
// ---------------------------------------------------------------------------

int coremlActDtype(CoremlWeight w)
{
  switch (w)
  {
  case CoremlWeight::Fp32: return CML_FP32;
  case CoremlWeight::Bf16: return CML_BF16;
  default:                 return CML_FP16;
  }
}

int coremlStoredDtype(CoremlWeight w)
{
  switch (w)
  {
  case CoremlWeight::Fp32:        return CML_FP32;
  case CoremlWeight::Fp16:        return CML_FP16;
  case CoremlWeight::Bf16:        return CML_BF16;
  case CoremlWeight::Int8Channel: return CML_INT8;
  case CoremlWeight::Int8Qdq:     return CML_INT8;
  case CoremlWeight::Int4Block:   return CML_INT4;
  case CoremlWeight::Int4Lut:     return CML_UINT4;
  case CoremlWeight::Fp8Block:    return CML_FP8E4M3;
  }
  return CML_FP16;
}

uint64_t coremlWeightBytes(CoremlWeight w, int64_t K, int64_t N)
{
  const uint64_t n = (uint64_t)K * (uint64_t)N;
  switch (w)
  {
  case CoremlWeight::Int8Channel:
  case CoremlWeight::Int8Qdq:
    return n + (uint64_t)N * 2;                                   // per-column fp16 scale
  case CoremlWeight::Int4Block:
    return (n + 1) / 2 + (n / (uint64_t)kCoremlWeightBlock) * 2;  // per-block fp16 scale
  case CoremlWeight::Int4Lut:
    return (n + 1) / 2 + 16 * 2;                                  // one 16-entry table
  case CoremlWeight::Fp8Block:
    return n + (n / (uint64_t)kCoremlWeightBlock) * 2;
  default:
    return coremlElemBytes(coremlStoredDtype(w), (int64_t)n);
  }
}

int coremlSpecForWeight(CoremlWeight w)
{
  switch (w)
  {
  case CoremlWeight::Int4Block:
  case CoremlWeight::Int4Lut:   return 9;
  case CoremlWeight::Fp8Block:  return 10;
  default:                      return 8;
  }
}

int coremlSpecNeeded(CoremlWeight w, bool fusedAttention)
{
  return std::max(coremlSpecForWeight(w), fusedAttention ? 9 : 8);
}

const char *coremlWeightLabel(CoremlWeight w)
{
  switch (w)
  {
  case CoremlWeight::Fp32:        return "fp32";
  case CoremlWeight::Fp16:        return "fp16";
  case CoremlWeight::Bf16:        return "bf16";
  case CoremlWeight::Int8Channel: return "int8_weight";
  case CoremlWeight::Int4Block:   return "int4_weight";
  case CoremlWeight::Int4Lut:     return "int4_lut";
  case CoremlWeight::Fp8Block:    return "fp8_weight";
  case CoremlWeight::Int8Qdq:     return "int8_qdq";
  }
  return "?";
}

float coremlQdqActScale(float magnitude)
{
  // Inputs are uniform in [-0.5, 0.5) * magnitude; map the edge onto 127.
  return 0.5f * magnitude / 127.0f;
}

float coremlQdqOutScale(int64_t K, float magnitude)
{
  // Four sigma of a K-deep dot product of two such operands (variance
  // magnitude^2 / 12 each) onto the widest code.
  return (float)(magnitude * magnitude * std::sqrt((double)K) / 3.0 / 127.0);
}

namespace
{

// Round a scale to the width it will be stored in, so the codes are
// quantized against the value that actually comes back.
float storedScale(float s, int dtype)
{
  return dtype == CML_FP32 ? s : coremlHalfToFloat(coremlFloatToHalf(s));
}

// The value a decompressed weight takes for the accuracy reference: the
// exact product of the stored scale and the code.  Rounding it to fp16
// first -- on the theory that a unit decompresses into fp16 before it
// multiplies -- made the Neural Engine's int8 row read *worse* (348 ppm
// against 294 on the M1 Pro): the ANE keeps the codes and applies the scale
// after the accumulation, so the exact product is what it computes.  A unit
// that does round to fp16 first reads a little above its fp16 row instead,
// which the row descriptions say.
float dequantizedTo(float scale, float code, int dtype)
{
  (void)dtype;
  return scale * code;
}

} // namespace

std::string coremlEmitWeight(CoremlProgram &p, const std::string &outName,
                             CoremlWeight w, int64_t K, int64_t N, uint32_t seed,
                             float magnitude, std::vector<float> *dequantized)
{
  const int act = coremlActDtype(w);
  const CoremlDims dims = {K, N};
  const size_t count = (size_t)K * (size_t)N;
  if (dequantized)
    dequantized->assign(count, 0.0f);

  auto value = [&](int64_t i, int64_t j) { return coremlWeightAt(i, j, seed) * magnitude; };

  switch (w)
  {
  case CoremlWeight::Fp32:
  case CoremlWeight::Fp16:
  case CoremlWeight::Bf16:
  {
    std::string raw((size_t)coremlElemBytes(act, (int64_t)count), '\0');
    float *f = reinterpret_cast<float *>(&raw[0]);
    uint16_t *h = reinterpret_cast<uint16_t *>(&raw[0]);
    for (int64_t i = 0; i < K; i++)
      for (int64_t j = 0; j < N; j++)
      {
        const float v = value(i, j);
        const size_t k = (size_t)i * N + j;
        float stored = v;
        if (act == CML_FP32)
          f[k] = v;
        else if (act == CML_BF16)
        {
          h[k] = coremlFloatToBf16(v);
          stored = v; // no bf16 arithmetic exists to compare against
        }
        else
        {
          h[k] = coremlFloatToHalf(v);
          stored = coremlHalfToFloat(h[k]);
        }
        if (dequantized)
          (*dequantized)[k] = stored;
      }
    return p.constTensor(outName, act, dims, raw);
  }

  case CoremlWeight::Int8Channel:
  case CoremlWeight::Int8Qdq:
  {
    // One symmetric scale per output column: the column's largest magnitude
    // onto 127.
    std::string packed(count, '\0');
    std::string scales((size_t)coremlElemBytes(act, N), '\0');
    for (int64_t j = 0; j < N; j++)
    {
      float m = 0.0f;
      for (int64_t i = 0; i < K; i++)
        m = std::max(m, std::fabs(value(i, j)));
      const float scale = storedScale(m > 0.0f ? m / 127.0f : 1.0f, act);
      std::memcpy(&scales[(size_t)coremlElemBytes(act, j)], coremlFloatScalar(scale, act).data(),
                  (size_t)coremlElemBytes(act, 1));
      for (int64_t i = 0; i < K; i++)
      {
        int q = (int)std::lround(value(i, j) / scale);
        q = std::max(-127, std::min(127, q));
        packed[(size_t)i * N + j] = (char)(int8_t)q;
        if (dequantized)
          (*dequantized)[(size_t)i * N + j] = dequantizedTo(scale, (float)q, act);
      }
    }
    p.affineDequantize({outName, act, dims}, CML_INT8, packed, scales, N, act, 1);
    return outName;
  }

  case CoremlWeight::Int4Block:
  case CoremlWeight::Fp8Block:
  {
    // One symmetric scale per block of kCoremlWeightBlock rows in each column.
    const int64_t B = kCoremlWeightBlock;
    const int64_t nb = K / B;
    const bool fp8 = (w == CoremlWeight::Fp8Block);
    std::string packed(fp8 ? count : (count + 1) / 2, '\0');
    std::string scales((size_t)coremlElemBytes(act, nb * N), '\0');
    const double top = fp8 ? 448.0 : 7.0;
    for (int64_t b = 0; b < nb; b++)
      for (int64_t j = 0; j < N; j++)
      {
        float m = 0.0f;
        for (int64_t i = b * B; i < (b + 1) * B; i++)
          m = std::max(m, std::fabs(value(i, j)));
        const float scale = storedScale(m > 0.0f ? (float)(m / top) : 1.0f, act);
        std::memcpy(&scales[(size_t)coremlElemBytes(act, b * N + j)],
                    coremlFloatScalar(scale, act).data(), (size_t)coremlElemBytes(act, 1));
        for (int64_t i = b * B; i < (b + 1) * B; i++)
        {
          const size_t k = (size_t)i * N + j;
          const float v = value(i, j) / scale;
          if (fp8)
          {
            const uint8_t code = coremlFloatToFp8E4M3(v);
            packed[k] = (char)code;
            if (dequantized)
              (*dequantized)[k] = dequantizedTo(scale, coremlFp8E4M3ToFloat(code), act);
          }
          else
          {
            int q = (int)std::lround(v);
            q = std::max(-8, std::min(7, q));
            storeNibble(packed, (int64_t)k, (uint8_t)(q & 0xf));
            if (dequantized)
              (*dequantized)[k] = dequantizedTo(scale, (float)q, act);
          }
        }
      }
    p.blockwiseDequantize({outName, act, dims}, fp8 ? CML_FP8E4M3 : CML_INT4, dims, packed,
                          act, {nb, N}, scales);
    return outName;
  }

  case CoremlWeight::Int4Lut:
  {
    // One 16-entry table for the whole tensor: the sixteen levels of a
    // symmetric 4-bit grid over the tensor's largest magnitude, so the row
    // stores the same grid the blockwise row does with one scale instead of
    // one per block -- the difference between the two rows is then the
    // lookup itself.
    float m = 0.0f;
    for (int64_t i = 0; i < K; i++)
      for (int64_t j = 0; j < N; j++)
        m = std::max(m, std::fabs(value(i, j)));
    const float scale = storedScale(m > 0.0f ? m / 7.0f : 1.0f, act);
    std::string lut((size_t)coremlElemBytes(act, 16), '\0');
    float levels[16];
    for (int q = 0; q < 16; q++)
    {
      levels[q] = scale * (float)(q - 8);
      std::memcpy(&lut[(size_t)coremlElemBytes(act, q)], coremlFloatScalar(levels[q], act).data(),
                  (size_t)coremlElemBytes(act, 1));
      if (act == CML_FP16)
        levels[q] = coremlHalfToFloat(coremlFloatToHalf(levels[q]));
    }
    std::string packed((count + 1) / 2, '\0');
    for (int64_t i = 0; i < K; i++)
      for (int64_t j = 0; j < N; j++)
      {
        const size_t k = (size_t)i * N + j;
        int q = (int)std::lround(value(i, j) / scale);
        q = std::max(-8, std::min(7, q));
        storeNibble(packed, (int64_t)k, (uint8_t)(q + 8));
        if (dequantized)
          (*dequantized)[k] = levels[q + 8];
      }
    p.lutToDense({outName, act, dims}, 4, packed, lut);
    return outName;
  }
  }
  return outName;
}

void coremlEmitProjection(CoremlProgram &p, const std::string &out,
                          const std::string &in, int64_t M, int64_t K, int64_t N,
                          CoremlWeight w, uint32_t seed, float magnitude,
                          std::vector<float> *dequantized)
{
  const int act = coremlActDtype(w);
  const std::string wName = coremlEmitWeight(p, out + "_w", w, K, N, seed, magnitude, dequantized);

  if (w != CoremlWeight::Int8Qdq)
  {
    p.op("matmul",
         {{"x", in}, {"y", wName},
          {"transpose_x", p.constBool(out + "_tx", false)},
          {"transpose_y", p.constBool(out + "_ty", false)}},
         {out, act, {M, N}});
    return;
  }

  // W8A8: quantize the activations, dequantize into the multiply, quantize
  // the result and dequantize it back out -- the shape a quantized layer has,
  // and the pattern Core ML's compiler fuses into an integer multiply on
  // hardware that has one.
  const std::string aScale = p.constFloat(out + "_as", act, coremlQdqActScale(magnitude));
  const std::string cScale = p.constFloat(out + "_cs", act, coremlQdqOutScale(K, magnitude));
  const std::string zp = p.constInt8(out + "_zp", 0);
  const std::string dt = p.constString(out + "_dt", "int8");
  p.op("quantize", {{"input", in}, {"scale", aScale}, {"zero_point", zp}, {"output_dtype", dt}},
       {out + "_aq", CML_INT8, {M, K}});
  p.op("dequantize", {{"input", out + "_aq"}, {"scale", aScale}, {"zero_point", zp}},
       {out + "_af", act, {M, K}});
  p.op("matmul",
       {{"x", out + "_af"}, {"y", wName},
        {"transpose_x", p.constBool(out + "_tx", false)},
        {"transpose_y", p.constBool(out + "_ty", false)}},
       {out + "_mm", act, {M, N}});
  p.op("quantize", {{"input", out + "_mm"}, {"scale", cScale}, {"zero_point", zp}, {"output_dtype", dt}},
       {out + "_cq", CML_INT8, {M, N}});
  p.op("dequantize", {{"input", out + "_cq"}, {"scale", cScale}, {"zero_point", zp}},
       {out, act, {M, N}});
}

// ---------------------------------------------------------------------------
// Recipes
// ---------------------------------------------------------------------------

namespace
{

// reduce_max over `axes` of `in`, emitted with its two constant parameters.
void reduceMax(CoremlProgram &p, const std::string &out, const std::string &in,
               const std::vector<int32_t> &axes, bool keepDims, int dtype,
               const CoremlDims &outDims)
{
  p.op("reduce_max",
       {{"x", in}, {"axes", p.constInts(out + "_axes", axes)},
        {"keep_dims", p.constBool(out + "_kd", keepDims)}},
       {out, dtype, outDims});
}

} // namespace

CoremlProgram coremlResidentMatMulModel(int spec, int64_t M, int64_t K, int64_t N,
                                        CoremlWeight w, bool resultScaled)
{
  (void)spec;
  CoremlProgram p(coremlSpecNeeded(w));
  const int act = coremlActDtype(w);
  const bool qdq = (w == CoremlWeight::Int8Qdq);
  // Bf16 has no arithmetic to scale in; the attempt fails at the multiply,
  // which is the answer wanted.  Its scalar rides in as fp16.
  const int ioDtype = (act == CML_BF16) ? CML_FP16 : act;

  p.input("s", ioDtype, {1});
  p.constTensor("A", act, {M, K}, coremlFillFloats(act, M * K, 0x243f6a88u));
  std::string x = "A";
  if (!resultScaled || qdq)
  {
    p.op("mul", {{"x", "A"}, {"y", "s"}}, {"xs", act, {M, K}});
    x = "xs";
  }
  coremlEmitProjection(p, "y", x, M, K, N, w, 0x85a308d3u, 1.0f);
  if (resultScaled && !qdq)
  {
    reduceMax(p, "r", "y", {0}, true, act, {1, N});
    p.op("mul", {{"x", "r"}, {"y", "s"}}, {"out", ioDtype, {1, N}});
  }
  else
    reduceMax(p, "out", "y", {0}, true, act, {1, N});
  p.output("out", ioDtype, {1, N});
  return p;
}

CoremlProgram coremlPlainMatMulModel(int spec, int64_t M, int64_t K, int64_t N,
                                     CoremlWeight w, std::vector<float> *dequantized)
{
  (void)spec;
  CoremlProgram p(coremlSpecNeeded(w));
  const int act = coremlActDtype(w);
  const int ioDtype = (act == CML_BF16) ? CML_FP16 : act;
  p.input("x", ioDtype, {M, K});
  std::string in = "x";
  if (act == CML_BF16)
  {
    p.op("cast", {{"x", "x"}, {"dtype", p.constString("cast_dt", "bfloat16")}}, {"xb", act, {M, K}});
    in = "xb";
  }
  coremlEmitProjection(p, "y", in, M, K, N, w, 0x85a308d3u, 1.0f, dequantized);
  p.output("y", act == CML_BF16 ? act : ioDtype, {M, N});
  return p;
}

CoremlProgram coremlGemvModel(int spec, int64_t d, int64_t cols, uint32_t seed)
{
  (void)spec;
  CoremlProgram p(coremlSpecNeeded(CoremlWeight::Fp16));
  // Uniform over +/-sqrt(3/d) so the result keeps the vector's magnitude: a
  // d-deep fp16 dot product of larger values would saturate.
  const float lim = std::sqrt(3.0f / (float)d);
  p.input("x", CML_FP16, {1, d});
  p.constTensor("W", CML_FP16, {d, cols}, coremlFillFloats(CML_FP16, d * cols, seed, 2.0f * lim));
  p.op("matmul",
       {{"x", "x"}, {"y", "W"}, {"transpose_x", p.constBool("tx", false)},
        {"transpose_y", p.constBool("ty", false)}},
       {"y", CML_FP16, {1, cols}});
  p.output("y", CML_FP16, {1, cols});
  return p;
}

CoremlProgram coremlConvModel(int spec, int64_t channels, int64_t spatial,
                              int64_t kernel, int64_t group, int dtype)
{
  (void)spec;
  CoremlProgram p(coremlSpecNeeded(CoremlWeight::Fp16));
  const int64_t inPerGroup = channels / group;
  p.input("s", dtype, {1});
  p.constTensor("X0", dtype, {1, channels, spatial, spatial},
                coremlFillFloats(dtype, channels * spatial * spatial, 0x9e3779b9u));
  p.op("mul", {{"x", "X0"}, {"y", "s"}}, {"X", dtype, {1, channels, spatial, spatial}});
  p.constTensor("Wt", dtype, {channels, inPerGroup, kernel, kernel},
                coremlFillFloats(dtype, channels * inPerGroup * kernel * kernel, 0x7f4a7c15u,
                                 // Keep a 3x3 x 256 sum from growing past
                                 // fp16 range: shrink the weights with the
                                 // fan-in.
                                 2.0f / std::sqrt((float)(inPerGroup * kernel * kernel))));
  // `pad` is documented as custom-padding only, but the parser requires it
  // whatever `pad_type` says ("Required param 'pad' is missing"); coremltools
  // always emits the zeros.
  p.op("conv",
       {{"x", "X"}, {"weight", "Wt"},
        {"strides", p.constInts("strides", {1, 1})},
        {"pad_type", p.constString("pad_type", "same")},
        {"pad", p.constInts("pad", {0, 0, 0, 0})},
        {"dilations", p.constInts("dilations", {1, 1})},
        {"groups", p.constInt("groups", (int32_t)group)}},
       {"Y", dtype, {1, channels, spatial, spatial}});
  reduceMax(p, "out", "Y", {2, 3}, false, dtype, {1, channels});
  p.output("out", dtype, {1, channels});
  return p;
}

CoremlProgram coremlActivationModel(int spec, int64_t rows, int64_t cols,
                                    CoremlActivation act)
{
  (void)spec;
  CoremlProgram p(coremlSpecNeeded(CoremlWeight::Fp16));
  p.input("s", CML_FP16, {1});
  p.constTensor("X0", CML_FP16, {rows, cols}, coremlFillFloats(CML_FP16, rows * cols, 0x6a09e667u, 4.0f));
  p.op("mul", {{"x", "X0"}, {"y", "s"}}, {"X", CML_FP16, {rows, cols}});
  std::string y = "X";
  switch (act)
  {
  case CoremlActivation::None:
    break;
  case CoremlActivation::Silu:
    p.op("silu", {{"x", "X"}}, {"Y", CML_FP16, {rows, cols}});
    y = "Y";
    break;
  case CoremlActivation::Softmax:
    p.op("softmax", {{"x", "X"}, {"axis", p.constInt("axis", -1)}}, {"Y", CML_FP16, {rows, cols}});
    y = "Y";
    break;
  case CoremlActivation::LayerNorm:
    p.op("layer_norm",
         {{"x", "X"}, {"axes", p.constInts("axes", {1})},
          {"epsilon", p.constFloat("eps", CML_FP16, 1.0e-5f)}},
         {"Y", CML_FP16, {rows, cols}});
    y = "Y";
    break;
  }
  reduceMax(p, "out", y, {0}, true, CML_FP16, {1, cols});
  p.output("out", CML_FP16, {1, cols});
  return p;
}

CoremlProgram coremlTransferModel(int spec, CoremlTransfer dir, int64_t rows, int64_t K,
                                  int64_t N, bool resultScaled)
{
  (void)spec;
  CoremlProgram p(coremlSpecNeeded(CoremlWeight::Fp16));
  p.constTensor("W", CML_FP16, {K, N}, coremlFillFloats(CML_FP16, K * N, 0x85a308d3u));
  std::string x;
  if (dir == CoremlTransfer::Resident)
  {
    p.input("s", CML_FP16, {1});
    p.constTensor("X0", CML_FP16, {rows, K}, coremlFillFloats(CML_FP16, rows * K, 0x243f6a88u));
    x = "X0";
    if (!resultScaled)
    {
      p.op("mul", {{"x", "X0"}, {"y", "s"}}, {"X", CML_FP16, {rows, K}});
      x = "X";
    }
  }
  else
  {
    p.input("x", CML_FP16, {rows, K});
    x = "x";
  }
  p.op("matmul",
       {{"x", x}, {"y", "W"}, {"transpose_x", p.constBool("tx", false)},
        {"transpose_y", p.constBool("ty", false)}},
       {"y", CML_FP16, {rows, N}});
  if (dir == CoremlTransfer::RoundTrip)
  {
    p.output("y", CML_FP16, {rows, N});
    return p;
  }
  if (dir == CoremlTransfer::Resident && resultScaled)
  {
    reduceMax(p, "r", "y", {0}, true, CML_FP16, {1, N});
    p.op("mul", {{"x", "r"}, {"y", "s"}}, {"out", CML_FP16, {1, N}});
  }
  else
    reduceMax(p, "out", "y", {0}, true, CML_FP16, {1, N});
  p.output("out", CML_FP16, {1, N});
  return p;
}

CoremlProgram coremlTrivialModel(int spec, int64_t rows, int64_t cols, uint32_t salt)
{
  (void)spec;
  CoremlProgram p(coremlSpecNeeded(CoremlWeight::Fp16));
  p.input("x", CML_FP16, {rows, cols});
  p.constTensor("K", CML_FP16, {rows, cols}, coremlFillFloats(CML_FP16, rows * cols, salt | 1u, 2.0f));
  p.op("mul", {{"x", "x"}, {"y", "K"}}, {"y", CML_FP16, {rows, cols}});
  p.output("y", CML_FP16, {rows, cols});
  return p;
}

CoremlProgram coremlSmallMatMulModel(int spec, int64_t d)
{
  (void)spec;
  CoremlProgram p(coremlSpecNeeded(CoremlWeight::Fp16));
  p.input("x", CML_FP16, {d, d});
  p.constTensor("W", CML_FP16, {d, d}, coremlFillFloats(CML_FP16, d * d, 0x243f6a88u));
  p.op("matmul",
       {{"x", "x"}, {"y", "W"}, {"transpose_x", p.constBool("tx", false)},
        {"transpose_y", p.constBool("ty", false)}},
       {"y", CML_FP16, {d, d}});
  p.output("y", CML_FP16, {d, d});
  return p;
}

// ---------------------------------------------------------------------------
// Transformer decoder block
// ---------------------------------------------------------------------------

CoremlProgram coremlBlockModel(int spec, const CoremlBlockShape &sh)
{
  (void)spec;
  CoremlProgram p(coremlSpecNeeded(sh.weights, sh.fusedAttention));
  const int64_t d = sh.dModel;
  const int64_t H = sh.heads;
  const int64_t Dh = sh.headDim;
  const int64_t ffn = sh.ffnHidden;
  const int64_t S = sh.seq;
  const bool decode = sh.kvLen > 0;
  const int64_t ctx = decode ? sh.kvLen : S;
  const CoremlWeight w = sh.weights;
  const int act = coremlActDtype(w);
  const int ioDtype = (act == CML_BF16) ? CML_FP16 : act;
  // Weights and activations in [-0.25, 0.25): the SwiGLU squares magnitudes
  // and the down projection sums thousands of terms, which overflowed fp16
  // at the [-0.5, 0.5) the GEMM rows use.
  const float mag = 0.5f;

  // The activations are a constant scaled by a runtime scalar, and the result
  // leaves as one reduced row (see onnxBlockModel for why).
  p.input("s", ioDtype, {1});
  p.constTensor("X0", act, {S, d}, coremlFillFloats(act, S * d, 0xa5a5a5a5u, mag));
  p.op("mul", {{"x", "X0"}, {"y", "s"}}, {"X", act, {S, d}});

  auto projection = [&](const std::string &out, const std::string &in, int64_t K, int64_t N,
                        uint32_t seed)
  {
    coremlEmitProjection(p, out, in, S, K, N, w, seed, mag);
  };

  // ---- QKV projection ----------------------------------------------------
  projection("Q", "X", d, d, 0x11111111u);
  projection("Knew", "X", d, d, 0x22222222u);
  projection("Vnew", "X", d, d, 0x33333333u);

  const std::string shHeads = p.constInts("sh_heads", {(int32_t)S, (int32_t)H, (int32_t)Dh});
  const std::string permHeads = p.constInts("perm_heads", {1, 0, 2});
  p.op("reshape", {{"x", "Q"}, {"shape", shHeads}}, {"Qr", act, {S, H, Dh}});
  p.op("transpose", {{"x", "Qr"}, {"perm", permHeads}}, {"Qh", act, {H, S, Dh}});

  // ---- Attention ---------------------------------------------------------
  // Decode reads a constant cache [H, ctx, Dh]; prefill builds K/V from this
  // pass.  The cache is stored in `act` unless the int8 row asks otherwise,
  // in which case it is dequantized on the way into attention -- a weight
  // decompression, which is how Core ML treats any constant.
  std::string Kh, Vh;
  if (decode)
  {
    if (sh.int8Kv)
    {
      // Quantized per tensor: the cache spends the whole int8 range and one
      // scale quarters it back to the [-0.25, 0.25) the fp16 cache holds.
      auto cache = [&](const std::string &name, uint32_t seed)
      {
        const int64_t rows = H * ctx;
        std::string packed((size_t)rows * Dh, '\0');
        for (int64_t i = 0; i < rows; i++)
          for (int64_t j = 0; j < Dh; j++)
          {
            int q = (int)std::lround(coremlWeightAt(i, j, seed) * 2.0f * 127.0f);
            q = std::max(-127, std::min(127, q));
            packed[(size_t)i * Dh + j] = (char)(int8_t)q;
          }
        const float scale = coremlHalfToFloat(coremlFloatToHalf(0.25f / 127.0f));
        p.affineDequantize({name, act, {H, ctx, Dh}}, CML_INT8, packed,
                           coremlFloatScalar(scale, act), 1, act, 0);
      };
      cache("Kc", 0x88888888u);
      cache("Vc", 0x99999999u);
    }
    else
    {
      p.constTensor("Kc", act, {H, ctx, Dh}, coremlFillFloats(act, H * ctx * Dh, 0x88888888u, mag));
      p.constTensor("Vc", act, {H, ctx, Dh}, coremlFillFloats(act, H * ctx * Dh, 0x99999999u, mag));
    }
    Kh = "Kc";
    Vh = "Vc";
  }
  else
  {
    p.op("reshape", {{"x", "Knew"}, {"shape", shHeads}}, {"Kr", act, {S, H, Dh}});
    p.op("transpose", {{"x", "Kr"}, {"perm", permHeads}}, {"Kh", act, {H, S, Dh}});
    p.op("reshape", {{"x", "Vnew"}, {"shape", shHeads}}, {"Vr", act, {S, H, Dh}});
    p.op("transpose", {{"x", "Vr"}, {"perm", permHeads}}, {"Vh", act, {H, S, Dh}});
    Kh = "Kh";
    Vh = "Vh";
  }

  if (sh.fusedAttention)
  {
    // Core ML's own attention op (iOS 18): softmax(Q K^T / sqrt(Dh)) V in
    // one node, the form a converted language model carries.
    p.op("scaled_dot_product_attention",
         {{"query", "Qh"}, {"key", Kh}, {"value", Vh}}, {"Ctx", act, {H, S, Dh}});
  }
  else
  {
    p.op("transpose", {{"x", Kh}, {"perm", p.constInts("perm_kt", {0, 2, 1})}},
         {"KT", act, {H, Dh, ctx}});
    p.op("matmul",
         {{"x", "Qh"}, {"y", "KT"}, {"transpose_x", p.constBool("s_tx", false)},
          {"transpose_y", p.constBool("s_ty", false)}},
         {"Scores", act, {H, S, ctx}});
    p.op("mul", {{"x", "Scores"}, {"y", p.constFloat("scale", act, 1.0f / std::sqrt((float)Dh))}},
         {"ScoresS", act, {H, S, ctx}});
    p.op("softmax", {{"x", "ScoresS"}, {"axis", p.constInt("sm_axis", -1)}}, {"P", act, {H, S, ctx}});
    p.op("matmul",
         {{"x", "P"}, {"y", Vh}, {"transpose_x", p.constBool("c_tx", false)},
          {"transpose_y", p.constBool("c_ty", false)}},
         {"Ctx", act, {H, S, Dh}});
  }

  p.op("transpose", {{"x", "Ctx"}, {"perm", permHeads}}, {"CtxT", act, {S, H, Dh}});
  p.op("reshape", {{"x", "CtxT"}, {"shape", p.constInts("sh_flat", {(int32_t)S, (int32_t)d})}},
       {"CtxF", act, {S, d}});
  projection("AttnOut", "CtxF", d, d, 0x44444444u);
  p.op("add", {{"x", "X"}, {"y", "AttnOut"}}, {"R1", act, {S, d}});

  // ---- SwiGLU feed-forward ----------------------------------------------
  projection("G", "R1", d, ffn, 0x55555555u);
  projection("U", "R1", d, ffn, 0x66666666u);
  p.op("silu", {{"x", "G"}}, {"Act", act, {S, ffn}});
  p.op("mul", {{"x", "Act"}, {"y", "U"}}, {"Hh", act, {S, ffn}});
  projection("Down", "Hh", ffn, d, 0x77777777u);
  p.op("add", {{"x", "R1"}, {"y", "Down"}}, {"Y", act, {S, d}});

  reduceMax(p, "Yr", "Y", {0}, false, act, {d});
  p.output("Yr", act, {d});
  if (decode)
  {
    // The cache write a real step performs; also what keeps the K/V
    // projections live.  One row each, so they cost nothing to return.
    p.output("Knew", act, {S, d});
    p.output("Vnew", act, {S, d});
  }
  return p;
}

#endif // ENABLE_COREML
