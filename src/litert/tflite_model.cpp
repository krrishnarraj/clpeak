#ifdef ENABLE_LITERT

#include "tflite_model.h"

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace clpeak_tflite
{

// ---------------------------------------------------------------------------
// A minimal FlatBuffer builder
// ---------------------------------------------------------------------------
// The reference builder, reduced to what a .tflite needs.  The buffer is
// filled from its end towards its start ("back to front"), so every object
// is written before the object that refers to it and a reference is always a
// forward (unsigned) distance.  Positions are therefore measured as offsets
// from the END of the buffer, which is the only coordinate that stays fixed
// while the front keeps growing.
//
// Layout rules honoured:
//   * scalars aligned to their size, tables to 4, vectors' data to
//     max(4, element alignment) with the u32 length immediately before;
//   * a table is [soffset to its vtable][fields...] and the vtable is
//     [u16 size][u16 table size][u16 field offset per slot], 0 = absent;
//   * a union is two fields: the type byte in slot n, the offset in n + 1;
//   * Finish() aligns the whole buffer to the largest alignment used and
//     prefixes [u32 root offset]["TFL3"].
namespace
{

class Builder
{
public:
  explicit Builder(size_t reserve)
  {
    // The buffer is grown by reallocating with the used tail copied to the
    // new end; reserving up front avoids doing that to a large model.
    size_t cap = reserve + 4096;
    cap = (cap + 15) & ~size_t(15);
    buf_.resize(cap);
    head_ = cap;
  }

  // Bytes written so far == an object's offset-from-end right after it is
  // written.
  uint32_t size() const { return (uint32_t)(buf_.size() - head_); }

  // ---- scalars ----------------------------------------------------------
  template <class T> uint32_t pushScalar(T v)
  {
    align(sizeof(T));
    uint8_t *p = push(sizeof(T));
    std::memcpy(p, &v, sizeof(T));
    return size();
  }

  // ---- strings ----------------------------------------------------------
  uint32_t createString(const std::string &s)
  {
    preAlign(s.size() + 1, sizeof(uint32_t));
    uint8_t *p = push(s.size() + 1);
    std::memcpy(p, s.data(), s.size());
    p[s.size()] = 0;
    pushScalar<uint32_t>((uint32_t)s.size());
    return size();
  }

  // ---- vectors ----------------------------------------------------------
  // Scalars are little-endian on every target clpeak builds for, so a
  // vector of them is one block copy.
  template <class T> uint32_t createVector(const T *data, size_t n, size_t forceAlign = 0)
  {
    const size_t bytes = n * sizeof(T);
    startVector(bytes, sizeof(T) > forceAlign ? sizeof(T) : forceAlign);
    uint8_t *p = push(bytes);
    if (bytes)
      std::memcpy(p, data, bytes);
    return endVector((uint32_t)n);
  }
  template <class T> uint32_t createVector(const std::vector<T> &v)
  {
    return createVector(v.data(), v.size());
  }
  // A byte vector whose contents the caller writes in place.
  uint32_t createFilledBytes(size_t bytes, void (*fill)(uint8_t *, size_t, void *), void *ctx,
                             size_t forceAlign)
  {
    startVector(bytes, forceAlign > 1 ? forceAlign : 1);
    uint8_t *p = push(bytes);
    if (bytes)
      fill(p, bytes, ctx);
    return endVector((uint32_t)bytes);
  }
  uint32_t createOffsetVector(const std::vector<uint32_t> &offs)
  {
    startVector(offs.size() * 4, 4);
    // Back to front: the last element is written first.
    for (size_t i = offs.size(); i-- > 0;)
    {
      uint32_t rel = referTo(offs[i]);
      uint8_t *p = push(4);
      std::memcpy(p, &rel, 4);
    }
    return endVector((uint32_t)offs.size());
  }

  // ---- tables -----------------------------------------------------------
  void startTable()
  {
    fields_.clear();
    tableStart_ = size();
  }
  template <class T> void addScalar(int slot, T v, T deflt)
  {
    if (v == deflt)
      return;
    pushScalar<T>(v);
    noteField(slot);
  }
  template <class T> void addScalarAlways(int slot, T v)
  {
    pushScalar<T>(v);
    noteField(slot);
  }
  void addOffset(int slot, uint32_t off)
  {
    if (off == 0)
      return;
    pushScalar<uint32_t>(referTo(off));
    noteField(slot);
  }
  uint32_t endTable()
  {
    // The table's first word is the soffset to its vtable; write a
    // placeholder now and patch it once the vtable's position is known.
    const uint32_t tableOff = pushScalar<int32_t>(0);
    const uint32_t tableSize = tableOff - tableStart_;

    int maxSlot = -1;
    for (const auto &f : fields_)
      if (f.slot > maxSlot)
        maxSlot = f.slot;
    const int numSlots = maxSlot + 1;

    std::vector<uint16_t> vt((size_t)numSlots + 2, 0);
    vt[0] = (uint16_t)((numSlots + 2) * 2);
    vt[1] = (uint16_t)tableSize;
    for (const auto &f : fields_)
      vt[(size_t)f.slot + 2] = (uint16_t)(tableOff - f.off);

    // u16 array, written last element first.
    align(2);
    for (size_t i = vt.size(); i-- > 0;)
    {
      uint8_t *p = push(2);
      std::memcpy(p, &vt[i], 2);
    }
    const uint32_t vtOff = size();

    // soffset stored at the table start: table position minus vtable
    // position; both are offsets from the end, so the difference flips.
    const int32_t so = (int32_t)vtOff - (int32_t)tableOff;
    uint8_t *tp = buf_.data() + buf_.size() - tableOff;
    std::memcpy(tp, &so, 4);
    return tableOff;
  }

  // ---- finish -----------------------------------------------------------
  TfliteBytes finish(uint32_t root, const char ident[4])
  {
    preAlign(4 + 4, minAlign_);
    uint8_t *p = push(4);
    std::memcpy(p, ident, 4);
    pushScalar<uint32_t>(referTo(root));
    TfliteBytes out;
    out.head = head_;
    out.storage = std::move(buf_);
    return out;
  }

private:
  struct Field
  {
    int slot;
    uint32_t off;
  };

  std::vector<uint8_t> buf_;
  size_t head_ = 0;
  size_t minAlign_ = 1;
  std::vector<Field> fields_;
  uint32_t tableStart_ = 0;

  void noteField(int slot) { fields_.push_back({slot, size()}); }

  // Distance from a reference written right now to an object at `off`.
  uint32_t referTo(uint32_t off)
  {
    align(4);
    return size() - off + 4;
  }

  void ensure(size_t bytes)
  {
    if (head_ >= bytes)
      return;
    size_t cap = buf_.size();
    while (cap - (buf_.size() - head_) < bytes + 16)
      cap *= 2;
    cap = (cap + 15) & ~size_t(15);
    std::vector<uint8_t> nb(cap);
    const size_t used = buf_.size() - head_;
    std::memcpy(nb.data() + cap - used, buf_.data() + head_, used);
    head_ = cap - used;
    buf_.swap(nb);
  }
  uint8_t *push(size_t bytes)
  {
    ensure(bytes);
    head_ -= bytes;
    return buf_.data() + head_;
  }
  void pad(size_t n)
  {
    if (n == 0)
      return;
    uint8_t *p = push(n);
    std::memset(p, 0, n);
  }
  // Pad so that the next `sizeof` object written lands aligned.
  void align(size_t a)
  {
    if (a > minAlign_)
      minAlign_ = a;
    pad((a - (size() & (a - 1))) & (a - 1));
  }
  // Pad so that after `len` more bytes the position is `a`-aligned.
  void preAlign(size_t len, size_t a)
  {
    if (a > minAlign_)
      minAlign_ = a;
    pad((a - ((size() + len) & (a - 1))) & (a - 1));
  }
  void startVector(size_t bytes, size_t elemAlign)
  {
    preAlign(bytes, 4);
    preAlign(bytes, elemAlign);
  }
  uint32_t endVector(uint32_t count) { return pushScalar<uint32_t>(count); }
};

// schema.fbs `union BuiltinOptions`: the member's 1-based position.
enum class BuiltinOptionsType : uint8_t
{
  None = 0,
  Conv2DOptions = 1,
  DepthwiseConv2DOptions = 2,
  FullyConnectedOptions = 8,
  SoftmaxOptions = 9,
  ConcatenationOptions = 10,
  AddOptions = 11,
  ReshapeOptions = 17,
  MulOptions = 21,
  ReducerOptions = 27,
  SubOptions = 28,
  CastOptions = 37,
  BatchMatMulOptions = 101,
  GeluOptions = 116,
};

// schema.fbs `union BuiltinOptions2` (position 21: after nineteen
// stablehlo.* option tables and the deprecated ReduceWindowOptions).
enum class BuiltinOptions2Type : uint8_t
{
  None = 0,
  StableHLOCompositeOptions = 21,
};

// Writes one options table; returns its offset and which union it belongs
// to.  Field ids are the schema's declaration order within each table.
struct OptionsRef
{
  BuiltinOptionsType type = BuiltinOptionsType::None;
  BuiltinOptions2Type type2 = BuiltinOptions2Type::None;
  uint32_t off = 0;
};

OptionsRef writeOptions(Builder &b, const TfOptions &o)
{
  OptionsRef r;
  using K = TfOptions::Kind;
  switch (o.kind)
  {
  case K::None:
    return r;

  case K::FullyConnected:
    // FullyConnectedOptions: fused_activation_function(0), weights_format(1),
    // keep_num_dims(2), asymmetric_quantize_inputs(3), quantized_bias_type(4)
    b.startTable();
    b.addScalar<int8_t>(0, (int8_t)o.act, 0);
    b.addScalar<uint8_t>(2, o.keepNumDims ? 1 : 0, 0);
    b.addScalar<uint8_t>(3, o.asymmetricQuantizeInputs ? 1 : 0, 0);
    b.addScalar<int8_t>(4, (int8_t)o.quantizedBiasType, 0);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::FullyConnectedOptions;
    return r;

  case K::BatchMatMul:
    // BatchMatMulOptions: adj_x(0), adj_y(1), asymmetric_quantize_inputs(2)
    b.startTable();
    b.addScalar<uint8_t>(0, o.adjX ? 1 : 0, 0);
    b.addScalar<uint8_t>(1, o.adjY ? 1 : 0, 0);
    b.addScalar<uint8_t>(2, o.asymmetricQuantizeInputs ? 1 : 0, 0);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::BatchMatMulOptions;
    return r;

  case K::Conv2d:
    // Conv2DOptions: padding(0), stride_w(1), stride_h(2),
    // fused_activation_function(3), dilation_w_factor(4)=1,
    // dilation_h_factor(5)=1, quantized_bias_type(6)
    b.startTable();
    b.addScalar<int8_t>(0, (int8_t)o.padding, 0);
    b.addScalar<int32_t>(1, o.strideW, 0);
    b.addScalar<int32_t>(2, o.strideH, 0);
    b.addScalar<int8_t>(3, (int8_t)o.act, 0);
    b.addScalar<int32_t>(4, o.dilationW, 1);
    b.addScalar<int32_t>(5, o.dilationH, 1);
    b.addScalar<int8_t>(6, (int8_t)o.quantizedBiasType, 0);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::Conv2DOptions;
    return r;

  case K::DepthwiseConv2d:
    // DepthwiseConv2DOptions: padding(0), stride_w(1), stride_h(2),
    // depth_multiplier(3), fused_activation_function(4),
    // dilation_w_factor(5)=1, dilation_h_factor(6)=1
    b.startTable();
    b.addScalar<int8_t>(0, (int8_t)o.padding, 0);
    b.addScalar<int32_t>(1, o.strideW, 0);
    b.addScalar<int32_t>(2, o.strideH, 0);
    b.addScalar<int32_t>(3, o.depthMultiplier, 0);
    b.addScalar<int8_t>(4, (int8_t)o.act, 0);
    b.addScalar<int32_t>(5, o.dilationW, 1);
    b.addScalar<int32_t>(6, o.dilationH, 1);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::DepthwiseConv2DOptions;
    return r;

  case K::Softmax:
    // SoftmaxOptions: beta(0)
    b.startTable();
    b.addScalar<float>(0, o.beta, 0.0f);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::SoftmaxOptions;
    return r;

  case K::Mul:
    // MulOptions: fused_activation_function(0)
    b.startTable();
    b.addScalar<int8_t>(0, (int8_t)o.act, 0);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::MulOptions;
    return r;

  case K::Add:
    // AddOptions: fused_activation_function(0), pot_scale_int16(1)=true
    b.startTable();
    b.addScalar<int8_t>(0, (int8_t)o.act, 0);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::AddOptions;
    return r;

  case K::Sub:
    // SubOptions: fused_activation_function(0), pot_scale_int16(1)=true
    b.startTable();
    b.addScalar<int8_t>(0, (int8_t)o.act, 0);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::SubOptions;
    return r;

  case K::Reducer:
    // ReducerOptions: keep_dims(0)
    b.startTable();
    b.addScalar<uint8_t>(0, o.keepDims ? 1 : 0, 0);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::ReducerOptions;
    return r;

  case K::Reshape:
  {
    // ReshapeOptions: new_shape(0)
    const uint32_t shape = b.createVector(o.newShape);
    b.startTable();
    b.addOffset(0, shape);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::ReshapeOptions;
    return r;
  }

  case K::Cast:
    // CastOptions: in_data_type(0), out_data_type(1)
    b.startTable();
    b.addScalarAlways<int8_t>(0, (int8_t)o.castIn);
    b.addScalarAlways<int8_t>(1, (int8_t)o.castOut);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::CastOptions;
    return r;

  case K::Gelu:
    // GeluOptions: approximate(0)
    b.startTable();
    b.addScalar<uint8_t>(0, o.approximate ? 1 : 0, 0);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::GeluOptions;
    return r;

  case K::Concatenation:
    // ConcatenationOptions: axis(0), fused_activation_function(1)
    b.startTable();
    b.addScalar<int32_t>(0, o.axis, 0);
    b.addScalar<int8_t>(1, (int8_t)o.act, 0);
    r.off = b.endTable();
    r.type = BuiltinOptionsType::ConcatenationOptions;
    return r;

  case K::Composite:
  {
    // StableHLOCompositeOptions: name(0), decomposition_subgraph_index(1),
    // composite_attributes(2), composite_attributes_format(3) (FLEXBUFFERS =
    // 0, the default), version(4)
    const uint32_t name = b.createString(o.compositeName);
    const uint32_t attrs = o.compositeAttributes.empty() ? 0 : b.createVector(o.compositeAttributes);
    b.startTable();
    b.addOffset(0, name);
    b.addScalar<int32_t>(1, o.decompositionSubgraph, 0);
    if (attrs)
      b.addOffset(2, attrs);
    b.addScalar<int32_t>(4, o.compositeVersion, 0);
    r.off = b.endTable();
    r.type2 = BuiltinOptions2Type::StableHLOCompositeOptions;
    return r;
  }
  }
  return r;
}

} // namespace

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

size_t tfliteElementBits(TfType t)
{
  switch (t)
  {
  case TfType::F32: case TfType::I32: return 32;
  case TfType::F16: case TfType::I16: case TfType::U16: case TfType::BF16: return 16;
  case TfType::U8: case TfType::I8: case TfType::Bool: case TfType::F8E4M3: case TfType::F8E5M2: return 8;
  case TfType::I64: return 64;
  case TfType::I4: case TfType::U4: return 4;
  case TfType::I2: return 2;
  }
  return 8;
}

size_t tflitePackedBytes(TfType t, size_t count)
{
  const size_t bits = tfliteElementBits(t);
  return (count * bits + 7) / 8;
}

const char *tfliteTypeName(TfType t)
{
  switch (t)
  {
  case TfType::F32: return "float32";
  case TfType::F16: return "float16";
  case TfType::I32: return "int32";
  case TfType::U8: return "uint8";
  case TfType::I64: return "int64";
  case TfType::Bool: return "bool";
  case TfType::I16: return "int16";
  case TfType::I8: return "int8";
  case TfType::U16: return "uint16";
  case TfType::I4: return "int4";
  case TfType::BF16: return "bfloat16";
  case TfType::I2: return "int2";
  case TfType::U4: return "uint4";
  case TfType::F8E4M3: return "float8_e4m3fn";
  case TfType::F8E5M2: return "float8_e5m2";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// TfliteModel
// ---------------------------------------------------------------------------

TfliteModel::TfliteModel()
{
  subgraphs_.push_back(Subgraph{"main", {}, {}, {}, {}});
  buffers_.push_back(Buffer{nullptr, 0, nullptr, nullptr});   // buffer 0: empty
}

void TfliteModel::reserveBytes(size_t bytes) { reserve_ = bytes; }

int TfliteModel::addBuffer(const void *data, size_t bytes)
{
  buffers_.push_back(Buffer{static_cast<const uint8_t *>(data), bytes, nullptr, nullptr});
  return (int)buffers_.size() - 1;
}

int TfliteModel::addBufferFill(size_t bytes, void (*fill)(uint8_t *, size_t, void *), void *ctx)
{
  buffers_.push_back(Buffer{nullptr, bytes, fill, ctx});
  return (int)buffers_.size() - 1;
}

int TfliteModel::addTensor(int sg, const std::vector<int32_t> &shape, TfType type,
                           const std::string &name, int buffer, const TfQuant &q)
{
  Subgraph &s = subgraphs_.at((size_t)sg);
  s.tensors.push_back(Tensor{shape, type, buffer, name, q});
  return (int)s.tensors.size() - 1;
}

void TfliteModel::addOp(int sg, TfOp op, int version, const std::vector<int32_t> &inputs,
                        const std::vector<int32_t> &outputs, const TfOptions &opts)
{
  subgraphs_.at((size_t)sg).ops.push_back(Op{op, version, inputs, outputs, opts});
}

void TfliteModel::setInputs(int sg, const std::vector<int32_t> &t) { subgraphs_.at((size_t)sg).inputs = t; }
void TfliteModel::setOutputs(int sg, const std::vector<int32_t> &t) { subgraphs_.at((size_t)sg).outputs = t; }

int TfliteModel::addSubgraph(const std::string &name)
{
  subgraphs_.push_back(Subgraph{name, {}, {}, {}, {}});
  return (int)subgraphs_.size() - 1;
}

TfliteBytes TfliteModel::build(const std::string &description) const
{
  const auto t0 = std::chrono::steady_clock::now();
  size_t weightBytes = reserve_;
  for (const auto &bf : buffers_)
    weightBytes += bf.bytes + 32;
  Builder b(weightBytes + 64 * 1024);

  // ---- buffers ------------------------------------------------------------
  // Buffer: data(0) [ubyte] force_align 16, offset(1), size(2).  Written
  // first: they are the bulk of the model and the weights land aligned.
  std::vector<uint32_t> bufferOffs;
  bufferOffs.reserve(buffers_.size());
  for (const auto &bf : buffers_)
  {
    uint32_t data = 0;
    if (bf.fill)
      data = b.createFilledBytes(bf.bytes, bf.fill, bf.ctx, 16);
    else if (bf.bytes)
      data = b.createVector<uint8_t>(bf.data, bf.bytes, 16);
    b.startTable();
    b.addOffset(0, data);
    bufferOffs.push_back(b.endTable());
  }
  const uint32_t buffersVec = b.createOffsetVector(bufferOffs);

  // ---- operator codes -----------------------------------------------------
  // One entry per distinct (op, version), in first-use order.
  struct Code
  {
    TfOp op;
    int version;
  };
  std::vector<Code> codes;
  auto codeIndex = [&](TfOp op, int version) -> uint32_t {
    for (size_t i = 0; i < codes.size(); i++)
      if (codes[i].op == op && codes[i].version == version)
        return (uint32_t)i;
    codes.push_back({op, version});
    return (uint32_t)codes.size() - 1;
  };
  for (const auto &sg : subgraphs_)
    for (const auto &op : sg.ops)
      codeIndex(op.op, op.version);

  std::vector<uint32_t> codeOffs;
  for (const auto &c : codes)
  {
    // OperatorCode: deprecated_builtin_code(0) byte, custom_code(1),
    // version(2) = 1, builtin_code(3).  Readers take the larger of the two
    // code fields, so codes past 127 put the placeholder in the byte one.
    const int32_t code = (int32_t)c.op;
    b.startTable();
    b.addScalarAlways<int8_t>(0, (int8_t)(code > 127 ? 127 : code));
    b.addScalar<int32_t>(2, c.version, 1);
    b.addScalar<int32_t>(3, code, 0);
    codeOffs.push_back(b.endTable());
  }
  const uint32_t codesVec = b.createOffsetVector(codeOffs);

  // ---- subgraphs ----------------------------------------------------------
  std::vector<uint32_t> sgOffs;
  for (const auto &sg : subgraphs_)
  {
    std::vector<uint32_t> tensorOffs;
    for (const auto &t : sg.tensors)
    {
      const uint32_t shape = b.createVector(t.shape);
      const uint32_t name = b.createString(t.name);
      uint32_t quant = 0;
      if (!t.quant.empty())
      {
        // QuantizationParameters: min(0), max(1), scale(2), zero_point(3),
        // details_type(4), details(5), quantized_dimension(6)
        uint32_t scale = 0, zp = 0, details = 0;
        if (t.quant.blockSize > 0)
        {
          // BlockwiseQuantization: scales(0), zero_points(1), block_size(2);
          // union member 2 of QuantizationDetails.
          b.startTable();
          b.addScalarAlways<int32_t>(0, t.quant.scalesTensor);
          b.addScalar<int32_t>(1, t.quant.zeroPointsTensor, 0);
          b.addScalarAlways<int32_t>(2, t.quant.blockSize);
          details = b.endTable();
        }
        else
        {
          scale = b.createVector(t.quant.scale);
          zp = b.createVector(t.quant.zeroPoint);
        }
        b.startTable();
        b.addOffset(2, scale);
        b.addOffset(3, zp);
        if (details)
        {
          b.addScalarAlways<uint8_t>(4, 2);
          b.addOffset(5, details);
        }
        b.addScalar<int32_t>(6, t.quant.quantizedDim, 0);
        quant = b.endTable();
      }
      // Tensor: shape(0), type(1), buffer(2), name(3), quantization(4),
      // is_variable(5), sparsity(6), shape_signature(7), has_rank(8)
      b.startTable();
      b.addOffset(0, shape);
      b.addScalar<int8_t>(1, (int8_t)t.type, 0);
      b.addScalar<uint32_t>(2, (uint32_t)t.buffer, 0);
      b.addOffset(3, name);
      b.addOffset(4, quant);
      b.addScalarAlways<uint8_t>(8, 1);   // has_rank: every shape here is static
      tensorOffs.push_back(b.endTable());
    }
    const uint32_t tensorsVec = b.createOffsetVector(tensorOffs);

    std::vector<uint32_t> opOffs;
    for (const auto &op : sg.ops)
    {
      const uint32_t inputs = b.createVector(op.inputs);
      const uint32_t outputs = b.createVector(op.outputs);
      const OptionsRef opt = writeOptions(b, op.opts);
      // Operator: opcode_index(0), inputs(1), outputs(2),
      // builtin_options_type(3), builtin_options(4), custom_options(5),
      // custom_options_format(6), mutating_variable_inputs(7),
      // intermediates(8), large_custom_options_offset(9),
      // large_custom_options_size(10), builtin_options_2_type(11),
      // builtin_options_2(12), debug_metadata_index(13)
      b.startTable();
      b.addScalar<uint32_t>(0, codeIndex(op.op, op.version), 0);
      b.addOffset(1, inputs);
      b.addOffset(2, outputs);
      if (opt.type != BuiltinOptionsType::None)
      {
        b.addScalarAlways<uint8_t>(3, (uint8_t)opt.type);
        b.addOffset(4, opt.off);
      }
      if (opt.type2 != BuiltinOptions2Type::None)
      {
        b.addScalarAlways<uint8_t>(11, (uint8_t)opt.type2);
        b.addOffset(12, opt.off);
      }
      opOffs.push_back(b.endTable());
    }
    const uint32_t opsVec = b.createOffsetVector(opOffs);
    const uint32_t inputsVec = b.createVector(sg.inputs);
    const uint32_t outputsVec = b.createVector(sg.outputs);
    const uint32_t name = b.createString(sg.name);

    // SubGraph: tensors(0), inputs(1), outputs(2), operators(3), name(4)
    b.startTable();
    b.addOffset(0, tensorsVec);
    b.addOffset(1, inputsVec);
    b.addOffset(2, outputsVec);
    b.addOffset(3, opsVec);
    b.addOffset(4, name);
    sgOffs.push_back(b.endTable());
  }
  const uint32_t subgraphsVec = b.createOffsetVector(sgOffs);

  // ---- signature --------------------------------------------------------
  // One signature over subgraph 0 naming its inputs and outputs by tensor
  // name, as a converted model carries; a runtime that indexes by signature
  // then finds one.  TensorMap: name(0), tensor_index(1).  SignatureDef:
  // inputs(0), outputs(1), signature_key(2), deprecated_tag(3),
  // subgraph_index(4).
  uint32_t signaturesVec = 0;
  {
    const Subgraph &main = subgraphs_[0];
    auto maps = [&](const std::vector<int32_t> &idx) {
      std::vector<uint32_t> offs;
      for (int32_t ti : idx)
      {
        const uint32_t nm = b.createString(main.tensors.at((size_t)ti).name);
        b.startTable();
        b.addOffset(0, nm);
        b.addScalar<uint32_t>(1, (uint32_t)ti, 0);
        offs.push_back(b.endTable());
      }
      return b.createOffsetVector(offs);
    };
    const uint32_t ins = maps(main.inputs);
    const uint32_t outs = maps(main.outputs);
    const uint32_t key = b.createString("serving_default");
    b.startTable();
    b.addOffset(0, ins);
    b.addOffset(1, outs);
    b.addOffset(2, key);
    const uint32_t sig = b.endTable();
    signaturesVec = b.createOffsetVector({sig});
  }

  // ---- model --------------------------------------------------------------
  const uint32_t desc = b.createString(description);
  // Model: version(0), operator_codes(1), subgraphs(2), description(3),
  // buffers(4), metadata_buffer(5), metadata(6), signature_defs(7)
  b.startTable();
  b.addScalarAlways<uint32_t>(0, 3);
  b.addOffset(1, codesVec);
  b.addOffset(2, subgraphsVec);
  b.addOffset(3, desc);
  b.addOffset(4, buffersVec);
  b.addOffset(7, signaturesVec);
  const uint32_t root = b.endTable();

  TfliteBytes out = b.finish(root, "TFL3");
  out.description = description;
  out.buildUs = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  return out;
}

} // namespace clpeak_tflite

#endif // ENABLE_LITERT
