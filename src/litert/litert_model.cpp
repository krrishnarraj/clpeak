#ifdef ENABLE_LITERT

#include "litert_model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <thread>

using namespace clpeak_tflite;

// AArch64 converts fp32 <-> fp16 in one instruction (fcvt: round to nearest
// even, subnormals kept), which the fills lean on: a phone rounds a
// 64M-element constant to half in the time the bit-level routine below
// takes for a fraction of it.  The two agree to the bit -- the routine is
// the same rounding spelt out -- and the bit-level one stays as the other
// platforms' path and the definition.
#if defined(__aarch64__) && defined(__ARM_FP16_FORMAT_IEEE)
#define CLPEAK_LITERT_HW_HALF 1
#else
#define CLPEAK_LITERT_HW_HALF 0
#endif

// ---------------------------------------------------------------------------
// Formats and plans
// ---------------------------------------------------------------------------

const char *litertFormatLabel(LitertFormat f)
{
  switch (f)
  {
  case LitertFormat::Fp32: return "fp32";
  case LitertFormat::Fp16: return "fp16";
  case LitertFormat::Fp16Acc32: return "fp16_acc32";
  case LitertFormat::Bf16: return "bf16";
  case LitertFormat::Int8Qdq: return "int8_qdq";
  case LitertFormat::Int16x8: return "int16x8";
  case LitertFormat::Int8Weight: return "int8_weight";
  case LitertFormat::Int4Weight: return "int4_weight";
  case LitertFormat::Fp8Weight: return "fp8_weight";
  }
  return "?";
}

LitertPlan litertPlanFor(LitertFormat f, LitertAccel accel)
{
  LitertPlan p;
  // The GPU accelerator's graph reader accepts fp32, fp16, int8, uint8,
  // int4, int2, bool and int32 tensors and CHECK-fails on anything else --
  // an abort, not a refusal (object_reader.cc, LiteRT 2.2.0: "Tensor
  // type(INT16) is not supported" followed by the process dying).  A
  // benchmark that takes the whole run down with it to prove that would be
  // proving the wrong thing, so those formats never reach the GPU.
  auto gpuAborts = [&](const char *type) {
    p.applies = false;
    p.whyNot = std::string("the GPU accelerator's graph reader aborts the process on ") + type +
               " tensors rather than declining them (LiteRT 2.2.0), so this format is not sent to it";
  };
  switch (f)
  {
  case LitertFormat::Fp32:
    p.gpuPrecision = kLiteRtDelegatePrecisionFp32;
    break;
  case LitertFormat::Fp16:
    // The GPU accelerator refuses fp16-typed tensors outright ("input->type
    // != kTfLiteFloat32" from the kernel it falls back to), and its own fp16
    // is a policy over an fp32 graph: weights converted to half at load,
    // half arithmetic throughout.  That is fp16 storage and arithmetic as
    // surely as a half-typed graph is, so it is what the row runs there.
    if (accel == LitertAccel::Gpu)
    {
      p.gpuPrecision = kLiteRtDelegatePrecisionFp16;
      p.halfRounded = true;
      p.halfConstants = true;
      p.weight = TfType::F16;
    }
    else
      p.act = p.weight = TfType::F16;
    break;
  case LitertFormat::Fp16Acc32:
    if (accel != LitertAccel::Gpu)
    {
      p.applies = false;
      p.whyNot = "a GPU precision policy (fp16 storage with fp32 accumulation); the " +
                 std::string(litertAccelName(accel)) + " has no such setting";
    }
    p.gpuPrecision = kLiteRtDelegatePrecisionFp16WithFp32Accum;
    p.halfRounded = true;
    p.halfConstants = true;
    p.weight = TfType::F16;
    break;
  case LitertFormat::Bf16:
    if (accel == LitertAccel::Gpu)
      gpuAborts("bfloat16");
    p.act = p.weight = TfType::BF16;
    break;
  case LitertFormat::Int8Qdq:
    p.act = TfType::I8;
    p.weight = TfType::I8;
    p.perChannel = true;
    p.integerOps = true;
    p.gpuAllowQuantized = true;
    break;
  case LitertFormat::Int16x8:
    if (accel == LitertAccel::Gpu)
      gpuAborts("int16");
    p.act = TfType::I16;
    p.weight = TfType::I8;
    p.perChannel = true;
    p.integerOps = true;
    p.gpuAllowQuantized = true;
    break;
  case LitertFormat::Int8Weight:
    p.weight = TfType::I8;
    p.perChannel = true;
    p.dynamicQuant = true;
    break;
  case LitertFormat::Int4Weight:
    // Blockwise int4 is the one format the GPU accelerator gets memory-
    // unsafely wrong: its weight conversion transposes the packed nibbles
    // with an 8-bit 16x16 transpose microkernel (xnn_x8_transposec_ukernel)
    // that runs past the buffer, which guard malloc catches on the first
    // model and a normal run pays for as a heap corruption some rungs later
    // (4096-cubed on an M1 Pro).  Per-row int4 goes through a different
    // path and is safe, but it is not the format language models ship in,
    // and a row that measures a different format under the same label
    // would mislead more than a refusal does.
    if (accel == LitertAccel::Gpu)
    {
      p.applies = false;
      p.whyNot = "the GPU accelerator's blockwise int4 weight conversion overruns its buffer "
                 "(an 8-bit transpose microkernel reads and writes past the packed nibbles, "
                 "LiteRT 2.2.0) and corrupts the heap, so this format is not sent to it";
    }
    p.weight = TfType::I4;
    p.weightBlock = 32;
    p.dynamicQuant = true;
    break;
  case LitertFormat::Fp8Weight:
    if (accel == LitertAccel::Gpu)
      gpuAborts("float8");
    p.weight = TfType::F8E4M3;
    p.perChannel = true;
    p.dynamicQuant = true;
    break;
  }
  return p;
}

// tflite/converter/tools/versioning/op_version.cc, FULLY_CONNECTED, in the
// rules' order: the float and dynamically quantized graphs carry no bias
// (two inputs, version 6); the full-integer ones carry the int32 bias a
// converted model always has, with keep_num_dims, which is version 5.
int litertFcVersion(const LitertPlan &p, bool hasBias)
{
  if (p.weight == TfType::I2)
    return 14;
  if (p.act == TfType::I16 && p.weight == TfType::I4)
    return 13;
  if (p.act == TfType::F32 && p.weight == TfType::I8 && p.perChannel)
    return 12;
  if (p.act == TfType::I16 && p.weight == TfType::I16)
    return 7;
  if (!hasBias)
    return 6;   // two inputs (an fp16 weight dequantized into an fp32 graph included)
  return 5;   // keep_num_dims
}

// ---------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------

namespace
{

uint16_t softFloatToHalf(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if (((x >> 23) & 0xffu) == 0xffu)   // inf / nan
    return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
  if (exp >= 31)
    return (uint16_t)(sign | 0x7c00u);
  if (exp <= 0)
  {
    if (exp < -10)
      return (uint16_t)sign;
    mant |= 0x800000u;
    const uint32_t shift = (uint32_t)(14 - exp);
    uint32_t half = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1);
    const uint32_t halfway = 1u << (shift - 1);
    if (rem > halfway || (rem == halfway && (half & 1u)))
      half++;
    return (uint16_t)(sign | half);
  }
  uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
    half++;   // carries into the exponent correctly
  return (uint16_t)half;
}

float softHalfToFloat(uint16_t h)
{
  const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t out;
  if (exp == 0)
  {
    if (mant == 0)
      out = sign;
    else
    {
      // subnormal: normalise
      int e = -1;
      do
      {
        e++;
        mant <<= 1;
      } while ((mant & 0x400u) == 0);
      mant &= 0x3ffu;
      out = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
    }
  }
  else if (exp == 31)
    out = sign | 0x7f800000u | (mant << 13);
  else
    out = sign | ((exp + 112u) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

} // namespace

uint16_t litertFloatToHalf(float f)
{
#if CLPEAK_LITERT_HW_HALF
  const __fp16 h = (__fp16)f;
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
#else
  return softFloatToHalf(f);
#endif
}

float litertHalfToFloat(uint16_t h)
{
#if CLPEAK_LITERT_HW_HALF
  __fp16 v;
  std::memcpy(&v, &h, 2);
  return (float)v;
#else
  return softHalfToFloat(h);
#endif
}

bool litertHalfConversionsAgree()
{
#if CLPEAK_LITERT_HW_HALF
  // Every half-precision code back through both paths, and every float
  // with a half-precision rounding case: the subnormal band, the halfway
  // points, the overflow edge.
  for (uint32_t h = 0; h < 0x10000u; h++)
  {
    const float a = litertHalfToFloat((uint16_t)h), b = softHalfToFloat((uint16_t)h);
    if (std::memcmp(&a, &b, 4) != 0 && !(std::isnan(a) && std::isnan(b)))
      return false;
  }
  for (uint32_t i = 0; i < (1u << 22); i++)
  {
    // A float per 2^10 codes across the whole range, plus the halfway points.
    const uint32_t x = i << 10;
    for (uint32_t y : {x, x + 0x1000u, x + 0x1fffu, x + 0x2000u})
    {
      float f;
      std::memcpy(&f, &y, 4);
      if (std::isnan(f))
        continue;
      if (litertFloatToHalf(f) != softFloatToHalf(f))
        return false;
    }
  }
#endif
  return true;
}

uint16_t litertFloatToBf16(float f)
{
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if (((x >> 23) & 0xffu) == 0xffu)
    return (uint16_t)((x >> 16) | ((x & 0x7fffffu) ? 0x40u : 0u));
  const uint32_t lsb = (x >> 16) & 1u;
  x += 0x7fffu + lsb;   // round to nearest even
  return (uint16_t)(x >> 16);
}

float litertBf16ToFloat(uint16_t b)
{
  const uint32_t x = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

// E4M3FN: bias 7, three mantissa bits, largest finite 448, no infinity, the
// all-ones mantissa at the top exponent is NaN.
uint8_t litertFloatToFp8E4M3(float f)
{
  if (std::isnan(f))
    return 0x7f;
  const uint8_t sign = f < 0.0f ? 0x80 : 0x00;
  float a = std::fabs(f);
  if (a > 448.0f)
    a = 448.0f;
  if (a < 0.0009765625f)   // below half the smallest subnormal (2^-9): zero
    return sign;
  int e;
  const float m = std::frexp(a, &e);   // a = m * 2^e, m in [0.5, 1)
  int exp = e - 1 + 7;                  // biased exponent of the leading one
  float mant;
  if (exp <= 0)
  {
    // subnormal: value = mant * 2^-9 with 3 bits
    mant = a / 0.001953125f;            // 2^-9
    int q = (int)std::lround(mant);
    if (q >= 8)
      return (uint8_t)(sign | 0x08);    // rounds up into the first normal
    return (uint8_t)(sign | q);
  }
  mant = (m * 2.0f - 1.0f) * 8.0f;      // 0..8
  int q = (int)std::lround(mant);
  if (q == 8)
  {
    q = 0;
    exp++;
  }
  if (exp > 15 || (exp == 15 && q == 7))
    return (uint8_t)(sign | 0x7e);      // 448
  return (uint8_t)(sign | (exp << 3) | q);
}

float litertFp8E4M3ToFloat(uint8_t v)
{
  const float sign = (v & 0x80) ? -1.0f : 1.0f;
  const int exp = (v >> 3) & 0xf;
  const int mant = v & 0x7;
  if (exp == 15 && mant == 7)
    return NAN;
  if (exp == 0)
    return sign * (float)mant * 0.001953125f;   // 2^-9
  return sign * std::ldexp(1.0f + (float)mant / 8.0f, exp - 7);
}

// ---------------------------------------------------------------------------
// Operand values and scales
// ---------------------------------------------------------------------------

namespace
{

uint32_t hash32(uint32_t h)
{
  h ^= h >> 16;
  h *= 0x7feb352du;
  h ^= h >> 15;
  h *= 0x846ca68bu;
  h ^= h >> 16;
  return h;
}

// Row i's key, then one mix per element.  The inner loop over j is integer
// multiply/shift/xor and a float convert with nothing carried between
// elements, so it vectorises, and rows are independent, so a fill runs on
// every core (fillTensor below).
uint32_t rowKey(int64_t i, uint32_t seed) { return hash32((uint32_t)i * 2654435761u ^ seed); }

float valueIn(uint32_t key, int64_t j)
{
  const uint32_t h = hash32(key ^ ((uint32_t)j * 2246822519u));
  return (float)(h & 0xffffffu) * (1.0f / 16777216.0f) - 0.5f;
}

} // namespace

float litertValueAt(int64_t i, int64_t j, uint32_t seed) { return valueIn(rowKey(i, seed), j); }

float litertActScale(int bits)
{
  // Uniform in [-0.5, 0.5): the edge onto the widest code.
  return 0.5f / (float)((1 << (bits - 1)) - 1);
}

float litertWeightScale(int bits)
{
  return 0.5f / (float)((1 << (bits - 1)) - 1);
}

float litertOutScale(int64_t K, int bits)
{
  // Four sigma of a K-deep dot product of two such operands (variance 1/12
  // each) onto the widest code.
  return (float)(std::sqrt((double)K) / 3.0 / (double)((1 << (bits - 1)) - 1));
}

TfType litertConstantType(const LitertPlan &p)
{
  return (p.halfConstants && p.act == TfType::F32) ? TfType::F16 : p.act;
}

size_t litertElemBytes(TfType t, int64_t count)
{
  return tflitePackedBytes(t, (size_t)count);
}

std::string litertScalarBytes(TfType t, float v)
{
  std::string out;
  switch (t)
  {
  case TfType::F32:
    out.assign(reinterpret_cast<const char *>(&v), 4);
    break;
  case TfType::F16:
  {
    const uint16_t h = litertFloatToHalf(v);
    out.assign(reinterpret_cast<const char *>(&h), 2);
    break;
  }
  case TfType::BF16:
  {
    const uint16_t b = litertFloatToBf16(v);
    out.assign(reinterpret_cast<const char *>(&b), 2);
    break;
  }
  case TfType::I8:
  {
    // Quantized with the scalar's own scale 1/127: 1.0 is code 127.
    const int8_t q = (int8_t)std::max(-127L, std::min(127L, std::lround(v * 127.0f)));
    out.assign(reinterpret_cast<const char *>(&q), 1);
    break;
  }
  case TfType::I16:
  {
    const int16_t q = (int16_t)std::max(-32767L, std::min(32767L, std::lround(v * 32767.0f)));
    out.assign(reinterpret_cast<const char *>(&q), 2);
    break;
  }
  default:
    out.assign(reinterpret_cast<const char *>(&v), 4);
    break;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tensor fills
// ---------------------------------------------------------------------------
// Constants are generated straight into the model bytes (TfliteModel::
// addBufferFill), element by element from litertValueAt, so a rung's
// weights exist once in memory.  A fill's context lives in the Recipe until
// build() has run.

namespace
{

struct Fill
{
  TfType type;
  int64_t rows, cols;   // element (i, j) = litertValueAt(i, j, seed) * magnitude
  uint32_t seed;
  float magnitude;
  float scale;          // integer / fp8 types: value / scale, rounded and clamped
  int qmax;
  bool halfRounded;     // fp32 values pre-rounded to fp16 (LitertPlan::halfRounded)
};

#if CLPEAK_LITERT_HW_HALF
float roundHalf(float v) { return (float)(__fp16)v; }
#else
float roundHalf(float v) { return litertHalfToFloat(litertFloatToHalf(v)); }
#endif

// Run `rowFn(i)` for every row on every hardware thread, rows handed out
// in small chunks so a phone's little cores do not decide when the big
// ones finish.  Small fills stay on the calling thread.
template <typename RowFn> void forEachRow(int64_t rows, int64_t cols, RowFn &&rowFn)
{
  unsigned threads = std::thread::hardware_concurrency();
  if (threads == 0 || rows * cols < (1 << 18) || rows < 2)
    threads = 1;
  threads = std::min<unsigned>(threads, (unsigned)rows);
  if (threads == 1)
  {
    for (int64_t i = 0; i < rows; i++)
      rowFn(i);
    return;
  }
  const int64_t chunk = std::max<int64_t>(1, rows / ((int64_t)threads * 16));
  std::atomic<int64_t> next(0);
  std::vector<std::thread> pool;
  pool.reserve(threads);
  for (unsigned t = 0; t < threads; t++)
    pool.emplace_back([&]() {
      for (;;)
      {
        const int64_t begin = next.fetch_add(chunk);
        if (begin >= rows)
          return;
        const int64_t end = std::min(rows, begin + chunk);
        for (int64_t i = begin; i < end; i++)
          rowFn(i);
      }
    });
  for (auto &th : pool)
    th.join();
}

void fillTensor(uint8_t *dst, size_t bytes, void *ctx)
{
  const Fill &f = *static_cast<const Fill *>(ctx);
  const int64_t cols = f.cols;
  // The integer types: the stored code for a value, rounded and clamped.
  auto code = [&](float v) -> long {
    const long q = std::lround(v / f.scale);
    return std::max(-(long)f.qmax, std::min((long)f.qmax, q));
  };
  switch (f.type)
  {
  case TfType::F32:
  {
    float *p = reinterpret_cast<float *>(dst);
    forEachRow(f.rows, cols, [&](int64_t i) {
      const uint32_t key = rowKey(i, f.seed);
      float *row = p + i * cols;
      if (f.halfRounded)
        for (int64_t j = 0; j < cols; j++)
          row[j] = roundHalf(valueIn(key, j) * f.magnitude);
      else
        for (int64_t j = 0; j < cols; j++)
          row[j] = valueIn(key, j) * f.magnitude;
    });
    break;
  }
  case TfType::F16:
  {
    uint16_t *p = reinterpret_cast<uint16_t *>(dst);
    forEachRow(f.rows, cols, [&](int64_t i) {
      const uint32_t key = rowKey(i, f.seed);
      uint16_t *row = p + i * cols;
      for (int64_t j = 0; j < cols; j++)
        row[j] = litertFloatToHalf(valueIn(key, j) * f.magnitude);
    });
    break;
  }
  case TfType::BF16:
  {
    uint16_t *p = reinterpret_cast<uint16_t *>(dst);
    forEachRow(f.rows, cols, [&](int64_t i) {
      const uint32_t key = rowKey(i, f.seed);
      uint16_t *row = p + i * cols;
      for (int64_t j = 0; j < cols; j++)
        row[j] = litertFloatToBf16(valueIn(key, j) * f.magnitude);
    });
    break;
  }
  case TfType::F8E4M3:
    forEachRow(f.rows, cols, [&](int64_t i) {
      const uint32_t key = rowKey(i, f.seed);
      uint8_t *row = dst + i * cols;
      for (int64_t j = 0; j < cols; j++)
        row[j] = litertFloatToFp8E4M3(valueIn(key, j) * f.magnitude / f.scale);
    });
    break;
  case TfType::I8:
  {
    int8_t *p = reinterpret_cast<int8_t *>(dst);
    forEachRow(f.rows, cols, [&](int64_t i) {
      const uint32_t key = rowKey(i, f.seed);
      int8_t *row = p + i * cols;
      for (int64_t j = 0; j < cols; j++)
        row[j] = (int8_t)code(valueIn(key, j) * f.magnitude);
    });
    break;
  }
  case TfType::I16:
  {
    int16_t *p = reinterpret_cast<int16_t *>(dst);
    forEachRow(f.rows, cols, [&](int64_t i) {
      const uint32_t key = rowKey(i, f.seed);
      int16_t *row = p + i * cols;
      for (int64_t j = 0; j < cols; j++)
        row[j] = (int16_t)code(valueIn(key, j) * f.magnitude);
    });
    break;
  }
  case TfType::I4:
  {
    // Two per byte, the first in the low nibble, two's complement.  A row
    // of an even width owns whole bytes, so rows fill independently; an
    // odd width (no recipe has one) goes element by element.
    std::memset(dst, 0, bytes);
    if (cols % 2 == 0)
      forEachRow(f.rows, cols, [&](int64_t i) {
        const uint32_t key = rowKey(i, f.seed);
        uint8_t *row = dst + i * (cols / 2);
        for (int64_t j = 0; j < cols; j += 2)
        {
          const uint8_t lo = (uint8_t)(code(valueIn(key, j) * f.magnitude) & 0xf);
          const uint8_t hi = (uint8_t)(code(valueIn(key, j + 1) * f.magnitude) & 0xf);
          row[j / 2] = (uint8_t)(lo | (hi << 4));
        }
      });
    else
    {
      const int64_t n = f.rows * cols;
      for (int64_t e = 0; e < n; e++)
      {
        const uint8_t nib = (uint8_t)(code(litertValueAt(e / cols, e % cols, f.seed) * f.magnitude) & 0xf);
        dst[e / 2] |= (uint8_t)((e & 1) ? (nib << 4) : nib);
      }
    }
    break;
  }
  default:
    std::memset(dst, 0, bytes);
    break;
  }
}

// The value a stored code stands for, for the accuracy reference: the exact
// product of the stored scale and the code.
float storedValue(const Fill &f, int64_t i, int64_t j)
{
  const float v = litertValueAt(i, j, f.seed) * f.magnitude;
  switch (f.type)
  {
  case TfType::F32: return f.halfRounded ? roundHalf(v) : v;
  case TfType::F16: return litertHalfToFloat(litertFloatToHalf(v));
  case TfType::BF16: return litertBf16ToFloat(litertFloatToBf16(v));
  case TfType::F8E4M3: return litertFp8E4M3ToFloat(litertFloatToFp8E4M3(v / f.scale)) * f.scale;
  case TfType::I8:
  case TfType::I16:
  case TfType::I4:
  {
    const long q = std::lround(v / f.scale);
    return f.scale * (float)std::max(-(long)f.qmax, std::min((long)f.qmax, q));
  }
  default: return v;
  }
}

int bitsOf(TfType t) { return (int)tfliteElementBits(t); }
int qmaxOf(TfType t) { return (1 << (bitsOf(t) - 1)) - 1; }

bool isInteger(TfType t) { return t == TfType::I8 || t == TfType::I16 || t == TfType::I4; }

// A model under construction plus everything its fills and buffers point at.
struct Recipe
{
  TfliteModel m;
  std::vector<std::unique_ptr<Fill>> fills;
  std::vector<std::vector<uint16_t>> halfBuffers;   // for addBuffer, which does not copy
  std::vector<std::vector<int32_t>> intBuffers;
  bool halfRounded = false;   // from the plan; applies to every fp32 fill

  int constant(int64_t rows, int64_t cols, TfType type, uint32_t seed, float magnitude,
               float scale = 1.0f, int qmax = 0)
  {
    fills.push_back(std::unique_ptr<Fill>(
        new Fill{type, rows, cols, seed, magnitude, scale, qmax, halfRounded && type == TfType::F32}));
    return m.addBufferFill(litertElemBytes(type, rows * cols), fillTensor, fills.back().get());
  }
  int ints(std::vector<int32_t> v)
  {
    intBuffers.push_back(std::move(v));
    return m.addBuffer(intBuffers.back().data(), intBuffers.back().size() * 4);
  }
  int halves(std::vector<uint16_t> v)
  {
    halfBuffers.push_back(std::move(v));
    return m.addBuffer(halfBuffers.back().data(), halfBuffers.back().size() * 2);
  }

  // An fp16 constant dequantized into an fp32 graph (DEQUANTIZE version 3,
  // the converter's own float16-quantization form): the tensor the graph
  // consumes is fp32, the bytes in the model are half.
  int dequantized(const std::vector<int32_t> &shape, const std::string &name, int buf, const TfQuant &q)
  {
    const int packed = m.addTensor(0, shape, TfType::F16, name + "_h", buf);
    const int t = m.addTensor(0, shape, TfType::F32, name, 0, q);
    m.addOp(0, TfOp::Dequantize, 3, {packed}, {t});
    return t;
  }

  // An activation-shaped constant as a tensor of the activation type, stored
  // as the plan says (litertConstantType).  `scale`/`qmax` are the integer
  // types' code mapping.
  int actConstant(const LitertPlan &p, const std::vector<int32_t> &shape, const std::string &name,
                  int64_t rows, int64_t cols, uint32_t seed, float magnitude, float scale = 1.0f,
                  int qmax = 0, const TfQuant &q = TfQuant())
  {
    const TfType stored = litertConstantType(p);
    const int buf = constant(rows, cols, stored, seed, magnitude, scale, qmax);
    if (stored == p.act)
      return m.addTensor(0, shape, p.act, name, buf, q);
    return dequantized(shape, name, buf, q);
  }

  // A float weight in the plan's weight type; an fp16 weight in an fp32
  // graph is dequantized into it.
  int floatWeight(const LitertPlan &p, const std::vector<int32_t> &shape, const std::string &name,
                  int64_t rows, int64_t cols, uint32_t seed, float magnitude)
  {
    const int buf = constant(rows, cols, p.weight, seed, magnitude);
    if (p.weight == TfType::F16 && p.act == TfType::F32)
      return dequantized(shape, name, buf, TfQuant());
    return m.addTensor(0, shape, p.weight, name, buf);
  }
};

// The quantization a plan's activations carry, at a given scale.
TfQuant actQuant(const LitertPlan &p, float scale)
{
  TfQuant q;
  if (isInteger(p.act))
  {
    q.scale = {scale};
    q.zeroPoint = {0};
  }
  return q;
}

// W [N, K] in the plan's weight format; returns the tensor index and,
// when asked, the fill so a reference can recover the stored values.
int addWeight(Recipe &r, const LitertPlan &p, int64_t N, int64_t K, uint32_t seed, const Fill **fill)
{
  const TfType wt = p.weight;
  int tensor;
  if (wt == TfType::F32 || wt == TfType::F16 || wt == TfType::BF16)
    tensor = r.floatWeight(p, {(int32_t)N, (int32_t)K}, "W", N, K, seed, 1.0f);
  else if (wt == TfType::F8E4M3)
  {
    // One scale per row; codes span a comfortable part of the format's range
    // (values up to 0.5 land at up to 8.0 in the format).
    const float scale = 1.0f / 16.0f;
    const int buf = r.constant(N, K, wt, seed, 1.0f, scale);
    TfQuant q;
    q.scale.assign((size_t)N, scale);
    q.zeroPoint.assign((size_t)N, 0);
    q.quantizedDim = 0;
    tensor = r.m.addTensor(0, {(int32_t)N, (int32_t)K}, wt, "W", buf, q);
  }
  else
  {
    const float scale = litertWeightScale(bitsOf(wt));
    const int buf = r.constant(N, K, wt, seed, 1.0f, scale, qmaxOf(wt));
    TfQuant q;
    if (p.weightBlock > 0)
    {
      // Blockwise: one fp16 scale per block of `weightBlock` along K, in a
      // tensor of its own that the weight's quantization names by index.
      const int64_t nb = K / p.weightBlock;
      const int sbuf = r.halves(std::vector<uint16_t>((size_t)(N * nb), litertFloatToHalf(scale)));
      const int st = r.m.addTensor(0, {(int32_t)N, (int32_t)nb}, TfType::F16, "W_scales", sbuf);
      q.blockSize = p.weightBlock;
      q.scalesTensor = st;
      q.quantizedDim = 0;
    }
    else
    {
      q.scale.assign((size_t)N, scale);
      q.zeroPoint.assign((size_t)N, 0);
      q.quantizedDim = 0;
    }
    tensor = r.m.addTensor(0, {(int32_t)N, (int32_t)K}, wt, "W", buf, q);
  }
  if (fill)
    *fill = r.fills.back().get();
  return tensor;
}

// Whether a plan's FULLY_CONNECTED carries a bias: the full-integer int8
// graphs do, because a converted model always has one and an accelerator's
// int8 kernel is written for that form (Mali's int8 path returned a wrong
// answer for the bias-less form on a Pixel 7a; whether the bias is what it
// wanted is what the next run there says).  The int16 plan stays bias-less:
// its reference kernel wants an int64 bias and nothing else runs it.
bool fcHasBias(const LitertPlan &p) { return p.act == TfType::I8 && p.weight == TfType::I8; }

// The int32 zero bias of an integer FULLY_CONNECTED: one per output row,
// at the product of the input and weight scales.
int addFcBias(Recipe &r, int64_t N, float aScale, float wScale)
{
  TfQuant bq;
  bq.scale.assign((size_t)N, aScale * wScale);
  bq.zeroPoint.assign((size_t)N, 0);
  bq.quantizedDim = 0;
  return r.m.addTensor(0, {(int32_t)N}, TfType::I32, "bias", r.ints(std::vector<int32_t>((size_t)N, 0)), bq);
}

// The scalar input that keeps a graph live, in the activation type.
int addScalar(Recipe &r, const LitertPlan &p)
{
  TfQuant q;
  if (p.act == TfType::I8)
  {
    q.scale = {1.0f / 127.0f};
    q.zeroPoint = {0};
  }
  else if (p.act == TfType::I16)
  {
    q.scale = {1.0f / 32767.0f};
    q.zeroPoint = {0};
  }
  return r.m.addTensor(0, {1}, p.act, "s", 0, q);
}

int mulVersion(const LitertPlan &p)
{
  // op_version.cc: fp16 8, int16 4, int8 2 (the scale product is below one
  // here, so never 3), else 1.
  if (p.act == TfType::F16) return 8;
  if (p.act == TfType::I16) return 4;
  if (p.act == TfType::I8) return 2;
  return 1;
}

int reduceVersion(const LitertPlan &p)
{
  // REDUCE_MAX: int16 3, int8 2, else 1.
  if (p.act == TfType::I16) return 3;
  if (p.act == TfType::I8) return 2;
  return 1;
}

TfOptions fcOptions(const LitertPlan &p)
{
  TfOptions o;
  o.kind = TfOptions::Kind::FullyConnected;
  o.keepNumDims = true;
  o.asymmetricQuantizeInputs = p.dynamicQuant;
  return o;
}

TfOptions reduceOptions()
{
  TfOptions o;
  o.kind = TfOptions::Kind::Reducer;
  o.keepDims = true;
  return o;
}

TfOptions mulOptions()
{
  TfOptions o;
  o.kind = TfOptions::Kind::Mul;
  return o;
}

std::string describe(const LitertPlan &p, const char *what)
{
  return std::string("clpeak ") + what + " " + tfliteTypeName(p.act) + "/" + tfliteTypeName(p.weight);
}

} // namespace

// ---------------------------------------------------------------------------
// Recipes
// ---------------------------------------------------------------------------

TfliteBytes litertMatMulModel(const LitertPlan &p, int64_t M, int64_t K, int64_t N,
                              uint32_t seedA, uint32_t seedW)
{
  Recipe r;
  r.halfRounded = p.halfRounded;
  r.m.reserveBytes(litertElemBytes(litertConstantType(p), M * K) + litertElemBytes(p.weight, N * K));
  const int abits = bitsOf(p.act);
  const float aScale = isInteger(p.act) ? litertActScale(abits) : 1.0f;
  const TfQuant aq = actQuant(p, aScale);

  const int a = r.actConstant(p, {1, (int32_t)M, (int32_t)K}, "A", M, K, seedA, 1.0f,
                              isInteger(p.act) ? aScale : 1.0f, isInteger(p.act) ? qmaxOf(p.act) : 0, aq);
  const int s = addScalar(r, p);
  const int as = r.m.addTensor(0, {1, (int32_t)M, (int32_t)K}, p.act, "As", 0, aq);
  const int w = addWeight(r, p, N, K, seedW, nullptr);
  const bool bias = fcHasBias(p);
  const int b = bias ? addFcBias(r, N, aScale, litertWeightScale(bitsOf(p.weight))) : -1;
  const TfQuant cq = actQuant(p, isInteger(p.act) ? litertOutScale(K, abits) : 1.0f);
  const int c = r.m.addTensor(0, {1, (int32_t)M, (int32_t)N}, p.act, "C", 0, cq);
  const int ax = r.m.addTensor(0, {1}, TfType::I32, "axes", r.ints({1}));
  const int out = r.m.addTensor(0, {1, 1, (int32_t)N}, p.act, "out", 0, cq);

  r.m.addOp(0, TfOp::Mul, mulVersion(p), {a, s}, {as}, mulOptions());
  r.m.addOp(0, TfOp::FullyConnected, litertFcVersion(p, bias), {as, w, b}, {c}, fcOptions(p));
  r.m.addOp(0, TfOp::ReduceMax, reduceVersion(p), {c, ax}, {out}, reduceOptions());
  r.m.setInputs(0, {s});
  r.m.setOutputs(0, {out});
  return r.m.build(describe(p, "matmul"));
}

TfliteBytes litertPlainMatMulModel(const LitertPlan &p, int64_t M, int64_t K, int64_t N,
                                   std::vector<float> *weights, uint32_t seedW)
{
  Recipe r;
  r.halfRounded = p.halfRounded;
  r.m.reserveBytes(litertElemBytes(p.weight, N * K));
  const int abits = bitsOf(p.act);
  const float aScale = isInteger(p.act) ? litertActScale(abits) : 1.0f;
  const TfQuant aq = actQuant(p, aScale);
  const int x = r.m.addTensor(0, {1, (int32_t)M, (int32_t)K}, p.act, "x", 0, aq);
  const Fill *wf = nullptr;
  const int w = addWeight(r, p, N, K, seedW, &wf);
  const bool bias = fcHasBias(p);
  const int b = bias ? addFcBias(r, N, aScale, litertWeightScale(bitsOf(p.weight))) : -1;
  const TfQuant yq = actQuant(p, isInteger(p.act) ? litertOutScale(K, abits) : 1.0f);
  const int y = r.m.addTensor(0, {1, (int32_t)M, (int32_t)N}, p.act, "y", 0, yq);
  r.m.addOp(0, TfOp::FullyConnected, litertFcVersion(p, bias), {x, w, b}, {y}, fcOptions(p));
  r.m.setInputs(0, {x});
  r.m.setOutputs(0, {y});
  if (weights && wf)
  {
    weights->resize((size_t)(N * K));
    for (int64_t n = 0; n < N; n++)
      for (int64_t k = 0; k < K; k++)
        (*weights)[(size_t)(n * K + k)] = storedValue(*wf, n, k);
  }
  return r.m.build(describe(p, "plain matmul"));
}

TfliteBytes litertGemvModel(const LitertPlan &p, int64_t K, int64_t N, uint32_t seedW)
{
  Recipe r;
  r.halfRounded = p.halfRounded;
  r.m.reserveBytes(litertElemBytes(p.weight, N * K));
  const int abits = bitsOf(p.act);
  const float xScale = isInteger(p.act) ? litertActScale(abits) : 1.0f;
  const TfQuant xq = actQuant(p, xScale);
  const int x = r.m.addTensor(0, {1, 1, (int32_t)K}, p.act, "x", 0, xq);
  const int w = addWeight(r, p, N, K, seedW, nullptr);
  const bool bias = fcHasBias(p);
  const int b = bias ? addFcBias(r, N, xScale, litertWeightScale(bitsOf(p.weight))) : -1;
  const TfQuant yq = actQuant(p, isInteger(p.act) ? litertOutScale(K, abits) : 1.0f);
  const int y = r.m.addTensor(0, {1, 1, (int32_t)N}, p.act, "y", 0, yq);
  r.m.addOp(0, TfOp::FullyConnected, litertFcVersion(p, bias), {x, w, b}, {y}, fcOptions(p));
  r.m.setInputs(0, {x});
  r.m.setOutputs(0, {y});
  return r.m.build(describe(p, "gemv"));
}

TfliteBytes litertActivationModel(const LitertPlan &p, int64_t rows, int64_t cols,
                                  LitertActivation act)
{
  Recipe r;
  r.halfRounded = p.halfRounded;
  r.m.reserveBytes(litertElemBytes(litertConstantType(p), rows * cols));
  const std::vector<int32_t> shape = {1, (int32_t)rows, (int32_t)cols};
  // Magnitude 4: softmax and the normalisation need a spread to work on.
  const int x0 = r.actConstant(p, shape, "X0", rows, cols, 0x6a09e667u, 4.0f);
  const int s = addScalar(r, p);
  const int x = r.m.addTensor(0, shape, p.act, "X", 0);
  r.m.addOp(0, TfOp::Mul, mulVersion(p), {x0, s}, {x}, mulOptions());
  int y = x;
  switch (act)
  {
  case LitertActivation::None:
    break;
  case LitertActivation::Silu:
  {
    const int sig = r.m.addTensor(0, shape, p.act, "sig", 0);
    y = r.m.addTensor(0, shape, p.act, "Y", 0);
    r.m.addOp(0, TfOp::Logistic, 1, {x}, {sig});
    r.m.addOp(0, TfOp::Mul, mulVersion(p), {x, sig}, {y}, mulOptions());
    break;
  }
  case LitertActivation::Softmax:
  {
    y = r.m.addTensor(0, shape, p.act, "Y", 0);
    TfOptions o;
    o.kind = TfOptions::Kind::Softmax;
    o.beta = 1.0f;
    r.m.addOp(0, TfOp::Softmax, 1, {x}, {y}, o);
    break;
  }
  case LitertActivation::LayerNorm:
  {
    // The decomposition every converter emits: mean, centre, variance,
    // rsqrt, rescale -- seven operators over the row.
    const std::vector<int32_t> rowShape = {1, (int32_t)rows, 1};
    const int axes = r.m.addTensor(0, {1}, TfType::I32, "ln_axes", r.ints({2}));
    const int mu = r.m.addTensor(0, rowShape, p.act, "mu", 0);
    const int xc = r.m.addTensor(0, shape, p.act, "xc", 0);
    const int sq = r.m.addTensor(0, shape, p.act, "sq", 0);
    const int var = r.m.addTensor(0, rowShape, p.act, "var", 0);
    const std::string epsBytes = litertScalarBytes(p.act, 1.0e-5f);
    r.halfBuffers.push_back(std::vector<uint16_t>((epsBytes.size() + 1) / 2, 0));
    std::memcpy(r.halfBuffers.back().data(), epsBytes.data(), epsBytes.size());
    const int epsBuf = r.m.addBuffer(r.halfBuffers.back().data(), epsBytes.size());
    const int eps = r.m.addTensor(0, {1}, p.act, "eps", epsBuf);
    const int ve = r.m.addTensor(0, rowShape, p.act, "ve", 0);
    const int inv = r.m.addTensor(0, rowShape, p.act, "inv", 0);
    y = r.m.addTensor(0, shape, p.act, "Y", 0);
    r.m.addOp(0, TfOp::Mean, 1, {x, axes}, {mu}, reduceOptions());
    TfOptions sub;
    sub.kind = TfOptions::Kind::Sub;
    r.m.addOp(0, TfOp::Sub, 1, {x, mu}, {xc}, sub);
    r.m.addOp(0, TfOp::Square, 1, {xc}, {sq});
    r.m.addOp(0, TfOp::Mean, 1, {sq, axes}, {var}, reduceOptions());
    TfOptions add;
    add.kind = TfOptions::Kind::Add;
    r.m.addOp(0, TfOp::Add, 1, {var, eps}, {ve}, add);
    r.m.addOp(0, TfOp::Rsqrt, 1, {ve}, {inv});
    r.m.addOp(0, TfOp::Mul, mulVersion(p), {xc, inv}, {y}, mulOptions());
    break;
  }
  }
  const int ax = r.m.addTensor(0, {1}, TfType::I32, "axes", r.ints({1}));
  const int out = r.m.addTensor(0, {1, 1, (int32_t)cols}, p.act, "out", 0);
  r.m.addOp(0, TfOp::ReduceMax, reduceVersion(p), {y, ax}, {out}, reduceOptions());
  r.m.setInputs(0, {s});
  r.m.setOutputs(0, {out});
  return r.m.build(describe(p, "activation"));
}

TfliteBytes litertTransferModel(const LitertPlan &p, LitertTransfer dir, int64_t elems)
{
  Recipe r;
  const std::vector<int32_t> shape = {1, (int32_t)elems};
  const int x = r.m.addTensor(0, shape, p.act, "X", 0);
  if (dir == LitertTransfer::ToDevice)
  {
    // Everything arrives; one element goes back.  A slice rather than a
    // reduction, so that on an accelerator with no real transfer the row
    // does not become its reduction rate.
    const int begin = r.m.addTensor(0, {2}, TfType::I32, "begin", r.ints({0, 0}));
    const int end = r.m.addTensor(0, {2}, TfType::I32, "end", r.ints({1, 1}));
    const int strides = r.m.addTensor(0, {2}, TfType::I32, "strides", r.ints({1, 1}));
    const int y = r.m.addTensor(0, {1, 1}, p.act, "Y", 0);
    r.m.addOp(0, TfOp::StridedSlice, 1, {x, begin, end, strides}, {y});
    r.m.setInputs(0, {x});
    r.m.setOutputs(0, {y});
  }
  else
  {
    // Squared rather than scaled: the same shape on both operands needs no
    // broadcast and no constant.
    const int y = r.m.addTensor(0, shape, p.act, "Y", 0);
    r.m.addOp(0, TfOp::Mul, mulVersion(p), {x, x}, {y}, mulOptions());
    r.m.setInputs(0, {x});
    r.m.setOutputs(0, {y});
  }
  return r.m.build(describe(p, "transfer"));
}

TfliteBytes litertTrivialModel(const LitertPlan &p, int64_t width)
{
  Recipe r;
  const std::vector<int32_t> shape = {1, (int32_t)width};
  const int x = r.m.addTensor(0, shape, p.act, "X", 0);
  // A full-size constant rather than a scalar: a scalar broadcast is a
  // different kernel on some accelerators.
  const int k = r.m.addTensor(0, shape, p.act, "K", r.constant(1, width, p.act, 0x13198a2eu, 1.0f));
  const int y = r.m.addTensor(0, shape, p.act, "Y", 0);
  r.m.addOp(0, TfOp::Mul, mulVersion(p), {x, k}, {y}, mulOptions());
  r.m.setInputs(0, {x});
  r.m.setOutputs(0, {y});
  return r.m.build(describe(p, "trivial"));
}

uint64_t litertWeightBytes(const LitertPlan &p, int64_t N, int64_t K)
{
  uint64_t bytes = litertElemBytes(p.weight, N * K);
  if (p.weightBlock > 0)
    bytes += (uint64_t)(N * (K / p.weightBlock)) * 2;   // fp16 scale per block
  return bytes;
}

namespace
{

// A 4-D filter in the plan's weight format: [outC, k, k, inPerGroup] for
// CONV_2D, [1, k, k, channels] for DEPTHWISE_CONV_2D.  The weights are
// shrunk with the fan-in so a 3x3 x 256 sum stays inside fp16.  Blockwise
// scales are a FULLY_CONNECTED notion, so the int4 block plan degrades to
// per-channel here (the accuracy of that is not what a conv row measures).
int addFilter(Recipe &r, const LitertPlan &p, int64_t channels, int64_t kernel, bool depthwise)
{
  const int64_t inPerGroup = depthwise ? 1 : channels;
  const int64_t fanIn = inPerGroup * kernel * kernel;
  const float mag = 2.0f / std::sqrt((float)fanIn);
  const std::vector<int32_t> shape = depthwise
      ? std::vector<int32_t>{1, (int32_t)kernel, (int32_t)kernel, (int32_t)channels}
      : std::vector<int32_t>{(int32_t)channels, (int32_t)kernel, (int32_t)kernel, (int32_t)inPerGroup};
  // As a 2-D fill: rows = outC (or 1, depthwise), cols = the rest.
  const int64_t rows = depthwise ? 1 : channels;
  const int64_t cols = depthwise ? kernel * kernel * channels : fanIn;
  const TfType wt = p.weight;
  if (wt == TfType::F32 || wt == TfType::F16 || wt == TfType::BF16)
    return r.floatWeight(p, shape, "W", rows, cols, 0x7f4a7c15u, mag);
  const float scale = litertWeightScale(bitsOf(wt)) * mag;
  const int buf = r.constant(rows, cols, wt, 0x7f4a7c15u, mag, scale, qmaxOf(wt));
  TfQuant q;
  q.scale.assign((size_t)channels, scale);
  q.zeroPoint.assign((size_t)channels, 0);
  q.quantizedDim = depthwise ? 3 : 0;
  return r.m.addTensor(0, shape, wt, "W", buf, q);
}

int convVersion(const LitertPlan &p, bool depthwise)
{
  // op_version.cc: int8 in/out with int8 weights -> 3; int8 in with int4
  // weights -> 7; float in with int8 per-channel weights -> 5 (conv) / 6
  // (depthwise); else 1.
  if (p.act == TfType::I8 && p.weight == TfType::I8)
    return 3;
  if (p.act == TfType::I8 && p.weight == TfType::I4)
    return 7;
  if (p.act == TfType::F32 && p.weight == TfType::I8)
    return depthwise ? 6 : 5;
  return 1;
}

} // namespace

TfliteBytes litertConvModel(const LitertPlan &p, int64_t channels, int64_t spatial, int64_t kernel,
                            bool depthwise)
{
  Recipe r;
  r.halfRounded = p.halfRounded;
  r.m.reserveBytes(litertElemBytes(litertConstantType(p), channels * spatial * spatial));
  const int abits = bitsOf(p.act);
  const float aScale = isInteger(p.act) ? litertActScale(abits) : 1.0f;
  const TfQuant aq = actQuant(p, aScale);
  const std::vector<int32_t> shape = {1, (int32_t)spatial, (int32_t)spatial, (int32_t)channels};

  const int x0 = r.actConstant(p, shape, "X0", spatial * spatial, channels, 0x9e3779b9u, 1.0f,
                               isInteger(p.act) ? aScale : 1.0f, isInteger(p.act) ? qmaxOf(p.act) : 0, aq);
  const int s = addScalar(r, p);
  const int x = r.m.addTensor(0, shape, p.act, "X", 0, aq);
  r.m.addOp(0, TfOp::Mul, mulVersion(p), {x0, s}, {x}, mulOptions());

  const int w = addFilter(r, p, channels, kernel, depthwise);
  // The convolution kernels want their bias ("Tensor at index 2 was optional
  // but was expected"): zeros, int32 at the product of the two scales for the
  // integer plan, the activation type otherwise.
  int bias;
  if (isInteger(p.act))
  {
    const float wScale = litertWeightScale(bitsOf(p.weight)) *
                         (2.0f / std::sqrt((float)((depthwise ? 1 : channels) * kernel * kernel)));
    TfQuant bq;
    bq.scale.assign((size_t)channels, aScale * wScale);
    bq.zeroPoint.assign((size_t)channels, 0);
    bq.quantizedDim = 0;
    bias = r.m.addTensor(0, {(int32_t)channels}, TfType::I32, "bias",
                         r.ints(std::vector<int32_t>((size_t)channels, 0)), bq);
  }
  else
  {
    r.halfBuffers.push_back(std::vector<uint16_t>((size_t)channels * 2, 0));   // zero in any width
    bias = r.m.addTensor(0, {(int32_t)channels}, p.act, "bias",
                         r.m.addBuffer(r.halfBuffers.back().data(), litertElemBytes(p.act, channels)));
  }
  // The output's scale: a fan-in-deep dot product of the activations and
  // the shrunk weights lands within the activations' own range.
  const TfQuant yq = actQuant(p, aScale);
  const int y = r.m.addTensor(0, shape, p.act, "Y", 0, yq);
  TfOptions co;
  co.kind = depthwise ? TfOptions::Kind::DepthwiseConv2d : TfOptions::Kind::Conv2d;
  co.padding = TfPadding::Same;
  co.strideW = co.strideH = 1;
  co.depthMultiplier = 1;
  r.m.addOp(0, depthwise ? TfOp::DepthwiseConv2d : TfOp::Conv2d, convVersion(p, depthwise), {x, w, bias}, {y}, co);

  const int ax = r.m.addTensor(0, {2}, TfType::I32, "axes", r.ints({1, 2}));
  const int out = r.m.addTensor(0, {1, 1, 1, (int32_t)channels}, p.act, "out", 0, yq);
  r.m.addOp(0, TfOp::ReduceMax, reduceVersion(p), {y, ax}, {out}, reduceOptions());
  r.m.setInputs(0, {s});
  r.m.setOutputs(0, {out});
  return r.m.build(describe(p, depthwise ? "depthwise conv" : "conv"));
}

// ---------------------------------------------------------------------------
// The transformer block
// ---------------------------------------------------------------------------

namespace
{

// The composite's attributes as a FlexBuffer: the map {"scale": <double>}
// AI Edge Torch writes for odml.scaled_dot_product_attention, laid out
// byte for byte as the FlexBuffers encoder does (a 64-bit-wide map because
// 1/sqrt(128) is not a float32; checked against the Python encoder's
// output).  Trailing: root offset, root type (map, 64-bit), root width.
std::vector<uint8_t> sdpaAttributes(double scale)
{
  std::vector<uint8_t> out = {'s', 'c', 'a', 'l', 'e', 0,   // the key
                              1, 7};                         // keys vector: size, offset back to the key
  auto u64 = [&](uint64_t v) {
    for (int i = 0; i < 8; i++)
      out.push_back((uint8_t)(v >> (8 * i)));
  };
  u64(1);   // offset back to the keys vector
  u64(1);   // the keys vector's byte width
  u64(1);   // one entry
  uint64_t bits;
  std::memcpy(&bits, &scale, 8);
  u64(bits);                      // the value
  out.push_back(0x0f);            // its type: FBT_FLOAT, 64-bit
  out.push_back(0x09);            // root: offset back to the values
  out.push_back(0x27);            // root type: FBT_MAP, 64-bit
  out.push_back(0x01);            // root byte width
  return out;
}

// The activation scale for a quantized projection input: four sigma of a
// d-deep dot product of [-0.25, 0.25) operands, the ONNX backend's
// blockQdqScale.  One scale for all seven projections; the feed-forward's
// second input runs larger and saturates, which costs nothing here (int8
// saturation is finite) -- this row measures rate.
float blockActScale(int64_t d)
{
  return (float)(4.0 * std::sqrt((double)d) / 3.0 / 127.0);
}

} // namespace

TfliteBytes litertBlockModel(const LitertPlan &p, const LitertBlockShape &sh)
{
  Recipe r;
  r.halfRounded = p.halfRounded;
  const int64_t d = sh.dModel, H = sh.heads, Dh = sh.headDim, ffn = sh.ffnHidden, S = sh.seq;
  const bool decode = sh.kvLen > 0;
  const int64_t ctx = decode ? sh.kvLen : S;
  // Weights and activations in [-0.25, 0.25): the SwiGLU squares magnitudes
  // and the down projection sums thousands of terms, which overflowed fp16
  // at the [-0.5, 0.5) the GEMM rows use.
  const float mag = 0.5f;
  // The quantized (int8_qdq) block keeps everything but the projections in
  // float: TFLite's QUANTIZE takes float32, and attention, the softmax and
  // the SwiGLU are never quantized in a real deployment either.
  const bool qdq = (p.act == TfType::I8);
  const TfType act = qdq ? TfType::F32 : p.act;
  LitertPlan fp = p;   // the float parts' plan: same act type, float weights
  fp.act = act;
  fp.weight = act;
  fp.perChannel = false;
  fp.weightBlock = 0;
  fp.dynamicQuant = false;

  const TfType stored = litertConstantType(fp);   // the float constants' stored type
  r.m.reserveBytes(4 * litertWeightBytes(p, d, d) + 2 * litertWeightBytes(p, ffn, d) +
                   litertWeightBytes(p, d, ffn) +
                   (decode ? 2 * litertElemBytes(sh.int8Kv ? TfType::I8 : stored, H * ctx * Dh) : 0) +
                   litertElemBytes(stored, S * d));

  auto tensor = [&](const std::vector<int32_t> &shape, const std::string &name) {
    return r.m.addTensor(0, shape, act, name, 0);
  };
  const std::vector<int32_t> xShape = {1, (int32_t)S, (int32_t)d};

  const int x0 = r.actConstant(fp, xShape, "X0", S, d, 0xa5a5a5a5u, mag);
  const int s = addScalar(r, fp);
  const int x = tensor(xShape, "X");
  r.m.addOp(0, TfOp::Mul, mulVersion(fp), {x0, s}, {x}, mulOptions());

  // ---- One projection, in whichever format the plan asks for -------------
  // Distinct seeds so no two projections share a matrix; a repeated weight
  // would let a runtime cache or fold work that a real model cannot.
  const float qa = blockActScale(d);
  auto projection = [&](const std::string &name, int in, int64_t K, int64_t N, uint32_t seed) -> int {
    const std::vector<int32_t> outShape = {1, (int32_t)S, (int32_t)N};
    // W [N, K] in the plan's format, shrunk to [-0.25, 0.25).
    int w;
    {
      const TfType wt = p.weight;
      if (wt == TfType::F32 || wt == TfType::F16 || wt == TfType::BF16)
        w = r.floatWeight(p, {(int32_t)N, (int32_t)K}, "W_" + name, N, K, seed, mag);
      else if (wt == TfType::F8E4M3)
      {
        const float scale = 1.0f / 16.0f;
        TfQuant q;
        q.scale.assign((size_t)N, scale);
        q.zeroPoint.assign((size_t)N, 0);
        q.quantizedDim = 0;
        w = r.m.addTensor(0, {(int32_t)N, (int32_t)K}, wt, "W_" + name,
                          r.constant(N, K, wt, seed, mag, scale), q);
      }
      else
      {
        const float scale = litertWeightScale(bitsOf(wt)) * mag;
        const int buf = r.constant(N, K, wt, seed, mag, scale, qmaxOf(wt));
        TfQuant q;
        if (p.weightBlock > 0)
        {
          const int64_t nb = K / p.weightBlock;
          const int sbuf = r.halves(std::vector<uint16_t>((size_t)(N * nb), litertFloatToHalf(scale)));
          q.blockSize = p.weightBlock;
          q.scalesTensor = r.m.addTensor(0, {(int32_t)N, (int32_t)nb}, TfType::F16, "Ws_" + name, sbuf);
          q.quantizedDim = 0;
        }
        else
        {
          q.scale.assign((size_t)N, scale);
          q.zeroPoint.assign((size_t)N, 0);
          q.quantizedDim = 0;
        }
        w = r.m.addTensor(0, {(int32_t)N, (int32_t)K}, wt, "W_" + name, buf, q);
      }
    }
    if (!qdq)
    {
      const int out = tensor(outShape, name);
      r.m.addOp(0, TfOp::FullyConnected, litertFcVersion(p, false), {in, w, -1}, {out}, fcOptions(p));
      return out;
    }
    // W8A8: quantize in, integer multiply, dequantize out.
    TfQuant aq;
    aq.scale = {qa};
    aq.zeroPoint = {0};
    const int inQ = r.m.addTensor(0, {1, (int32_t)S, (int32_t)K}, TfType::I8, name + "_q", 0, aq);
    const int outQ = r.m.addTensor(0, outShape, TfType::I8, name + "_oq", 0, aq);
    const int out = tensor(outShape, name);
    r.m.addOp(0, TfOp::Quantize, 1, {in}, {inQ});
    LitertPlan ip = p;
    ip.act = TfType::I8;
    const int b = addFcBias(r, N, qa, litertWeightScale(bitsOf(p.weight)) * mag);
    r.m.addOp(0, TfOp::FullyConnected, litertFcVersion(ip, true), {inQ, w, b}, {outQ}, fcOptions(ip));
    r.m.addOp(0, TfOp::Dequantize, 2, {outQ}, {out});
    return out;
  };

  // ---- QKV projection ----------------------------------------------------
  const int q = projection("Q", x, d, d, 0x11111111u);
  const std::vector<int32_t> headsShape = {1, (int32_t)S, (int32_t)H, (int32_t)Dh};
  const std::vector<int32_t> hsdShape = {1, (int32_t)H, (int32_t)S, (int32_t)Dh};
  const int permHeads = r.m.addTensor(0, {4}, TfType::I32, "perm_heads", r.ints({0, 2, 1, 3}));
  auto toHeads = [&](int t, const std::string &name) {
    const int rs = tensor(headsShape, name + "r");
    TfOptions ro;
    ro.kind = TfOptions::Kind::Reshape;
    ro.newShape = headsShape;
    const int shapeT = r.m.addTensor(0, {4}, TfType::I32, name + "_shape", r.ints(headsShape));
    r.m.addOp(0, TfOp::Reshape, 1, {t, shapeT}, {rs}, ro);
    const int h = tensor(hsdShape, name + "h");
    r.m.addOp(0, TfOp::Transpose, 1, {rs, permHeads}, {h});
    return h;
  };
  // ---- Attention ---------------------------------------------------------
  // Heads go to the front ([1, H, S, Dh]) for the explicit form.  The
  // composite form keeps AI Edge Torch's boundary layout, [1, S, H, Dh]
  // (batch, tokens, heads, head size), and does the transposes inside the
  // decomposition, the way its exporter does; an accelerator that fuses
  // the composite expects that layout.  Decode reads a resident cache in
  // whichever layout the form wants; prefill builds K/V from this pass.
  const std::vector<int32_t> hsdCache = {1, (int32_t)H, (int32_t)ctx, (int32_t)Dh};
  const std::vector<int32_t> shdCache = {1, (int32_t)ctx, (int32_t)H, (int32_t)Dh};
  const std::vector<int32_t> &cacheShape = sh.composite ? shdCache : hsdCache;
  const int64_t cacheRows = sh.composite ? ctx * H : H * ctx;   // rows of Dh either way
  auto cache = [&](const std::string &name, uint32_t seed) -> int {
    if (!sh.int8Kv)
      return r.actConstant(fp, cacheShape, name, cacheRows, Dh, seed, mag);
    // Quantized per tensor: the cache spends the whole int8 range and one
    // scale quarters it back to the [-0.25, 0.25) the float cache holds.
    const float scale = 0.25f / 127.0f;
    TfQuant cq;
    cq.scale = {scale};
    cq.zeroPoint = {0};
    const int packed = r.m.addTensor(0, cacheShape, TfType::I8, name + "_q",
                                     r.constant(cacheRows, Dh, TfType::I8, seed, mag, scale, 127), cq);
    const int deq = tensor(cacheShape, name);
    r.m.addOp(0, TfOp::Dequantize, 2, {packed}, {deq});
    return deq;
  };
  auto transposeHeads = [&](int sg, int t, const std::vector<int32_t> &to, const std::string &name) {
    const int perm = r.m.addTensor(sg, {4}, TfType::I32, name + "_perm", r.ints({0, 2, 1, 3}));
    const int out = r.m.addTensor(sg, to, act, name, 0);
    r.m.addOp(sg, TfOp::Transpose, 1, {t, perm}, {out});
    return out;
  };

  // Scaled dot-product attention over [1, H, ., Dh] operands: scores =
  // Q K^T * 1/sqrt(Dh), softmax, times V.  Spelt out in subgraph `sg`: the
  // main graph, or the composite's decomposition subgraph.
  const std::vector<int32_t> scoresShape = {1, (int32_t)H, (int32_t)S, (int32_t)ctx};
  const float attnScale = 1.0f / std::sqrt((float)Dh);
  auto attention = [&](int sg, int q_, int k_, int v_) -> int {
    const int scores = r.m.addTensor(sg, scoresShape, act, "Scores", 0);
    TfOptions bmm;
    bmm.kind = TfOptions::Kind::BatchMatMul;
    bmm.adjY = true;
    r.m.addOp(sg, TfOp::BatchMatMul, 1, {q_, k_}, {scores}, bmm);
    const std::string scaleBytes = litertScalarBytes(act, attnScale);
    r.halfBuffers.push_back(std::vector<uint16_t>((scaleBytes.size() + 1) / 2, 0));
    std::memcpy(r.halfBuffers.back().data(), scaleBytes.data(), scaleBytes.size());
    const int scaleT = r.m.addTensor(sg, {1}, act, "attn_scale",
                                     r.m.addBuffer(r.halfBuffers.back().data(), scaleBytes.size()));
    const int scoresS = r.m.addTensor(sg, scoresShape, act, "ScoresS", 0);
    r.m.addOp(sg, TfOp::Mul, mulVersion(fp), {scores, scaleT}, {scoresS}, mulOptions());
    const int probs = r.m.addTensor(sg, scoresShape, act, "P", 0);
    TfOptions so;
    so.kind = TfOptions::Kind::Softmax;
    so.beta = 1.0f;
    r.m.addOp(sg, TfOp::Softmax, 1, {scoresS}, {probs}, so);
    const int out = r.m.addTensor(sg, hsdShape, act, "Ctx", 0);
    TfOptions bmm2;
    bmm2.kind = TfOptions::Kind::BatchMatMul;
    r.m.addOp(sg, TfOp::BatchMatMul, 1, {probs, v_}, {out}, bmm2);
    return out;
  };

  int ctxT;   // attention's answer as [1, S, H, Dh]
  if (!sh.composite)
  {
    const int qh = toHeads(q, "Q");
    int kh, vh;
    if (decode)
    {
      kh = cache("Kc", 0x88888888u);
      vh = cache("Vc", 0x99999999u);
    }
    else
    {
      kh = toHeads(projection("Knew", x, d, d, 0x22222222u), "K");
      vh = toHeads(projection("Vnew", x, d, d, 0x33333333u), "V");
    }
    const int ctxH = attention(0, qh, kh, vh);
    ctxT = tensor(headsShape, "CtxT");
    r.m.addOp(0, TfOp::Transpose, 1, {ctxH, permHeads}, {ctxT});
  }
  else
  {
    // odml.scaled_dot_product_attention over (q, k, v) in [1, ., H, Dh],
    // with the explicit form -- transposes included -- as its decomposition.
    auto toTokens = [&](int t, const std::string &name) {
      const int rs = tensor(headsShape, name + "r");
      TfOptions ro;
      ro.kind = TfOptions::Kind::Reshape;
      ro.newShape = headsShape;
      const int shapeT = r.m.addTensor(0, {4}, TfType::I32, name + "_shape", r.ints(headsShape));
      r.m.addOp(0, TfOp::Reshape, 1, {t, shapeT}, {rs}, ro);
      return rs;
    };
    const int q4 = toTokens(q, "Q");
    int k4, v4;
    if (decode)
    {
      k4 = cache("Kc", 0x88888888u);
      v4 = cache("Vc", 0x99999999u);
    }
    else
    {
      k4 = toTokens(projection("Knew", x, d, d, 0x22222222u), "K");
      v4 = toTokens(projection("Vnew", x, d, d, 0x33333333u), "V");
    }
    const int dsg = r.m.addSubgraph("sdpa");
    const int dq = r.m.addTensor(dsg, headsShape, act, "q", 0);
    const int dk = r.m.addTensor(dsg, shdCache, act, "k", 0);
    const int dv = r.m.addTensor(dsg, shdCache, act, "v", 0);
    const int dout = attention(dsg, transposeHeads(dsg, dq, hsdShape, "qh"),
                               transposeHeads(dsg, dk, hsdCache, "kh"), transposeHeads(dsg, dv, hsdCache, "vh"));
    const int dctx = transposeHeads(dsg, dout, headsShape, "ctx");
    r.m.setInputs(dsg, {dq, dk, dv});
    r.m.setOutputs(dsg, {dctx});
    ctxT = tensor(headsShape, "CtxT");
    TfOptions co;
    co.kind = TfOptions::Kind::Composite;
    co.compositeName = "odml.scaled_dot_product_attention";
    co.decompositionSubgraph = dsg;
    co.compositeVersion = 1;
    co.compositeAttributes = sdpaAttributes(1.0 / std::sqrt((double)Dh));
    r.m.addOp(0, TfOp::StablehloComposite, 1, {q4, k4, v4}, {ctxT}, co);
  }
  const int ctxF = tensor(xShape, "CtxF");
  {
    TfOptions ro;
    ro.kind = TfOptions::Kind::Reshape;
    ro.newShape = xShape;
    const int shapeT = r.m.addTensor(0, {3}, TfType::I32, "flat_shape", r.ints(xShape));
    r.m.addOp(0, TfOp::Reshape, 1, {ctxT, shapeT}, {ctxF}, ro);
  }
  const int attnOut = projection("AttnOut", ctxF, d, d, 0x44444444u);
  const int r1 = tensor(xShape, "R1");
  TfOptions ao;
  ao.kind = TfOptions::Kind::Add;
  r.m.addOp(0, TfOp::Add, 1, {x, attnOut}, {r1}, ao);

  // ---- SwiGLU feed-forward ----------------------------------------------
  const std::vector<int32_t> ffShape = {1, (int32_t)S, (int32_t)ffn};
  const int g = projection("G", r1, d, ffn, 0x55555555u);
  const int u = projection("U", r1, d, ffn, 0x66666666u);
  const int sig = tensor(ffShape, "sig");
  r.m.addOp(0, TfOp::Logistic, 1, {g}, {sig});
  const int actT = tensor(ffShape, "Act");
  r.m.addOp(0, TfOp::Mul, mulVersion(fp), {g, sig}, {actT}, mulOptions());
  const int hh = tensor(ffShape, "Hh");
  r.m.addOp(0, TfOp::Mul, mulVersion(fp), {actT, u}, {hh}, mulOptions());
  const int down = projection("Down", hh, ffn, d, 0x77777777u);
  const int y = tensor(xShape, "Y");
  r.m.addOp(0, TfOp::Add, 1, {r1, down}, {y}, ao);

  const int ax = r.m.addTensor(0, {1}, TfType::I32, "axes", r.ints({1}));
  const int out = tensor({1, 1, (int32_t)d}, "out");
  r.m.addOp(0, TfOp::ReduceMax, reduceVersion(fp), {y, ax}, {out}, reduceOptions());
  r.m.setInputs(0, {s});
  r.m.setOutputs(0, {out});
  return r.m.build(describe(p, decode ? "block decode" : "block prefill"));
}

#endif // ENABLE_LITERT
