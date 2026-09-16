#ifndef CLPEAK_LITERT_MODEL_H
#define CLPEAK_LITERT_MODEL_H

// The models the LiteRT tests run, as recipes over tflite_model.h, and the
// formats they come in.
//
// A format here is what a shipped model *is* -- the storage type of its
// weights and activations and the quantization scheme -- and each
// accelerator runs it as it can: XNNPACK executes a 4-bit-weight graph as
// int8 arithmetic on dynamically quantized activations (its QD8/QB4W
// kernel), the GPU accelerator unpacks the same weights to fp16 and
// multiplies in float, an NPU compiler does whatever its silicon has.  The
// rate row says how fast, the numeric-error row says how accurately, and the
// profiled kernel name says which of those it was.  That is the point of
// keeping one graph per format rather than one per accelerator.
//
// Every activation tensor carries a leading batch dimension of one
// ([1, M, K] rather than [M, K]).  The GPU accelerator maps a 2-D tensor's
// first dimension onto its batch axis and reduces over that axis wrongly
// (a REDUCE_MAX over axis 0 of a [D, D] tensor returned one row of the
// input on an M1 Pro); with the batch axis explicit every accelerator
// agrees with the CPU to the last bit.

#include "tflite_model.h"

#include <litert/litert_peak.h>
#include "litert/c/litert_common.h"   // LiteRtDelegatePrecision

#include <cstdint>
#include <string>
#include <vector>

// The formats.  Labels are the ONNX and Core ML backends' where the format
// is the same one, so a block reading here divides by a GEMM reading there.
enum class LitertFormat
{
  Fp32,        // fp32 storage and arithmetic
  Fp16,        // fp16 storage and arithmetic (the GPU: an fp32 graph under its fp16 policy)
  Fp16Acc32,   // GPU only: fp16 storage, fp32 accumulation in the matmul-class operators
  Bf16,        // bfloat16 tensors: in the schema; whether any kernel takes them is the row
  Int8Qdq,     // full-integer: int8 activations, int8 per-channel weights, int8 result
  Int16x8,     // int16 activations over int8 per-channel weights, int16 result
  Int8Weight,  // int8 per-channel weights against float activations (dynamic-range quantization)
  Int4Weight,  // int4 blockwise weights (32 per scale) against float activations
  Fp8Weight,   // fp8 (E4M3) weights against float activations, if any kernel exists
};

const char *litertFormatLabel(LitertFormat f);

// How a format is expressed for one accelerator: the tensor types of the
// graph and the accelerator policy that goes with it.  `applies` is false
// for a format that is a policy of another accelerator (fp16_acc32 off the
// GPU), with `whyNot` saying so.
struct LitertPlan
{
  bool applies = true;
  std::string whyNot;
  clpeak_tflite::TfType act = clpeak_tflite::TfType::F32;      // activations and result
  clpeak_tflite::TfType weight = clpeak_tflite::TfType::F32;   // stored weight type
  bool perChannel = false;        // one scale per output row
  int32_t weightBlock = 0;        // > 0: blockwise weight scales
  bool dynamicQuant = false;      // FULLY_CONNECTED.asymmetric_quantize_inputs
  bool integerOps = false;        // counts in ops rather than flops
  LiteRtDelegatePrecision gpuPrecision = kLiteRtDelegatePrecisionDefault;
  bool gpuAllowQuantized = false;
};
LitertPlan litertPlanFor(LitertFormat f, LitertAccel accel);

// The version number a converted model would carry for FULLY_CONNECTED in
// this plan (tflite/converter/tools/versioning/op_version.cc).
int litertFcVersion(const LitertPlan &p);

// ---- scalar conversions ---------------------------------------------------
uint16_t litertFloatToHalf(float f);
float litertHalfToFloat(uint16_t h);
uint16_t litertFloatToBf16(float f);
float litertBf16ToFloat(uint16_t b);
uint8_t litertFloatToFp8E4M3(float f);
float litertFp8E4M3ToFloat(uint8_t v);

// ---- operand values -------------------------------------------------------
// Deterministic uniform values in [-0.5, 0.5) * magnitude: the same
// generator every backend uses, so operand statistics -- and therefore the
// accuracy rows -- mean the same thing across them.  Never raw random bits:
// NaN and denormal slow paths would understate the hardware.
float litertValueAt(int64_t i, int64_t j, uint32_t seed);

// The quantization scales the recipes use, shared with the accuracy test so
// the reference dequantizes with exactly the values the model stores.
float litertActScale(int bits);                       // activations in [-0.5, 0.5)
float litertOutScale(int64_t K, int bits);            // a K-deep dot product of two such
float litertWeightScale(int bits);                    // symmetric weights in [-0.5, 0.5)

// ---- recipes --------------------------------------------------------------

// The throughput matmul: A [1, M, K] constant scaled by the runtime scalar
// input `s` (so nothing is a constant expression), W [N, K] in the plan's
// weight format, and the [1, M, N] result reduced to one row with a maximum
// so nothing large leaves the device.  The reduction is a maximum rather
// than a sum on purpose: summing rows of A*W equals multiplying summed rows
// of A, a rewrite an optimiser is free to make.
clpeak_tflite::TfliteBytes litertMatMulModel(const LitertPlan &p, int64_t M, int64_t K, int64_t N,
                                             uint32_t seedA = 0x243f6a88u, uint32_t seedW = 0x85a308d3u);

// The accuracy matmul: x [1, M, K] is a model input and y [1, M, N] the
// whole result, so the host can hand over exact values and read back
// exactly what the accelerator computed.  `weights` receives what the model
// stores, dequantized -- the values a reference multiplies.
clpeak_tflite::TfliteBytes litertPlainMatMulModel(const LitertPlan &p, int64_t M, int64_t K, int64_t N,
                                                  std::vector<float> *weights,
                                                  uint32_t seedW = 0x85a308d3u);

// y [1, 1, N] = x [1, 1, K] (input) * W [N, K]: the streaming shape.
clpeak_tflite::TfliteBytes litertGemvModel(const LitertPlan &p, int64_t K, int64_t N, uint32_t seedW);

// The between-the-matmuls operations, over a resident [1, rows, cols]
// tensor scaled by `s`; None is the reference graph that only scales and
// reduces.
enum class LitertActivation { None, Silu, Softmax, LayerNorm };
clpeak_tflite::TfliteBytes litertActivationModel(const LitertPlan &p, int64_t rows, int64_t cols,
                                                 LitertActivation act);

// The transfer graphs: X [1, elems] as a model input.  ToDevice gathers one
// element back; RoundTrip returns X squared in full.
enum class LitertTransfer { ToDevice, RoundTrip };
clpeak_tflite::TfliteBytes litertTransferModel(const LitertPlan &p, LitertTransfer dir, int64_t elems);

// The smallest graph worth expressing: Y = X * K over [1, width].
clpeak_tflite::TfliteBytes litertTrivialModel(const LitertPlan &p, int64_t width);

// 2-D convolution over a resident [1, spatial, spatial, channels] feature
// map scaled by `s`, with a [channels, k, k, channels/group] filter in the
// plan's weight format (depthwise when group == channels), reduced over the
// map to [1, 1, 1, channels].
clpeak_tflite::TfliteBytes litertConvModel(const LitertPlan &p, int64_t channels, int64_t spatial,
                                           int64_t kernel, bool depthwise);

// One transformer decoder block: `seq` tokens in one pass (prefill) when
// kvLen is 0, else one token against a resident cache of kvLen entries
// (decode).  The seven projections are in the plan's weight format; the
// attention, softmax, SwiGLU and residuals stay in the plan's activation
// type.  `int8Kv` stores the decode cache as int8 dequantized on the way in.
struct LitertBlockShape
{
  int64_t dModel = 2048;
  int64_t heads = 16;
  int64_t headDim = 128;
  int64_t ffnHidden = 5504;
  int64_t seq = 512;
  int64_t kvLen = 0;
  bool int8Kv = false;
};
clpeak_tflite::TfliteBytes litertBlockModel(const LitertPlan &p, const LitertBlockShape &sh);

// Bytes of one [N, K] projection matrix in the plan's weight format,
// including blockwise scales.
uint64_t litertWeightBytes(const LitertPlan &p, int64_t N, int64_t K);

// Element bytes for a tensor type.
size_t litertElemBytes(clpeak_tflite::TfType t, int64_t count);

// One value of the activation type from a float, for scalar inputs.
std::string litertScalarBytes(clpeak_tflite::TfType t, float v);

#endif // CLPEAK_LITERT_MODEL_H
