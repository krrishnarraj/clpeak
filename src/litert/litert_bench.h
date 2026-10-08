#ifndef CLPEAK_LITERT_BENCH_H
#define CLPEAK_LITERT_BENCH_H

// The measurement shape every timed test in this backend follows -- the one
// the GPU backends use and the ONNX backend documents (src/onnx/AGENTS.md,
// "Three phases per rung"):
//
//   1. warmup   1 + warmupCount untimed inferences.  The extra one is not a
//               spare: XNNPACK packs weights and the GPU accelerator finishes
//               compiling and uploading on the first run, so run one is a
//               different event from run two.
//   2. probe    exactly one timed inference, to size the batch.
//   3. timed    pickIters(probe, budget, forced, kLitertMaxIters) inferences;
//               when the budget affords only one, the probe already is the
//               measurement.
//
// Every batch, the probe included, ends with a wait for the accelerator: the
// GPU accelerator's OpenCL path returns from a run as soon as the work is
// submitted, and a probe that timed a submission would size the batch for
// a run a hundred times shorter than the real one -- on a phone that turned
// a one-second budget into ten.  `syncEach` waits after every run instead,
// for the latency rows, where a result read back is the unit of work.

#include <litert/litert_peak.h>
#include "litert_model.h"
#include "litert_session.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// What one session of a graph over resident constants holds at once, for a
// test's memory gate: three copies of the constants in their stored types --
// the model's bytes, which a session keeps because LiteRT runs a model in
// place; the accelerator's own (XNNPACK's packed weights, the GPU's buffers,
// on a phone the same memory); and the host-side graph the GPU accelerator
// compiles from, which leaves XNNPACK a copy of headroom -- and every
// intermediate tensor, since the accelerators keep each one rather than
// reusing two.  On an M1 Pro, the gemm test's 4096-wide fp32 chain grew
// XNNPACK's process by the model, its packed copy and fifteen 64 MB
// activations, and the Metal accelerator's by more; int8_weight's float
// activations come to four times its weights.  Counted as two copies and two
// activations, that rung was let onto a 15.8 GB phone, which killed the
// process.
inline uint64_t litertHeldBytes(uint64_t constants, uint64_t activations)
{
  return 3 * constants + activations;
}

// A kernel that declines a type at prepare time ("input->type !=
// kTfLiteFloat32", "failed to prepare") is the runtime saying it has no
// kernel for the format, which is a capability and not a failure.
inline ResultStatus litertFailureStatus(const std::string &error)
{
  if (error.find("failed to prepare") != std::string::npos ||
      error.find("not supported") != std::string::npos ||
      error.find("Unsupported") != std::string::npos ||
      error.find("unsupported") != std::string::npos)
    return ResultStatus::Unsupported;
  return ResultStatus::Error;
}

struct LitertMeasurement
{
  double meanUs = -1.0;     // mean per inference; negative on failure
  double probeUs = -1.0;    // the single calibration inference
  unsigned iters = 0;
  std::string error;
  ResultStatus status = ResultStatus::Ok;
};

inline LitertMeasurement litertMeasure(LitertSession &s, unsigned warmupCount, unsigned budgetUs,
                                       bool forceIters, unsigned forced,
                                       unsigned maxIters = kLitertMaxIters, bool syncEach = false)
{
  LitertMeasurement m;
  if (s.timeRuns(1 + warmupCount, m.error, syncEach) < 0.0)
  {
    if (m.error.empty())
      m.error = "inference failed";
    m.status = litertFailureStatus(m.error);
    return m;
  }
  m.probeUs = s.timeRuns(1, m.error, syncEach);
  if (m.probeUs <= 0.0)
  {
    m.status = ResultStatus::Error;
    if (m.error.empty())
      m.error = "inference failed";
    return m;
  }
  m.iters = pickIters(m.probeUs, budgetUs, forceIters ? forced : 0, maxIters);
  m.meanUs = (m.iters > 1) ? s.timeRuns(m.iters, m.error, syncEach) : m.probeUs;
  if (m.meanUs <= 0.0)
  {
    m.status = ResultStatus::Error;
    if (m.error.empty())
      m.error = "inference failed";
  }
  return m;
}

// Write the runtime scalar that keeps a throughput graph live into input 0.
// A hair above one for the float types, so the scaled values stay inside
// the range the fills chose; exactly one for the integer types, where the
// scalar's own scale (1/127, 1/32767) puts one at the top code.
inline bool litertBindScalar(LitertSession &s, const LitertPlan &p, std::string &error)
{
  // Under floatIo an integer plan's scalar goes in as a float32 one and is
  // quantized at the same scale, so the same one lands on the top code.
  const bool integer = (p.act == clpeak_tflite::TfType::I8 || p.act == clpeak_tflite::TfType::I16);
  const std::string v = litertScalarBytes(litertIoType(p), integer ? 1.0f : 1.0009765625f);
  return s.writeInput(0, v.data(), v.size(), error);
}

// What the timed runs computed, read once they are over: a NaN or an
// infinity in the row the graph reduces to (output 0, of type `t`)
// withholds the timing as an error, the ONNX backend's rule
// (onnxNonFiniteReason, src/onnx/onnx_session.cpp).  Nothing in these
// graphs can overflow -- the gemm chains keep each layer's magnitude and the
// block's largest value sits 31x under fp16's -- so either one says the
// accelerator computed something other than the graph; the answer check
// (wrongAnswer) sees a different graph, the accuracy matmul, and cannot
// stand in.  Only the reduced row is read, so a wrong but finite answer
// passes: a tripwire, not an accuracy test.  Integer rows hold no NaN and
// are not read.  Empty when the row is finite or cannot be read.
inline std::string litertNonFiniteReason(LitertSession &s, clpeak_tflite::TfType t,
                                         const std::string &where)
{
  using clpeak_tflite::TfType;
  if (t != TfType::F32 && t != TfType::F16 && t != TfType::BF16)
    return std::string();
  std::vector<uint8_t> raw;
  std::string err;
  const size_t es = (t == TfType::F32) ? 4 : 2;
  if (!s.outputBytes(0, raw, err) || raw.empty() || raw.size() % es != 0)
    return std::string();
  // Every exponent bit set is an infinity or a NaN; read as bits, so no
  // floating-point mode can compile the test away.
  const uint16_t exp16 = (t == TfType::F16) ? 0x7c00u : 0x7f80u;
  const size_t n = raw.size() / es;
  size_t bad = 0;
  for (size_t i = 0; i < n; i++)
  {
    if (es == 4)
    {
      uint32_t x;
      std::memcpy(&x, &raw[4 * i], 4);
      bad += (x & 0x7f800000u) == 0x7f800000u;
    }
    else
    {
      uint16_t h;
      std::memcpy(&h, &raw[2 * i], 2);
      bad += (h & exp16) == exp16;
    }
  }
  if (bad == 0)
    return std::string();
  const std::string count = std::to_string(n);
  return "the accelerator returned NaN or infinity " + where + " (" +
         (bad == n ? "all " + count : std::to_string(bad) + " of the " + count) +
         " values of the row the graph reduces to), and every value in this graph stays far "
         "inside its type's range: it computed something other than this graph, so its timing "
         "is withheld";
}

// Whether a profiled tag names a kernel that multiplies -- an operator or
// shader with "fully", "conv", "matmul" or "gemm" in its name -- and not one
// that converts weights for it: with constant sharing on, the GPU
// accelerator runs a `weights_convert_uint8_to_float16` per layer ahead of
// the layers, as many as they are, and "conv" is in "convert".
inline bool litertKernelMultiplies(const std::string &tag)
{
  std::string l = tag;
  std::transform(l.begin(), l.end(), l.begin(), ::tolower);
  if (l.find("convert") != std::string::npos)
    return false;
  return l.find("fully") != std::string::npos || l.find("conv") != std::string::npos ||
         l.find("matmul") != std::string::npos || l.find("gemm") != std::string::npos;
}

// The kernel that did a graph's multiplies, from a profiled run: the one
// most of them ran as.  A chain's 64-deep seed widening can be given another
// (the Metal accelerator runs it as conv_generic where the layers run
// conv_wave_matrix), and so can a block's attention; and a convolution's
// need not say "conv" -- XNNPACK runs a 1x1 one as its Fully Connected GEMM.
inline std::string litertMatMulKernel(const std::vector<std::string> &ops)
{
  std::map<std::string, int> count;
  std::string best;
  for (const std::string &o : ops)
    if (litertKernelMultiplies(o) && ++count[o] > (best.empty() ? 0 : count[best]))
      best = o;
  return best;
}

// What a profiled kernel tag says its multiplies were, if anything.
// XNNPACK names an operator by the types it packed, its input's first after
// the layout: `Fully Connected (NC, QD8, F32, QC8W)` multiplies int8
// activations it quantizes as it goes, where `(NC, F32, QC8W)` would unpack
// int8 weights into float multiplies -- integer weights alone say nothing.
// The GPU accelerator computes in float unless a kernel's own name, the part
// before the first arrow, says int8 or int4: Mali's `convolution_int8(
// conv_generic) -> dequantize_to_float16` quantizes its source as it goes
// (what allow_src_quantized_fc_conv_ops is documented to allow), while
// Metal's `convolution1x1(conv_wave_matrix) -> quantize_and_dequantize` is a
// float kernel between quantize and dequantize passes, so a tail proves
// nothing.  Elsewhere only such a pass says float: the interpreter's own
// reference kernels are plain op names ("FULLY_CONNECTED").
enum class LitertArithmetic
{
  Unknown,
  Float,
  Integer,
};

inline LitertArithmetic litertKernelArithmetic(const std::string &kernel, LitertAccel accel)
{
  if (kernel.empty())
    return LitertArithmetic::Unknown;
  std::string l = kernel;
  std::transform(l.begin(), l.end(), l.begin(), ::tolower);
  const size_t arrow = l.find(" -> ");
  const std::string head = arrow == std::string::npos ? l : l.substr(0, arrow);
  // XNNPACK's "<operator> (<layout>, <input type>, ...)": QS8, QD8, QP8 and
  // the rest are integer, F32, F16, BF16 and PF32 float.
  const size_t open = head.find(" (");
  const size_t comma = open == std::string::npos ? std::string::npos : head.find(", ", open);
  if (comma != std::string::npos && comma > open + 2 && head.find_first_not_of("nchwd", open + 2) == comma)
    return head.compare(comma + 2, 1, "q") == 0 ? LitertArithmetic::Integer : LitertArithmetic::Float;
  for (const char *mark : {"int8", "int4", "int16", "quantized"})
    if (head.find(mark) != std::string::npos)
      return LitertArithmetic::Integer;
  if (accel == LitertAccel::Gpu || l.find("quantize") != std::string::npos)
    return LitertArithmetic::Float;
  return LitertArithmetic::Unknown;
}

// Whether a plan's weights are integer, which a kernel may multiply as
// integers or unpack to float: the formats whose kernel a row has to name.
inline bool litertIntegerWeights(const LitertPlan &p)
{
  using clpeak_tflite::TfType;
  return p.weight == TfType::I8 || p.weight == TfType::I4 || p.weight == TfType::I2 || p.weight == TfType::U8 ||
         p.weight == TfType::U4;
}

// The unit a rate row counts in.  A full-integer format
// (LitertPlan::integerOps) counts ops whatever ran it: its graph is integer
// arithmetic, which a GPU without integer kernels carries out in float
// between quantize and dequantize passes, and the row says so
// (litertKernelNote).  A weight-only format's graph is float arithmetic over
// compressed weights, counted in flops -- unless its kernel quantizes the
// activations and multiplies int8, as XNNPACK's and a GPU's 8-bit kernels
// do, and then it counts ops.
inline const char *litertRateUnit(const LitertPlan &p, const std::string &kernel, LitertAccel accel)
{
  if (p.integerOps)
    return "ops";
  return litertIntegerWeights(p) && litertKernelArithmetic(kernel, accel) == LitertArithmetic::Integer ? "ops"
                                                                                                      : "flops";
}

// The clause a row's kernel gets (appended to "..., as `kernel`") when it
// ran other arithmetic than its format's graph asks for.  A full-integer
// row on a float kernel -- one between quantize and dequantize passes -- is
// the accelerator's float rate with the format's traffic savings, still
// counted in ops; a weight-only row's int8 kernel quantizes the activations
// as it goes (litertRateUnit).
inline std::string litertKernelNote(const LitertPlan &p, const std::string &kernel, LitertAccel accel)
{
  const LitertArithmetic a = litertKernelArithmetic(kernel, accel);
  if (p.integerOps && a == LitertArithmetic::Float)
    return ", a float kernel";
  if (!p.integerOps && litertIntegerWeights(p) && a == LitertArithmetic::Integer)
    return ", which multiplies in int8";
  return std::string();
}

// The session config a plan asks for on a device, in one form (its GPU
// kernel policy; the rest of a form is the graph's, litertFormPlan).
inline LitertSessionConfig litertConfigFor(const LitertPlan &p, const LitertForm &form, bool profile = false)
{
  LitertSessionConfig cfg;
  cfg.gpuPrecision = p.gpuPrecision;
  cfg.gpuInt8Kernels = p.gpuInt8KernelChoice && form.gpuInt8Kernels;
  cfg.gpuShareConstants = cfg.gpuInt8Kernels && p.gpuShareConstants;
  cfg.profile = profile;
  return cfg;
}
inline LitertSessionConfig litertConfigFor(const LitertPlan &p, bool profile = false)
{
  return litertConfigFor(p, LitertForm(), profile);
}

#endif // CLPEAK_LITERT_BENCH_H
