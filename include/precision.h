#ifndef PRECISION_H
#define PRECISION_H

// Precision mode for floating-point compute tests.
// - Default: no fast-math compile flags. Whatever the toolchain emits at
//   default rounding/contraction settings.
// - Relaxed: aggressive fast-math (per backend: -cl-fast-relaxed-math,
//   --use_fast_math, MTLMathModeFast, SPIR-V FPFastMathDefault). No fp32->fp16
//   demotion (RelaxedPrecision/mediump) -- half-precision peak is reported by
//   the dedicated half tests.
//
// Every floating-point benchmark is run once per mode; the relaxed pass gets
// a " (relaxed math)" suffix on its display label and a "_relaxed" suffix on
// its result tag so JSON/CSV consumers can distinguish the two columns.
enum class PrecisionMode { Default, Relaxed };

inline const char *precisionLabelSuffix(PrecisionMode m)
{
  return m == PrecisionMode::Relaxed ? " (relaxed math)" : "";
}

inline const char *precisionResultSuffix(PrecisionMode m)
{
  return m == PrecisionMode::Relaxed ? "_relaxed" : "";
}

inline const char *precisionAttribute(PrecisionMode m)
{
  return m == PrecisionMode::Relaxed ? "relaxed" : "default";
}

#endif // PRECISION_H
