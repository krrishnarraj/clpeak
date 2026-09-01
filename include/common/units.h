#ifndef CLPEAK_UNITS_H
#define CLPEAK_UNITS_H

#include <string>
#include <common/benchmark_enums.h>

// ── Units, quantities, and which way is "better" ───────────────────────────
//
// Backends pass SI base tokens -- "flops", "bps", "s" -- and the logger
// resolves each one here, exactly once, into everything a consumer needs:
//
//   symbol     what to print         "FLOPS", "B/s", "s"
//   quantity   what is measured      Flops, BytesPerSecond, Seconds, …
//   direction  which way is better   higher for throughput, lower for latency
//
// Values are already SI (FLOP/s, byte/s, s, 1/s, ppm) and the presenters
// pick `G/T/P` or `µ/n` via the SI ladder instead of a per-test magnitude.

// What a reading measures.  The SI base unit for each is what `scale`
// normalizes to.
enum class Quantity {
    Flops,           // FLOP/s
    Ops,             // OP/s (integer)
    BytesPerSecond,  // byte/s
    Seconds,         // second
    ItemsPerSecond,  // 1/s (texels, samples, tokens, …)
    Ratio,           // dimensionless
    Count,           // dimensionless whole number
    Unknown
};

// Which direction of change is an improvement.  `FromUnit` is the authoring
// default: it means "take the unit table's answer", and never appears in a
// resolved TestResult or in the dump.
enum class Direction {
    HigherIsBetter,
    LowerIsBetter,
    FromUnit
};

// Canonical lower-snake names used in the dump format.
const char *quantityString(Quantity q);
Quantity    quantityFromString(const std::string &s);

// `directionString` never emits "from_unit" -- FromUnit resolves to
// higher_is_better, since an unresolved direction is a bug and higher is the
// common case.
const char *directionString(Direction d);
Direction   directionFromString(const std::string &s);

struct UnitInfo {
    std::string symbol;     // display form, ready to print: "FLOPS", "B/s"
    Quantity    quantity = Quantity::Unknown;
    Direction   direction = Direction::HigherIsBetter;
};

// Resolve a unit token as backends write it ("flops", "s", …).  An unknown
// token is passed through unchanged as its own symbol, with quantity Unknown
// and higher-is-better -- a new unit therefore shows up correctly in the
// output before anyone gets round to adding it here.
UnitInfo unitInfo(const std::string &token);

// ── Magnitude-scaled display ────────────────────────────────────────────────
//
// Whether a quantity has an SI ladder to slide along.  A ratio or a bare
// count does not: "4000 ppm" is not "4 kppm", and an unknown quantity has no
// described magnitude to rescale by.
bool quantityIsScalable(Quantity q);

// A reading's value and unit as they should be printed together.
struct ScaledValue {
    std::string text;   // the mantissa, decimals chosen for its width
    std::string unit;   // the symbol with the prefix that mantissa wants
};

// Auto-scaled display form of a reading: 4.476e12 FLOP/s -> ("4.48", "TFLOPS"),
// 1.5e-7 s -> ("150", "s").  This is the ladder the GUI's value column
// slides along (formatValue in app/lib/src/model/result_model.dart) — the
// two are kept identical so a reading prints the same in either presenter.
// Quantities without a ladder, a zero, and a non-finite value print as
// measured, in the authored unit.
ScaledValue formatScaledValue(double value, const UnitInfo &unit);

// Derive a test's category from its unit when the author did not name one.
// Units that several categories share (gbps is Bandwidth *and* the crypto and
// string tests; ns is Latency) cannot be resolved here -- those call sites
// pass Category explicitly.
Category categoryFromUnit(const std::string &unit);

#endif  // CLPEAK_UNITS_H
