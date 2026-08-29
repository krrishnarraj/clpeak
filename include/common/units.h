#ifndef CLPEAK_UNITS_H
#define CLPEAK_UNITS_H

#include <string>
#include <common/benchmark_enums.h>

// ── Units, quantities, and which way is "better" ───────────────────────────
//
// Backends keep passing the short lowercase tokens they always have --
// "gflops", "gbps", "us" -- and the logger resolves each one here, exactly
// once, into everything a consumer needs:
//
//   symbol     what to print         "GFLOPS", "GB/s", "µs"
//   quantity   what is measured      Flops, BytesPerSecond, Seconds, …
//   scale      value * scale = SI    1e9 for gflops, 1e-6 for us
//   direction  which way is better   higher for throughput, lower for latency
//
// `scale` is what makes readings comparable across tests and devices: clpeak
// reports GFLOPS in one test and TFLOPS in the next, so a raw value carries no
// meaning on its own.  Multiplying by `scale` normalizes every reading to the
// SI base unit for its quantity (FLOP/s, byte/s, second, 1/s), which is how
// the presenters pick an SI prefix instead of special-casing each token.

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
    std::string symbol;     // display form, ready to print: "GFLOPS", "GB/s"
    Quantity    quantity = Quantity::Unknown;
    double      scale    = 1.0;   // value * scale -> SI base unit
    Direction   direction = Direction::HigherIsBetter;
};

// Resolve a unit token as backends write it ("gflops", "us", …).  An unknown
// token is passed through unchanged as its own symbol, with quantity Unknown,
// scale 1 and higher-is-better -- a new unit therefore shows up correctly in
// the output before anyone gets round to adding it here.
UnitInfo unitInfo(const std::string &token);

// Derive a test's category from its unit when the author did not name one.
// Units that several categories share (gbps is Bandwidth *and* the crypto and
// string tests; ns is Latency) cannot be resolved here -- those call sites
// pass Category explicitly.
Category categoryFromUnit(const std::string &unit);

#endif  // CLPEAK_UNITS_H
