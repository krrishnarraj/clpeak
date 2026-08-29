#include <common/units.h>

// ── The unit table ─────────────────────────────────────────────────────────
//
// Keyed by the tokens backends already pass, so adding quantity/scale/
// direction to the model cost no backend churn.  A token missing from here is
// not an error: unitInfo() passes it through as its own symbol (see below).

namespace {

struct UnitRow {
    const char *token;
    const char *symbol;
    Quantity    quantity;
    double      scale;      // value * scale -> SI base unit
    Direction   direction;
};

// Latency and error units are the lower-is-better ones.  `ppm` is the ONNX
// numeric-error unit: the reading is a distance from the right answer, so
// less of it is better even though it is not a time.
const UnitRow kUnits[] = {
    { "gflops",  "GFLOPS",    Quantity::Flops,          1e9,  Direction::HigherIsBetter },
    { "tflops",  "TFLOPS",    Quantity::Flops,          1e12, Direction::HigherIsBetter },
    { "gops",    "GOPS",      Quantity::Ops,            1e9,  Direction::HigherIsBetter },
    { "tops",    "TOPS",      Quantity::Ops,            1e12, Direction::HigherIsBetter },
    { "gbps",    "GB/s",      Quantity::BytesPerSecond, 1e9,  Direction::HigherIsBetter },
    { "gtexels", "GTexel/s",  Quantity::ItemsPerSecond, 1e9,  Direction::HigherIsBetter },
    { "us",      "µs",   Quantity::Seconds,        1e-6, Direction::LowerIsBetter  },
    { "ns",      "ns",        Quantity::Seconds,        1e-9, Direction::LowerIsBetter  },
    { "ppm",     "ppm",       Quantity::Ratio,          1e-6, Direction::LowerIsBetter  },
};

} // namespace

// ── Enum <-> string ────────────────────────────────────────────────────────

const char *quantityString(Quantity q)
{
    switch (q)
    {
    case Quantity::Flops:          return "flops";
    case Quantity::Ops:            return "ops";
    case Quantity::BytesPerSecond: return "bytes_per_second";
    case Quantity::Seconds:        return "seconds";
    case Quantity::ItemsPerSecond: return "items_per_second";
    case Quantity::Ratio:          return "ratio";
    case Quantity::Count:          return "count";
    case Quantity::Unknown:        return "unknown";
    }
    return "unknown";
}

Quantity quantityFromString(const std::string &s)
{
    if (s == "flops")            return Quantity::Flops;
    if (s == "ops")              return Quantity::Ops;
    if (s == "bytes_per_second") return Quantity::BytesPerSecond;
    if (s == "seconds")          return Quantity::Seconds;
    if (s == "items_per_second") return Quantity::ItemsPerSecond;
    if (s == "ratio")            return Quantity::Ratio;
    if (s == "count")            return Quantity::Count;
    return Quantity::Unknown;
}

const char *directionString(Direction d)
{
    // FromUnit is an authoring default, not a value: it is resolved against
    // the table before anything is stored, so seeing it here means a reading
    // escaped resolution.  Report the common case rather than inventing a
    // third string the loaders would have to understand.
    return (d == Direction::LowerIsBetter) ? "lower_is_better"
                                           : "higher_is_better";
}

Direction directionFromString(const std::string &s)
{
    if (s == "lower_is_better") return Direction::LowerIsBetter;
    return Direction::HigherIsBetter;
}

// ── Resolution ─────────────────────────────────────────────────────────────

UnitInfo unitInfo(const std::string &token)
{
    for (const UnitRow &r : kUnits)
    {
        if (token != r.token) continue;
        UnitInfo info;
        info.symbol    = r.symbol;
        info.quantity  = r.quantity;
        info.scale     = r.scale;
        info.direction = r.direction;
        return info;
    }

    // Unknown token: show it as written rather than dropping it.  scale 1 and
    // quantity Unknown tell consumers not to rescale, which is the only safe
    // thing to do with a magnitude nobody has described.
    UnitInfo info;
    info.symbol    = token;
    info.quantity  = Quantity::Unknown;
    info.scale     = 1.0;
    info.direction = Direction::HigherIsBetter;
    return info;
}

Category categoryFromUnit(const std::string &unit)
{
    if (unit == "gflops" || unit == "tflops") return Category::FpCompute;
    if (unit == "gops"   || unit == "tops")   return Category::IntCompute;
    if (unit == "gbps")                       return Category::Bandwidth;
    if (unit == "us")                         return Category::Latency;
    return Category::Unknown;
}
