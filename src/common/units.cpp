#include <common/units.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <utility>

// ── The unit table ─────────────────────────────────────────────────────────
//
// Keyed by the tokens backends already pass, so adding quantity/
// direction to the model cost no backend churn.  A token missing from here is
// not an error: unitInfo() passes it through as its own symbol (see below).

namespace
{

    struct UnitRow
    {
        const char *token;
        const char *symbol;
        Quantity quantity;
        Direction direction;
    };

    // All values are already SI: FLOP/s, OP/s, byte/s, 1/s, s, ppm.
    const UnitRow kUnits[] = {
        {"flops", "FLOPS", Quantity::Flops, Direction::HigherIsBetter},
        {"ops", "OPS", Quantity::Ops, Direction::HigherIsBetter},
        {"bps", "B/s", Quantity::BytesPerSecond, Direction::HigherIsBetter},
        {"texels", "Texel/s", Quantity::ItemsPerSecond, Direction::HigherIsBetter},
        {"s", "s", Quantity::Seconds, Direction::LowerIsBetter},
        {"ppm", "ppm", Quantity::Ratio, Direction::LowerIsBetter},
    };

} // namespace

// ── Enum <-> string ────────────────────────────────────────────────────────

const char *quantityString(Quantity q)
{
    switch (q)
    {
    case Quantity::Flops:
        return "flops";
    case Quantity::Ops:
        return "ops";
    case Quantity::BytesPerSecond:
        return "bytes_per_second";
    case Quantity::Seconds:
        return "seconds";
    case Quantity::ItemsPerSecond:
        return "items_per_second";
    case Quantity::Ratio:
        return "ratio";
    case Quantity::Count:
        return "count";
    case Quantity::Unknown:
        return "unknown";
    }
    return "unknown";
}

Quantity quantityFromString(const std::string &s)
{
    if (s == "flops")
        return Quantity::Flops;
    if (s == "ops")
        return Quantity::Ops;
    if (s == "bytes_per_second")
        return Quantity::BytesPerSecond;
    if (s == "seconds")
        return Quantity::Seconds;
    if (s == "items_per_second")
        return Quantity::ItemsPerSecond;
    if (s == "ratio")
        return Quantity::Ratio;
    if (s == "count")
        return Quantity::Count;
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
    if (s == "lower_is_better")
        return Direction::LowerIsBetter;
    return Direction::HigherIsBetter;
}

// ── Resolution ─────────────────────────────────────────────────────────────

UnitInfo unitInfo(const std::string &token)
{
    for (const UnitRow &r : kUnits)
    {
        if (token != r.token)
            continue;
        UnitInfo info;
        info.symbol = r.symbol;
        info.quantity = r.quantity;
        info.direction = r.direction;
        return info;
    }

    // Unknown token: show it as written rather than dropping it.
    // Quantity Unknown tells consumers not to rescale, which is the only safe
    // thing to do with a magnitude nobody has described.
    UnitInfo info;
    info.symbol = token;
    info.quantity = Quantity::Unknown;
    info.direction = Direction::HigherIsBetter;
    return info;
}

Category categoryFromUnit(const std::string &unit)
{
    if (unit == "flops")
        return Category::Compute;
    if (unit == "ops")
        return Category::Compute;
    if (unit == "bps")
        return Category::Bandwidth;
    if (unit == "texels")
        return Category::Bandwidth;
    if (unit == "s")
        return Category::Latency;
    return Category::Unknown;
}

// ── Magnitude-scaled display ────────────────────────────────────────────────
//
// A unit symbol is <SI prefix><base>, so rescaling is a matter of swapping
// the prefix — which works for FLOPS → GFLOPS → TFLOPS, B/s → GB/s → TB/s,
// Texel/s → GTexel/s and s → ms/µs/ns alike, without a table of every unit
// clpeak might one day report.  Byte-wise prefix matching: every prefix is
// one logical character, "µ" included (two UTF-8 bytes).

namespace
{

    const struct
    {
        int exp;
        const char *prefix;
    } kSiPrefixes[] = {
        {-9, "n"},
        {-6, "u"},
        {-3, "m"},
        {0, ""},
        {3, "K"},
        {6, "M"},
        {9, "G"},
        {12, "T"},
        {15, "P"},
        {18, "E"},
    };

    // Strip a leading SI prefix off a unit symbol: "TFLOPS" -> (12, "FLOPS"),
    // "GB/s" -> (9, "B/s") , "s" -> (-9, "s").  Known bases are FLOPS/OPS/B/s/s/Texel/s
    // so "Texel/s" is not mistaken for T + "exel/s".
    std::pair<int, std::string> splitPrefix(const std::string &symbol)
    {
        for (const auto &p : kSiPrefixes)
        {
            const size_t len = std::strlen(p.prefix); // bytes; "µ" is two
            if (len == 0)
                continue;
            if (symbol.size() > len && symbol.compare(0, len, p.prefix) == 0)
            {
                std::string base = symbol.substr(len);
                if (base == "FLOPS" || base == "OPS" || base == "B/s" ||
                    base == "s" || base == "Texel/s")
                    return {p.exp, base};
            }
        }
        return {0, symbol};
    }

    // Decimals by magnitude: two while the mantissa is single-digit, one into
    // the tens, none past a hundred — the column stays narrow and "4476.00"
    // was noise.
    std::string fmtMantissa(double v)
    {
        const double a = std::fabs(v);
        const int prec = (a >= 100.0) ? 0 : (a >= 10.0) ? 1
                                                        : 2;
        char buf[40];
        std::snprintf(buf, sizeof buf, "%.*f", prec, v);
        return buf;
    }

} // namespace

bool quantityIsScalable(Quantity q)
{
    switch (q)
    {
    case Quantity::Ratio:
    case Quantity::Count:
    case Quantity::Unknown:
        return false;
    default:
        return true;
    }
}

ScaledValue formatScaledValue(double value, const UnitInfo &unit)
{
    // Nothing to slide along: a ratio is a ratio, a unit nobody has described
    // is printed exactly as measured rather than guessed at, and zero has no
    // magnitude to pick a prefix from.
    if (!quantityIsScalable(unit.quantity) || value == 0.0 ||
        !std::isfinite(value))
        return {fmtMantissa(value), unit.symbol};

    // Normalize to the quantity's SI base unit, then pick the largest ladder
    // step at or below the value's own magnitude, so the mantissa lands in
    // [1, 1000).
    const std::pair<int, std::string> split = splitPrefix(unit.symbol);
    const double si = value * std::pow(10.0, split.first);

    int exp = static_cast<int>(std::floor(std::log10(std::fabs(si)) / 3.0) * 3.0);
    exp = std::max(-9, std::min(15, exp));

    const double mantissa = si / std::pow(10.0, exp);

    std::string prefix;
    for (const auto &p : kSiPrefixes)
        if (p.exp == exp)
            prefix = p.prefix;

    return {fmtMantissa(mantissa), prefix + split.second};
}
