#ifndef CLPEAK_FORM_RACE_H
#define CLPEAK_FORM_RACE_H

// A race between two spellings of the same work: the same arithmetic built
// two ways, one of which a runtime runs faster on a given unit -- by an
// amount, and in a direction, that no rule written down would carry to the
// next chip.  A test times both, point by point, and reports the faster,
// saying which.  Core ML races the two orders a weight can be stored in
// (src/coreml/coreml_bench.h), LiteRT the two operators an int8 layer can be
// written as (src/litert/gemm.cpp).
//
// Each point a race stays open costs a second build and a second
// measurement, so a race closes as soon as its readings allow: once one form
// trails by half again (kFormRaceBehind) -- a slow path, which a larger size
// does not rescue -- or once the two read within kFormRaceTie of each other,
// where the spelling makes no difference.  Then the first form carries on,
// unless the second built several times faster (kFormRaceBuildGap): a tie
// names the same form from one run to the next instead of whichever noise
// favoured.  Only a race with a margin between the two is run again at the
// next size.

namespace clpeak
{

constexpr double kFormRaceTie = 1.03;
constexpr double kFormRaceBehind = 1.5;
constexpr double kFormRaceBuildGap = 2.0;

// The two forms are `false` and `true`: the first spelling and the second.
class FormRace
{
public:
  // Whether the next point times the form `second` names.
  bool runs(bool second) const { return open_[second]; }
  bool done() const { return !open_[0] && !open_[1]; }

  // A form that cannot run a point drops out: a larger size needs strictly
  // more of everything.
  void drop(bool second) { open_[second] = false; }

  // The form to take from a point at which both ran: `rate` is higher for
  // faster, `buildUs` what each took to build.
  static bool pick(const double rate[2], const double buildUs[2])
  {
    const bool faster = rate[1] > rate[0];
    if (rate[faster] <= rate[!faster] * kFormRaceTie)
      return buildUs[0] >= buildUs[1] * kFormRaceBuildGap;
    return faster;
  }

  // Whether two readings tie, and the build times decided which went on.
  static bool tieOnBuild(const double rate[2], const double buildUs[2])
  {
    const bool faster = rate[1] > rate[0];
    return rate[faster] <= rate[!faster] * kFormRaceTie &&
           (buildUs[0] >= buildUs[1] * kFormRaceBuildGap ||
            buildUs[1] >= buildUs[0] * kFormRaceBuildGap);
  }

  // Close the race if this point, at which both ran, settles it.
  void settle(const double rate[2], const double buildUs[2])
  {
    if (!open_[0] || !open_[1] || rate[0] <= 0.0 || rate[1] <= 0.0)
      return;
    const bool faster = rate[1] > rate[0];
    if (rate[faster] >= rate[!faster] * kFormRaceBehind)
      open_[!faster] = false;
    else if (rate[faster] <= rate[!faster] * kFormRaceTie)
    {
      const bool keep = pick(rate, buildUs);
      open_[!keep] = false;
      sameProgram_ = buildUs[!keep] >= buildUs[keep] * kFormRaceBuildGap;
    }
  }

  // The race for the same work in another shape -- a transformer block's
  // decode after its prefill: a fresh one, unless this one closed on a tie
  // that the build times split.  That is the compiler rewriting one form
  // into the other, the same program, which no shape changes.
  FormRace nextShape() const { return sameProgram_ ? *this : FormRace(); }

private:
  bool open_[2] = {true, true};
  bool sameProgram_ = false;
};

} // namespace clpeak

#endif // CLPEAK_FORM_RACE_H
