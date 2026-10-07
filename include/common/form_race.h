#ifndef CLPEAK_FORM_RACE_H
#define CLPEAK_FORM_RACE_H

// A race between two spellings of the same work: the same arithmetic built
// two ways, one of which a runtime runs faster on a given unit -- by an
// amount, and in a direction, that no rule written down would carry to the
// next chip.  A test times both, point by point, and reports the faster,
// saying which.  Core ML races the two orders a weight can be stored in
// (src/coreml/coreml_bench.h), and the Vulkan, OpenCL and oneAPI compute
// peaks the two sub-group widths their affine chain can be built at, across
// the vector widths (src/vulkan/compute_kernel.cpp, src/opencl/compute_test.cpp,
// src/oneapi/compute_float.cpp); LiteRT's int8 layers, with two choices to
// make, race as a MultiFormRace (below).
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

#include <algorithm>
#include <cstddef>
#include <vector>

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

  // Close only on a clear loser: drop a form that trails the other by
  // kFormRaceBehind and leave a tie open.  For a race whose margin moves from
  // one point to the next, where a tie at one says nothing about the next --
  // the compute peaks' sub-group widths, which a register allocator lays out
  // afresh at every vector width.  On an Arc A380 the mixed-precision affine
  // chain's two forms tied at width 4, and the one a tie kept there read
  // 16-28% under the other at widths 8 and 16.
  void dropTrailing(const double rate[2])
  {
    if (!open_[0] || !open_[1] || rate[0] <= 0.0 || rate[1] <= 0.0)
      return;
    const bool faster = rate[1] > rate[0];
    if (rate[faster] >= rate[!faster] * kFormRaceBehind)
      open_[!faster] = false;
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

// The same race over any number of forms, for work that has more than one
// choice to make -- LiteRT's int8 layers on a GPU are an operator and a
// kernel policy, four forms (src/litert/gemm.cpp).  Not two FormRaces, one
// per choice: settling each choice on its best reading picks the wrong pair
// when the choices interact, which is what a policy that changes the kernel
// an operator lowers to does.  The rule is FormRace::settle's over every form
// a point timed: a form trailing the fastest by kFormRaceBehind drops out,
// the forms within kFormRaceTie of the fastest are a tie that keeps one --
// the first listed, unless a later one built kFormRaceBuildGap times faster
// -- and a form between the two runs again at the next size.  For two forms
// it closes exactly what FormRace::settle does.
class MultiFormRace
{
public:
  explicit MultiFormRace(size_t forms) : open_(forms, true) {}

  size_t size() const { return open_.size(); }
  bool runs(size_t form) const { return open_[form]; }
  bool done() const
  {
    for (bool o : open_)
      if (o)
        return false;
    return true;
  }
  void drop(size_t form) { open_[form] = false; }

  // Close what a point settles: `rate[i]` is higher for faster and zero for a
  // form the point did not time, `buildUs[i]` what each took to build.
  void settle(const std::vector<double> &rate, const std::vector<double> &buildUs)
  {
    size_t timed = 0, lead = 0;
    for (size_t i = 0; i < open_.size(); i++)
      if (open_[i] && rate[i] > 0.0 && (timed++ == 0 || rate[i] > rate[lead]))
        lead = i;
    if (timed < 2)
      return;
    size_t keep = open_.size(), tied = 0;
    for (size_t i = 0; i < open_.size(); i++)
    {
      if (!open_[i] || rate[i] <= 0.0)
        continue;
      if (rate[lead] >= rate[i] * kFormRaceBehind)
        open_[i] = false;
      else if (rate[lead] <= rate[i] * kFormRaceTie)
      {
        tied++;
        if (keep == open_.size() || buildUs[keep] >= buildUs[i] * kFormRaceBuildGap)
          keep = i;
      }
    }
    if (tied < 2)
      return;
    for (size_t i = 0; i < open_.size(); i++)
      if (open_[i] && rate[i] > 0.0 && i != keep && rate[lead] <= rate[i] * kFormRaceTie)
        open_[i] = false;
  }

  // Whether `form` won a tie on its build time: it read within kFormRaceTie
  // of a form listed before it that took kFormRaceBuildGap times as long to
  // build.  What a row says so its form is not read as kept by mistake.
  static bool tieOnBuild(const std::vector<double> &rate, const std::vector<double> &buildUs, size_t form)
  {
    for (size_t i = 0; i < form; i++)
      if (rate[i] > 0.0 && std::max(rate[i], rate[form]) <= std::min(rate[i], rate[form]) * kFormRaceTie &&
          buildUs[i] >= buildUs[form] * kFormRaceBuildGap)
        return true;
    return false;
  }

private:
  std::vector<bool> open_;
};

} // namespace clpeak

#endif // CLPEAK_FORM_RACE_H
