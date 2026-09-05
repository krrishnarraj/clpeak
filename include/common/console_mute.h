#ifndef CLPEAK_CONSOLE_MUTE_H
#define CLPEAK_CONSOLE_MUTE_H

// Silence stdout+stderr at the file-descriptor level for a scope.
//
// Vendor runtimes print diagnostics straight to the console, below any log
// level or callback clpeak can set: hipBLASLt's Tensile/rocRoller internals
// walk instruction tables out loud, and loading an ONNX Runtime execution
// provider can emit hundreds of "Schema error: ... already registered" lines
// from the bundled ONNX library.  None of it is actionable -- the returned
// status already says whether the call worked -- and all of it wrecks a
// results table.
//
// Under --verbose the mute is a no-op, so the library's own diagnostics stay
// visible when someone is actually debugging.  Suppression that cannot be
// switched off would hide the one message that mattered.

namespace clpeak {

class ScopedConsoleMute
{
public:
  ScopedConsoleMute();
  ~ScopedConsoleMute();

  ScopedConsoleMute(const ScopedConsoleMute &) = delete;
  ScopedConsoleMute &operator=(const ScopedConsoleMute &) = delete;

private:
  int savedOut = -1;
  int savedErr = -1;
};

} // namespace clpeak

#endif // CLPEAK_CONSOLE_MUTE_H
