#ifndef CLPEAK_CONSOLE_MUTE_H
#define CLPEAK_CONSOLE_MUTE_H

#include <string>
#include <thread>

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
// Under --verbose the scope captures instead of discarding: both fds go to
// a pipe, and what the library printed is recorded on the run's log as
// `source: "console"` debug entries (the first kMaxLines of a scope; a
// count stands in for the rest) -- which is how those lines reach a file
// exported from a phone, where there is no console to read.  The CLI still
// sees them: the entries render to the real stderr, past the capture.
// Suppression that cannot be switched off would hide the one message that
// mattered.

namespace clpeak {

class ScopedConsoleMute
{
public:
  ScopedConsoleMute();
  ~ScopedConsoleMute();

  ScopedConsoleMute(const ScopedConsoleMute &) = delete;
  ScopedConsoleMute &operator=(const ScopedConsoleMute &) = delete;

  // Lines a single scope records before it starts counting instead: Tensile
  // alone prints thousands, and the first few already say what is running.
  static constexpr unsigned kMaxLines = 200;

private:
  void drain();

  int savedOut = -1;
  int savedErr = -1;

  // Capture only (--verbose).
  int         readFd = -1;
  std::thread reader;
  std::string pending;    // an unterminated last line, flushed at scope end
  unsigned    lines = 0;
};

} // namespace clpeak

#endif // CLPEAK_CONSOLE_MUTE_H
