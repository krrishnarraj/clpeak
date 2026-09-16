#ifndef CLPEAK_CONSOLE_MUTE_H
#define CLPEAK_CONSOLE_MUTE_H

#include <atomic>
#include <string>
#include <thread>
#include <vector>

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
  // Verbose: capture under --verbose, discard otherwise (the default).
  // Always: capture regardless, and keep the text for text() -- for a
  // library whose refusal reason only ever reaches the console.  TFLite's
  // kernel errors ("input->type != kTfLiteFloat32") go to its error reporter,
  // which is stderr, and a LiteRT compile that fails hands back a status
  // code and nothing else; the sentence that explains the row is here.
  enum class Capture { Verbose, Always };
  explicit ScopedConsoleMute(Capture mode = Capture::Verbose);
  // `watch` makes the scope note whether the muted stream ever contained any
  // of these substrings, readable through sawWatched(), even when the text is
  // otherwise discarded (a non-verbose scope) -- so a caller can silence a
  // library and still learn that it printed, say, "Device is lost".  Watching
  // forces the capture path (a pipe and a reader thread) that plain discarding
  // does not need.
  //
  // `stderrOnly` leaves stdout alone and mutes only stderr.  A benchmark's
  // result table is written to stdout while the vendor runtime's noise goes to
  // stderr, so a scope that must span whole tests -- to catch a library's
  // asynchronous worker-thread logging, which no per-call scope can bracket --
  // silences that noise without swallowing the rows.
  ScopedConsoleMute(Capture mode, std::vector<std::string> watch,
                    bool stderrOnly = false);
  ~ScopedConsoleMute();

  // Restore the console now (the destructor would otherwise), so text() can
  // be read while the scope is still in reach.  Idempotent.
  void finish();

  // Everything the library printed within this scope, Capture::Always only.
  const std::string &text() const { return captured; }

  // Did the muted stream contain any of the `watch` substrings?  Thread-safe,
  // so it may be read while the scope is still open (between the calls it is
  // muting) to react as soon as a marker appears.
  bool sawWatched() const { return watchHit.load(std::memory_order_relaxed); }

  ScopedConsoleMute(const ScopedConsoleMute &) = delete;
  ScopedConsoleMute &operator=(const ScopedConsoleMute &) = delete;

  // Lines a single scope records before it starts counting instead: Tensile
  // alone prints thousands, and the first few already say what is running.
  static constexpr unsigned kMaxLines = 200;

private:
  void drain();
  void record(const std::string &line);

  int savedOut = -1;
  int savedErr = -1;

  // Capture only (--verbose).
  int         readFd = -1;
  std::thread reader;
  std::string pending;    // an unterminated last line, flushed at scope end
  std::string captured;   // Capture::Always: every line, for text()
  bool        keepText = false;
  bool        stderrOnly = false;  // mute stderr but leave stdout (the table)
  bool        finished = false;
  unsigned    lines = 0;

  // Watching: substrings to look for, and whether one was seen.  The reader
  // thread sets the flag; sawWatched() reads it, possibly from another thread.
  std::vector<std::string> watch;
  std::atomic<bool>        watchHit{false};
};

} // namespace clpeak

#endif // CLPEAK_CONSOLE_MUTE_H
