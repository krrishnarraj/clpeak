#include <common/console_mute.h>
#include <common/common.h>

#include <cstdio>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define XPUPEAK_DUP    _dup
#define XPUPEAK_DUP2   _dup2
#define XPUPEAK_CLOSE  _close
#define XPUPEAK_FILENO _fileno
#define XPUPEAK_READ   _read
#define XPUPEAK_DEVNULL "NUL"
#else
#include <unistd.h>
#define XPUPEAK_DUP    dup
#define XPUPEAK_DUP2   dup2
#define XPUPEAK_CLOSE  close
#define XPUPEAK_FILENO fileno
#define XPUPEAK_READ   read
#define XPUPEAK_DEVNULL "/dev/null"
#endif

namespace xpupeak {

namespace {

// A pipe whose write end will stand in for fds 1 and 2.  Returns false and
// leaves the fds untouched when the platform refuses one.
bool makePipe(int fds[2])
{
#ifdef _WIN32
  return _pipe(fds, 1 << 16, _O_BINARY) == 0;
#else
  return pipe(fds) == 0;
#endif
}

} // namespace

ScopedConsoleMute::ScopedConsoleMute(Capture mode)
    : ScopedConsoleMute(mode, {}, false)
{
}

ScopedConsoleMute::ScopedConsoleMute(Capture mode, std::vector<std::string> watchFor,
                                     bool stderrOnlyMode)
    : stderrOnly(stderrOnlyMode), watch(std::move(watchFor))
{
  (void)fflush(stdout);
  (void)fflush(stderr);
  savedOut = XPUPEAK_DUP(XPUPEAK_FILENO(stdout));
  savedErr = XPUPEAK_DUP(XPUPEAK_FILENO(stderr));
  keepText = (mode == Capture::Always);
  // Inside another scope, fd 2 is that scope's pipe and savedErr a copy of
  // it: a line rendered there would be recorded again as library output,
  // and echoed back into the pipe, up to kMaxLines times.  The terminal is
  // the enclosing scope's bypass.
  outerRealErr = realStderrFd();
  const int realErr = outerRealErr >= 0 ? outerRealErr : savedErr;

  // Watching needs the stream in hand line by line, which only the capture
  // path gives; a non-verbose scope that merely wants to know whether a
  // phrase appeared still takes it.
  if (verboseEnabled() || keepText || !watch.empty())
  {
    // Capture.  The reader drains the pipe for as long as anything holds
    // its write end -- which is fds 1 and 2 until the destructor puts the
    // saved copies back -- so a library that prints more than the pipe
    // buffers never blocks on it.
    int fds[2];
    if (savedOut >= 0 && savedErr >= 0 && makePipe(fds))
    {
      readFd = fds[0];
      if (!stderrOnly)
        (void)XPUPEAK_DUP2(fds[1], XPUPEAK_FILENO(stdout));
      (void)XPUPEAK_DUP2(fds[1], XPUPEAK_FILENO(stderr));
      (void)XPUPEAK_CLOSE(fds[1]);
      // Whatever renders a captured line must not write it into the capture.
      setRealStderrFd(realErr);
      try
      {
        reader = std::thread([this] { drain(); });
      }
      catch (...)
      {
        // No thread: put the console back and run unmuted, as --verbose
        // always did.
        setRealStderrFd(outerRealErr);
        if (!stderrOnly)
          (void)XPUPEAK_DUP2(savedOut, XPUPEAK_FILENO(stdout));
        (void)XPUPEAK_DUP2(savedErr, XPUPEAK_FILENO(stderr));
        (void)XPUPEAK_CLOSE(readFd);
        readFd = -1;
      }
    }
    return;
  }

#ifdef _MSC_VER
  FILE *nul = nullptr;
  (void)fopen_s(&nul, XPUPEAK_DEVNULL, "w");
#else
  FILE *nul = fopen(XPUPEAK_DEVNULL, "w");
#endif
  if (nul)
  {
    if (savedOut >= 0 && !stderrOnly) (void)XPUPEAK_DUP2(XPUPEAK_FILENO(nul), XPUPEAK_FILENO(stdout));
    if (savedErr >= 0) (void)XPUPEAK_DUP2(XPUPEAK_FILENO(nul), XPUPEAK_FILENO(stderr));
    (void)fclose(nul);
    // Even while discarding the library's console, xpupeak's own diagnostics
    // must still reach the real stderr rather than the bit bucket -- the same
    // bypass the capture path installs, so a warning emitted inside a muted
    // scope is not lost with the noise.
    if (savedErr >= 0)
      setRealStderrFd(realErr);
  }
}

ScopedConsoleMute::~ScopedConsoleMute() { finish(); }

void ScopedConsoleMute::finish()
{
  if (finished)
    return;
  finished = true;
  (void)fflush(stdout);
  (void)fflush(stderr);
  if (savedOut >= 0 && !stderrOnly) (void)XPUPEAK_DUP2(savedOut, XPUPEAK_FILENO(stdout));
  if (savedErr >= 0) (void)XPUPEAK_DUP2(savedErr, XPUPEAK_FILENO(stderr));
  // fd 2 is what it was before this scope, so the bypass goes back to the
  // enclosing scope's (or none) before the saved copy is closed -- a line
  // the reader is still rendering must not land on a closed (or, worse,
  // reused) descriptor, nor on fd 2 while that is an outer scope's pipe.
  setRealStderrFd(outerRealErr);
  if (savedOut >= 0) (void)XPUPEAK_CLOSE(savedOut);
  if (savedErr >= 0) (void)XPUPEAK_CLOSE(savedErr);
  if (readFd < 0)
    return;

  // Restoring the fds closed the pipe's last write end; the reader sees EOF
  // and returns.
  if (reader.joinable())
    reader.join();
  (void)XPUPEAK_CLOSE(readFd);

  if (!pending.empty())
  {
    record(pending);
    pending.clear();
  }
  if (lines > kMaxLines && verboseEnabled())
    logMessage(LogLevel::Debug, "console",
               std::to_string(lines - kMaxLines) +
                   " more lines from the library not recorded");
}

// One captured line: onto the run log under --verbose (the first kMaxLines
// of a scope), and into text() when the scope keeps it.
void ScopedConsoleMute::record(const std::string &line)
{
  if (!watch.empty() && !watchHit.load(std::memory_order_relaxed))
  {
    for (const std::string &needle : watch)
      if (!needle.empty() && line.find(needle) != std::string::npos)
      {
        watchHit.store(true, std::memory_order_relaxed);
        break;
      }
  }
  if (keepText)
  {
    captured += line;
    captured += '\n';
  }
  if (lines++ < kMaxLines && verboseEnabled())
    logMessage(LogLevel::Debug, "console", line);
}

void ScopedConsoleMute::drain()
{
  char buf[4096];
  for (;;)
  {
    const auto n = XPUPEAK_READ(readFd, buf, sizeof buf);
    if (n <= 0)
      break;
    pending.append(buf, static_cast<size_t>(n));
    // Emit whole lines as they complete, each as it happened: recorded (and
    // on the sidecar) before the library's next line, which is the point
    // when the next line is the one that crashes it.
    size_t start = 0;
    for (;;)
    {
      const size_t nl = pending.find('\n', start);
      if (nl == std::string::npos)
        break;
      const std::string line = pending.substr(start, nl - start);
      start = nl + 1;
      if (line.empty())
        continue;
      record(line);
    }
    pending.erase(0, start);
  }
}

} // namespace xpupeak
