#include <common/console_mute.h>
#include <common/common.h>

#include <cstdio>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define CLPEAK_DUP    _dup
#define CLPEAK_DUP2   _dup2
#define CLPEAK_CLOSE  _close
#define CLPEAK_FILENO _fileno
#define CLPEAK_READ   _read
#define CLPEAK_DEVNULL "NUL"
#else
#include <unistd.h>
#define CLPEAK_DUP    dup
#define CLPEAK_DUP2   dup2
#define CLPEAK_CLOSE  close
#define CLPEAK_FILENO fileno
#define CLPEAK_READ   read
#define CLPEAK_DEVNULL "/dev/null"
#endif

namespace clpeak {

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
{
  (void)fflush(stdout);
  (void)fflush(stderr);
  savedOut = CLPEAK_DUP(CLPEAK_FILENO(stdout));
  savedErr = CLPEAK_DUP(CLPEAK_FILENO(stderr));
  keepText = (mode == Capture::Always);

  if (verboseEnabled() || keepText)
  {
    // Capture.  The reader drains the pipe for as long as anything holds
    // its write end -- which is fds 1 and 2 until the destructor puts the
    // saved copies back -- so a library that prints more than the pipe
    // buffers never blocks on it.
    int fds[2];
    if (savedOut >= 0 && savedErr >= 0 && makePipe(fds))
    {
      readFd = fds[0];
      (void)CLPEAK_DUP2(fds[1], CLPEAK_FILENO(stdout));
      (void)CLPEAK_DUP2(fds[1], CLPEAK_FILENO(stderr));
      (void)CLPEAK_CLOSE(fds[1]);
      // Whatever renders a captured line must not write it into the capture.
      setRealStderrFd(savedErr);
      try
      {
        reader = std::thread([this] { drain(); });
      }
      catch (...)
      {
        // No thread: put the console back and run unmuted, as --verbose
        // always did.
        setRealStderrFd(-1);
        (void)CLPEAK_DUP2(savedOut, CLPEAK_FILENO(stdout));
        (void)CLPEAK_DUP2(savedErr, CLPEAK_FILENO(stderr));
        (void)CLPEAK_CLOSE(readFd);
        readFd = -1;
      }
    }
    return;
  }

#ifdef _MSC_VER
  FILE *nul = nullptr;
  (void)fopen_s(&nul, CLPEAK_DEVNULL, "w");
#else
  FILE *nul = fopen(CLPEAK_DEVNULL, "w");
#endif
  if (nul)
  {
    if (savedOut >= 0) (void)CLPEAK_DUP2(CLPEAK_FILENO(nul), CLPEAK_FILENO(stdout));
    if (savedErr >= 0) (void)CLPEAK_DUP2(CLPEAK_FILENO(nul), CLPEAK_FILENO(stderr));
    (void)fclose(nul);
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
  if (savedOut >= 0) (void)CLPEAK_DUP2(savedOut, CLPEAK_FILENO(stdout));
  if (savedErr >= 0) (void)CLPEAK_DUP2(savedErr, CLPEAK_FILENO(stderr));
  // fd 2 is the console again, so the bypass goes before its saved copy is
  // closed -- a line the reader is still rendering must not land on a
  // closed (or, worse, reused) descriptor.
  setRealStderrFd(-1);
  if (savedOut >= 0) (void)CLPEAK_CLOSE(savedOut);
  if (savedErr >= 0) (void)CLPEAK_CLOSE(savedErr);
  if (readFd < 0)
    return;

  // Restoring the fds closed the pipe's last write end; the reader sees EOF
  // and returns.
  if (reader.joinable())
    reader.join();
  (void)CLPEAK_CLOSE(readFd);

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
    const auto n = CLPEAK_READ(readFd, buf, sizeof buf);
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

} // namespace clpeak
