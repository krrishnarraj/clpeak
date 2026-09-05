#include <common/console_mute.h>
#include <common/common.h>

#include <cstdio>

#ifdef _WIN32
#include <io.h>
#define CLPEAK_DUP    _dup
#define CLPEAK_DUP2   _dup2
#define CLPEAK_CLOSE  _close
#define CLPEAK_FILENO _fileno
#define CLPEAK_DEVNULL "NUL"
#else
#include <unistd.h>
#define CLPEAK_DUP    dup
#define CLPEAK_DUP2   dup2
#define CLPEAK_CLOSE  close
#define CLPEAK_FILENO fileno
#define CLPEAK_DEVNULL "/dev/null"
#endif

namespace clpeak {

ScopedConsoleMute::ScopedConsoleMute()
{
  if (verboseEnabled())
    return;
  (void)fflush(stdout);
  (void)fflush(stderr);
  savedOut = CLPEAK_DUP(CLPEAK_FILENO(stdout));
  savedErr = CLPEAK_DUP(CLPEAK_FILENO(stderr));
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

ScopedConsoleMute::~ScopedConsoleMute()
{
  (void)fflush(stdout);
  (void)fflush(stderr);
  if (savedOut >= 0) { (void)CLPEAK_DUP2(savedOut, CLPEAK_FILENO(stdout)); (void)CLPEAK_CLOSE(savedOut); }
  if (savedErr >= 0) { (void)CLPEAK_DUP2(savedErr, CLPEAK_FILENO(stderr)); (void)CLPEAK_CLOSE(savedErr); }
}

} // namespace clpeak
