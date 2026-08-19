#include <common/console_mute.h>
#include <common/common.h>

#include <cstdio>

#ifdef _WIN32
#include <io.h>
#define CLPEAK_DUP   _dup
#define CLPEAK_DUP2  _dup2
#define CLPEAK_CLOSE _close
#define CLPEAK_DEVNULL "NUL"
#else
#include <unistd.h>
#define CLPEAK_DUP   dup
#define CLPEAK_DUP2  dup2
#define CLPEAK_CLOSE close
#define CLPEAK_DEVNULL "/dev/null"
#endif

namespace clpeak {

ScopedConsoleMute::ScopedConsoleMute()
{
  if (verboseEnabled())
    return;
  (void)fflush(stdout);
  (void)fflush(stderr);
  savedOut = CLPEAK_DUP(fileno(stdout));
  savedErr = CLPEAK_DUP(fileno(stderr));
  FILE *nul = fopen(CLPEAK_DEVNULL, "w");
  if (nul)
  {
    if (savedOut >= 0) (void)CLPEAK_DUP2(fileno(nul), fileno(stdout));
    if (savedErr >= 0) (void)CLPEAK_DUP2(fileno(nul), fileno(stderr));
    (void)fclose(nul);
  }
}

ScopedConsoleMute::~ScopedConsoleMute()
{
  (void)fflush(stdout);
  (void)fflush(stderr);
  if (savedOut >= 0) { (void)CLPEAK_DUP2(savedOut, fileno(stdout)); (void)CLPEAK_CLOSE(savedOut); }
  if (savedErr >= 0) { (void)CLPEAK_DUP2(savedErr, fileno(stderr)); (void)CLPEAK_CLOSE(savedErr); }
}

} // namespace clpeak
