#ifndef CLPEAK_DYNLIB_H
#define CLPEAK_DYNLIB_H

// Tiny cross-platform dynamic-library loader.  Used to make optional vendor
// libraries (cuBLASLt / hipBLASLt / rocBLAS) load-on-demand: the benchmark
// resolves their symbols at run time and skips gracefully when the library is
// absent, so the shipped binary runs with only the GPU driver present.

#include <initializer_list>

namespace clpeak {

// Try each candidate name in order; return the first that loads, or nullptr.
void *dynOpen(std::initializer_list<const char *> names);

// Resolve a symbol; nullptr if missing.
void *dynSym(void *lib, const char *name);

void dynClose(void *lib);

} // namespace clpeak

#endif // CLPEAK_DYNLIB_H
