#ifndef CLPEAK_DYNLIB_H
#define CLPEAK_DYNLIB_H

// Tiny cross-platform dynamic-library loader.  Used to make optional vendor
// libraries (cuBLASLt / hipBLASLt / rocBLAS, ONNX Runtime, LiteRT) load-on-
// demand: the benchmark resolves their symbols at run time and skips
// gracefully when the library is absent, so the shipped binary runs with only
// the GPU driver present.
//
// There is deliberately no dynClose().  A library that loaded has run its
// static constructors, and nothing portable says whether they can be undone.
// The runtimes here keep worker threads and accelerator contexts alive past
// their last object, and the ai-edge-litert 2.2.0 wheel's libLiteRt.so
// registers its hundred-odd static destructors through __cxa_atexit but was
// linked without the finalizer (.fini_array / __cxa_finalize) that runs and
// retires them on unload -- so dlclose left that many pointers into unmapped
// memory in glibc's exit list, and the process died at exit() inside
// whatever library had since been mapped over the hole (libonnxruntime.so,
// in the run that found this).  Every handle therefore stays open for the
// life of the process, including one that turned out not to be the library
// it was opened as: a few megabytes of address space is the price of a clean
// exit.

#include <initializer_list>

namespace clpeak {

// Try each candidate name in order; return the first that loads, or nullptr.
void *dynOpen(std::initializer_list<const char *> names);

// Resolve a symbol; nullptr if missing.
void *dynSym(void *lib, const char *name);

} // namespace clpeak

#endif // CLPEAK_DYNLIB_H
