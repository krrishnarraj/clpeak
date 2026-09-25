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
#include <string>

namespace clpeak {

// Try each candidate name in order; return the first that loads, or nullptr.
void *dynOpen(std::initializer_list<const char *> names);

// Absolute form of a user-named module path for the OS loader.  A relative
// module path finds the file itself against the current directory but not
// the sibling libraries beside it, so on Windows the same file fails to
// load by relative path and succeeds by absolute one (the --onnx-winml
// steered runtime proved it).  Bare sonames ("libfoo.so") and Apple
// @-paths are loader search tokens rather than filesystem paths and pass
// through untouched, as does empty.
std::string absoluteModulePath(const char *name);

// Whether two module paths name one file: their absolute forms compared (on
// Windows without case and with either slash), then the filesystem asked
// whether they are the same file (a symlink, a hard link).  A loader that
// fixes its runtime for the process uses it to tell a real change of
// library from the same file named again.
bool sameModulePath(const std::string &a, const std::string &b);

// Resolve a symbol; nullptr if missing.
void *dynSym(void *lib, const char *name);

} // namespace clpeak

#endif // CLPEAK_DYNLIB_H
