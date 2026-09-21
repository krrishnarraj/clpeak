#ifndef CLPEAK_COREML_CACHE_H
#define CLPEAK_COREML_CACHE_H

// Core ML's runtime (E5RT) caches every model it specialises for the GPU or
// the Neural Engine under the user's Caches directory, keyed by process
// name or bundle identifier: `~/Library/Caches/<name>/com.apple.e5rt.e5bundlecache`.
// Each entry holds the compiled program *with its weights* -- twice, once as
// the bundle's weights.bin and once inside an MPSGraph package -- and
// nothing ever evicts them.  A benchmark that compiles a new model per size
// and precision leaves gigabytes behind per run: 292 GB, twelve thousand
// entries, accumulated on the development Mac between the CoreML execution
// provider's arrival in the ONNX backend and the day it was noticed, and one
// 16384-cube matmul session adds 4.5 GB.  macOS counts the directory as
// purgeable, so the volume reports hundreds of gigabytes "available" while
// a process writing a temporary file finds none.
//
// Both backends that reach Core ML -- the Core ML backend and the ONNX
// backend's CoreML execution provider -- call this after their models are
// gone.  It removes only this process's own E5RT cache directory (by process
// name and by bundle identifier), never the shared one at the Caches root.
// A no-op off Apple platforms.

namespace clpeak {

void purgeCoreMLCompileCache();

} // namespace clpeak

#endif // CLPEAK_COREML_CACHE_H
