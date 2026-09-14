// The one list of backends.  Compiled into both clpeak and clpeak_ffi (see
// src/common/cmake/backends.cmake), with the same ENABLE_* macros as the
// backend libraries it links, so the two binaries can never disagree about
// which backends exist or in what order they run.

#include <common/backend_registry.h>
#include <common/peak.h>
#include <algorithm>

#ifdef ENABLE_OPENCL
#include <opencl/cl_peak.h>
#endif
#ifdef ENABLE_VULKAN
#include <vulkan/vk_peak.h>
#endif
#ifdef ENABLE_CUDA
#include <cuda/cuda_peak.h>
#endif
#ifdef ENABLE_ROCM
#include <rocm/rocm_peak.h>
#endif
#ifdef ENABLE_METAL
#include <metal/mtl_peak.h>
#endif
#ifdef ENABLE_ONEAPI
#include <oneapi/oneapi_peak.h>
#endif
#ifdef ENABLE_CPU
#include <cpu/cpu_peak.h>
#endif
#ifdef ENABLE_ONNX
#include <onnx/onnx_peak.h>
#endif
#ifdef ENABLE_COREML
#include <coreml/coreml_peak.h>
#endif

namespace
{

// The class says which backend it is (P::kBackend); nothing is repeated here.
template <class P>
BackendEntry entry()
{
    return {P::kBackend,
            [] { return P::enumerate(); },
            []() -> std::unique_ptr<Peak> { return std::make_unique<P>(); }};
}

std::vector<BackendEntry> build()
{
    std::vector<BackendEntry> out;
#ifdef ENABLE_OPENCL
    out.push_back(entry<clPeak>());
#endif
#ifdef ENABLE_VULKAN
    out.push_back(entry<vkPeak>());
#endif
#ifdef ENABLE_CUDA
    out.push_back(entry<CudaPeak>());
#endif
#ifdef ENABLE_ROCM
    out.push_back(entry<RocmPeak>());
#endif
#ifdef ENABLE_METAL
    out.push_back(entry<MetalPeak>());
#endif
#ifdef ENABLE_ONEAPI
    out.push_back(entry<OneapiPeak>());
#endif
#ifdef ENABLE_CPU
    out.push_back(entry<CpuPeak>());
#endif
#ifdef ENABLE_ONNX
    out.push_back(entry<OnnxPeak>());
#endif
#ifdef ENABLE_COREML
    out.push_back(entry<CoreMLPeak>());
#endif
    // The enum is the order; the push_back order above is not.
    std::sort(out.begin(), out.end(),
              [](const BackendEntry &a, const BackendEntry &b) { return a.id < b.id; });
    return out;
}

} // namespace

const std::vector<BackendEntry> &backendRegistry()
{
    static const std::vector<BackendEntry> registry = build();
    return registry;
}
