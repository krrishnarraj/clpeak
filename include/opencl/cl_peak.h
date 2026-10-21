#ifndef CL_PEAK_H
#define CL_PEAK_H

#include <common/peak.h>
#include <opencl/cl_common.h>
#include <opencl/cl_utils.h>
#include <common/inventory.h>
#include <string>
#include <memory>
#include <vector>

struct CliOptions;

#define BUILD_OPTIONS " -cl-mad-enable "

// Shared note for one reading of a vector-width sweep.
static inline const char *clWidthNote(int width)
{
  switch (width)
  {
  case 1:  return "Scalar, one value per work-item.";
  case 2:  return "2-wide vector per work-item.";
  case 4:  return "4-wide vector per work-item.";
  case 8:  return "8-wide vector per work-item.";
  case 16: return "16-wide vector per work-item.";
  default: return "";
  }
}

// The same for int8_dp, whose v2..v16 kernels run that many independent
// chains of scalar dot products rather than a wider vector.
static inline const char *clChainNote(int chains)
{
  switch (chains)
  {
  case 1:  return "One dependent chain of dot products.";
  case 2:  return "2 independent chains.";
  case 4:  return "4 independent chains.";
  case 8:  return "8 independent chains.";
  case 16: return "16 independent chains.";
  default: return "";
  }
}

// The source of the program a test runs (defined in cl_kernels.cpp): a
// compute family behind the chain macros it expands, the global-bandwidth
// kernels the latency test borrows, or one of the local, image and int8
// programs.  Empty for a test with no kernel of its own.  One program per test,
// and only the selected tests', because on a cold driver cache the compile is
// most of a device's setup -- an Arc A380 spent ~45 s on all of them as one --
// and separate programs build side by side (runAll).
std::string clGetTestKernels(Benchmark which);

class clPeak : public Peak
{
public:
    clPeak();
    ~clPeak() override = default;

    // Set by runAll() before each device's benchmarks, read by runComputeTest()
    // and the per-benchmark methods.
    logger::DeviceScope *currentDeviceScope = nullptr;

    // Which backend this is -- the one place that says so; the registry,
    // the inventory and the device selector all read it from here.
    static constexpr Backend kBackend = Backend::OpenCL;
    Backend backend() const override { return kBackend; }
    int runAll() override;

    // Inventory.
    static BackendInventory enumerate();

    // Time a kernel batched as `iters` dispatches, where `iters` is calibrated
    // from a one-shot warmup so the timed phase lands at ~targetTimeUs.
    // Clamps the local size to the kernel's own work-group limit first (see
    // clampToKernelWG), so callers may pass the device-max local size freely.
    float run_kernel(cl::CommandQueue &queue, cl::Kernel &kernel,
                     cl::NDRange &globalSize, cl::NDRange &localSize,
                     unsigned int targetTimeUs, unsigned int forcedIters);

    // A kernel's own max work-group size (CL_KERNEL_WORK_GROUP_SIZE) can be
    // smaller than the device max -- e.g. register pressure on wide vector
    // widths, or a driver quirk -- and launching above it fails with -54
    // (CL_INVALID_WORK_GROUP_SIZE). Clamp the (1-D) local size to that limit and
    // re-align the global size down to a multiple of it, in place. No-op when
    // the requested local size already fits. run_kernel calls this for every
    // launch that goes through it; direct enqueue sites call it themselves.
    static void clampToKernelWG(const cl::Device &dev, cl::Kernel &kernel,
                                cl::NDRange &globalSize, cl::NDRange &localSize);

    // Total work-item count of an NDRange (product of its dimensions). Used to
    // compute throughput from the effective global size after clampToKernelWG.
    static uint64_t ndRangeTotal(const cl::NDRange &range);

    // Unified compute benchmark helper — replaces 7 nearly-identical runCompute* methods.
    // `description` is the test's one- or two-sentence explanation
    // (logger::TestSpec::description); the per-width readings are documented
    // by clWidthNote() (clChainNote() for int8_dp) inside the helper.  A
    // non-empty `buildError` says `prog` failed to build where it should
    // have, and every reading becomes that error.
    int runComputeTest(cl::CommandQueue &queue, cl::Program &prog,
                       const std::string &buildError,
                       device_info_t &devInfo, benchmark_config_t &cfg,
                       Benchmark which, const std::string &displayName,
                       const std::string &resultTag,
                       const std::string &kernelPrefix,
                       const std::string &typeName, const std::string &unit,
                       const std::string &description,
                       unsigned int workPerWI, unsigned int wgsPerCU,
                       size_t elemSize);

    // Per-benchmark methods.
    int runGlobalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog,
                               device_info_t &devInfo, benchmark_config_t &cfg);
    int runLocalBandwidthTest(cl::CommandQueue &queue, cl::Program &prog,
                              device_info_t &devInfo, benchmark_config_t &cfg);
    int runImageBandwidthTest(cl::CommandQueue &queue, cl::Program &prog,
                              device_info_t &devInfo, benchmark_config_t &cfg);
    int runTransferBandwidthTest(cl::CommandQueue &queue, cl::Program &prog,
                                 device_info_t &devInfo, benchmark_config_t &cfg);
    int runKernelLatency(cl::CommandQueue &queue, cl::Program &prog,
                         device_info_t &devInfo, benchmark_config_t &cfg);
};

#endif // CL_PEAK_H
