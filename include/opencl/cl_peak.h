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

// Kernel string accessors (defined in cl_kernels.cpp)
const std::string& clGetMainKernels();
const std::string& clGetLocalKernels();
const std::string& clGetImageKernels();
const std::string& clGetInt8DpKernels();

class clPeak : public Peak
{
public:
    // OpenCL-specific device selection.  Empty index list = run all.
    bool useEventTimer;
    std::vector<unsigned long> platformIndices, deviceIndices;

    clPeak();
    ~clPeak() override = default;

    // Set by runAll() before each device's benchmarks, read by runComputeTest()
    // and the per-benchmark methods.
    logger::DeviceScope *currentDeviceScope = nullptr;

    void applyOptions(const CliOptions &opts) override;
    int runAll() override;

    // Inventory.
    static BackendInventory enumerate();
    static void printInventory(const BackendInventory &inv, std::ostream &os);

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
    int runComputeTest(cl::CommandQueue &queue, cl::Program &prog,
                       device_info_t &devInfo, benchmark_config_t &cfg,
                       Benchmark which, const std::string &displayName,
                       const std::string &resultTag,
                       const std::string &kernelPrefix,
                       const std::string &typeName, const std::string &unit,
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
