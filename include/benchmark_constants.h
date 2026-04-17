#ifndef BENCHMARK_CONSTANTS_H
#define BENCHMARK_CONSTANTS_H

// Centralized tuning constants that MUST match the corresponding values
// hard-coded in the OpenCL kernel source files (.cl).
//
// If you change a value here, update the matching kernel too (and vice versa).

// global_bandwidth_kernels.cl & kernel_latency.cpp
static const unsigned int FETCH_PER_WI = 16;

// local_bandwidth_kernels.cl
static const unsigned int LMEM_REPS = 64;

// atomic_throughput_kernels.cl
static const unsigned int ATOMIC_REPS = 512;

// image_bandwidth_kernels.cl
static const unsigned int IMAGE_FETCH_PER_WI = 16;

// compute_sp/hp/dp_kernels.cl  (128 iters * MAD_16 * 2 ops per MAD = 4096)
static const unsigned int COMPUTE_FP_WORK_PER_WI = 4096;

// compute_integer/intfast/char/short_kernels.cl  (64 iters * MAD_16 * 2 = 2048)
static const unsigned int COMPUTE_INT_WORK_PER_WI = 2048;

// Max work-group size cap.  Hardware may report higher (1024 on most NVIDIA
// GPUs), but we clamp to 256 because v16 kernels hold a float16/double16
// accumulator (~50-64 registers per thread).  At localSize=1024 this exceeds
// the SM register file on e.g. RTX 5060 (65536 regs/SM), causing
// clEnqueueNDRangeKernel to fail with CL_OUT_OF_RESOURCES.  256 matches
// clpeak's historical cap and leaves broad headroom across all devices.
static const unsigned int MAX_WG_SIZE = 256;

#endif // BENCHMARK_CONSTANTS_H
