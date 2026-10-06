#include <string>
#include <opencl/cl_peak.h>

#define MSTRINGIFY(...) #__VA_ARGS__

// mad_chain.cl goes in front of every compute family: it defines the
// AF*/RT*/M24_* macros the compute_*_alt_v* kernels expand.  See it for why
// every compute family carries two chain shapes.  Each family is its own
// program (clGetTestKernels).
static const std::string chainKernels =
#include "kernels/mad_chain.cl"
    ;

static const std::string globalBandwidthKernels =
#include "kernels/global_bandwidth_kernels.cl"
    ;

static const std::string spKernels =
#include "kernels/compute_sp_kernels.cl"
    ;

static const std::string hpKernels =
#include "kernels/compute_hp_kernels.cl"
    ;

static const std::string mpKernels =
#include "kernels/compute_mp_kernels.cl"
    ;

static const std::string dpKernels =
#include "kernels/compute_dp_kernels.cl"
    ;

static const std::string int24Kernels =
#include "kernels/compute_int24_kernels.cl"
    ;

static const std::string integerKernels =
#include "kernels/compute_integer_kernels.cl"
    ;

static const std::string charKernels =
#include "kernels/compute_char_kernels.cl"
    ;

static const std::string shortKernels =
#include "kernels/compute_short_kernels.cl"
    ;

static const std::string stringifiedLocalKernels =
#include "kernels/local_bandwidth_kernels.cl"
    ;

static const std::string stringifiedImageKernels =
#include "kernels/image_bandwidth_kernels.cl"
    ;

static const std::string stringifiedInt8DpKernels =
#include "kernels/compute_int8_dp_kernels.cl"
    ;

std::string clGetTestKernels(Benchmark which)
{
  switch (which)
  {
  case Benchmark::GlobalBW:
  case Benchmark::KernelLatency:
    return globalBandwidthKernels;
  case Benchmark::LocalBW:
    return stringifiedLocalKernels;
  case Benchmark::ImageBW:
    return stringifiedImageKernels;
  case Benchmark::ComputeInt8DP:
    return stringifiedInt8DpKernels;
  case Benchmark::ComputeSP:
    return chainKernels + spKernels;
  case Benchmark::ComputeHP:
    return chainKernels + hpKernels;
  case Benchmark::ComputeMP:
    return chainKernels + mpKernels;
  case Benchmark::ComputeDP:
    return chainKernels + dpKernels;
  case Benchmark::ComputeIntFast:
    return chainKernels + int24Kernels;
  case Benchmark::ComputeInt:
    return chainKernels + integerKernels;
  case Benchmark::ComputeChar:
    return chainKernels + charKernels;
  case Benchmark::ComputeShort:
    return chainKernels + shortKernels;
  default:
    return std::string();
  }
}
