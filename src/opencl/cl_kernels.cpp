#include <string>
#include <opencl/cl_peak.h>

#define MSTRINGIFY(...) #__VA_ARGS__

// mad_chain.cl first: it defines the AF*/RT*/M24_* macros the compute_*_alt_v*
// kernels expand.  See it for why every compute family carries two chain
// shapes.  Each family below is self-contained after it.
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

std::string clGetMainKernels(const std::function<bool(Benchmark)> &selected)
{
  struct Part
  {
    Benchmark which;
    const std::string *source;
  };
  static const Part parts[] = {
    {Benchmark::GlobalBW, &globalBandwidthKernels},
    {Benchmark::KernelLatency, &globalBandwidthKernels},
    {Benchmark::ComputeSP, &spKernels},
    {Benchmark::ComputeHP, &hpKernels},
    {Benchmark::ComputeMP, &mpKernels},
    {Benchmark::ComputeDP, &dpKernels},
    {Benchmark::ComputeIntFast, &int24Kernels},
    {Benchmark::ComputeInt, &integerKernels},
    {Benchmark::ComputeChar, &charKernels},
    {Benchmark::ComputeShort, &shortKernels},
  };

  std::string src;
  const std::string *last = nullptr;
  for (const Part &p : parts)
  {
    // The latency test borrows a global-bandwidth kernel, so that source can
    // be wanted twice in a row; it goes in once.
    if (!selected(p.which) || p.source == last)
      continue;
    src += *p.source;
    last = p.source;
  }
  return src.empty() ? src : chainKernels + src;
}

const std::string& clGetLocalKernels()   { return stringifiedLocalKernels; }
const std::string& clGetImageKernels()   { return stringifiedImageKernels; }
const std::string& clGetInt8DpKernels()  { return stringifiedInt8DpKernels; }
