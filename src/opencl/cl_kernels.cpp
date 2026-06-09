#include <string>
#include <opencl/cl_peak.h>

#define MSTRINGIFY(...) #__VA_ARGS__

static const std::string stringifiedKernels =
#include "kernels/global_bandwidth_kernels.cl"
#include "kernels/compute_sp_kernels.cl"
#include "kernels/compute_hp_kernels.cl"
#include "kernels/compute_mp_kernels.cl"
#include "kernels/compute_dp_kernels.cl"
#include "kernels/compute_int24_kernels.cl"
#include "kernels/compute_integer_kernels.cl"
#include "kernels/compute_char_kernels.cl"
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

const std::string& clGetMainKernels()    { return stringifiedKernels; }
const std::string& clGetLocalKernels()   { return stringifiedLocalKernels; }
const std::string& clGetImageKernels()   { return stringifiedImageKernels; }
const std::string& clGetInt8DpKernels()  { return stringifiedInt8DpKernels; }
