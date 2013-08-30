
#define MSTRINGIFY(A) #A

const char* stringifiedKernels = 
#include "bandwidth_kernels.cl"
#include "compute_kernels.cl"
;
