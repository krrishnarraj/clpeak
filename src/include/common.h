#ifndef COMMON_H
#define COMMON_H

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include <stdlib.h>

#define TAB             "  "
#define NEWLINE         "\n"
#define uint            unsigned int

#define MAX(X, Y)       \
    (X > Y)? X: Y;
    
#define MIN(X, Y)       \
    (X < Y)? X: Y;


typedef struct {
    uint numCUs;
    uint maxWGSize;
    uint maxAllocSize;
    uint maxGlobalSize;
    bool doubleSupported;
    
} device_info_t;

device_info_t getDeviceInfo(cl::Device &d);

uint roundToPowOf2(uint number);

void populate(float *ptr, uint N);
void populate(double *ptr, uint N);

#endif  // COMMON_H

