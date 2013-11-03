#ifndef COMMON_H
#define COMMON_H

#include <CL/cl.hpp>
#include <chrono>

#define TAB             "  "
#define NEWLINE         "\n"
#define uint            unsigned int


class Timer
{
public:

    std::chrono::high_resolution_clock::time_point tick, tock;

    void start();

    // Stop and return time in micro-seconds
    float stopAndTime();
};
    

typedef struct {
    uint numCUs;
    uint maxWGSize;
    uint maxAllocSize;
    uint maxGlobalSize;
    
} device_info_t;

device_info_t getDeviceInfo(cl::Device &d);

uint roundToPowOf2(uint number);

void populate(float *ptr, uint N);
void populate(double *ptr, uint N);

#endif  // COMMON_H

