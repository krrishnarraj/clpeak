#ifndef COMMON_H
#define COMMON_H

#include <CL/cl.hpp>
#include <chrono>

#define TAB             "  "
#define NEWLINE         "\n"
#define uint            unsigned int

// Initialize timer
#define INIT_TIMER(timer_id)                            \
    auto timer_id = chrono::high_resolution_clock::now()

// Return elapsed time in micro-seconds
#define ELAPSED_TIME(timer_id)                          \
    (chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now() - timer_id).count()) 


typedef struct {
    uint numCUs;
    uint maxWGSize;
    
} device_info_t;

device_info_t getDeviceInfo(cl::Device &d);

int populate(float *ptr, uint N);
int populate(double *ptr, uint N);

#endif  // COMMON_H

