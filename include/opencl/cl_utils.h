#ifndef CL_UTILS_H
#define CL_UTILS_H

#include <string>
#include <cstdint>

// OpenCL-only utilities.  These are not used by Vulkan / CUDA / Metal.

#define UNUSED(expr) do { (void)(expr); } while (0)

// Round down to next multiple of the given base with an optional maximum value.
uint64_t roundToMultipleOf(uint64_t number, uint64_t base, uint64_t maxValue = UINT64_MAX);

// Fill an array with sequential values (used for bandwidth test input).
void populate(float *ptr, uint64_t N);

// Remove trailing null characters (from OpenCL info strings).
void trimString(std::string &str);

#endif // CL_UTILS_H
