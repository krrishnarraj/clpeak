#ifndef CL_UTILS_H
#define CL_UTILS_H

#include <string>
#include <cstdint>

// OpenCL-only utilities.  These are not used by Vulkan / CUDA / Metal.

#define UNUSED(expr) do { (void)(expr); } while (0)

// Round down to next multiple of the given base with an optional maximum value.
uint64_t roundToMultipleOf(uint64_t number, uint64_t base, uint64_t maxValue = UINT64_MAX);

// Remove trailing null characters (from OpenCL info strings).
void trimString(std::string &str);

#endif // CL_UTILS_H
