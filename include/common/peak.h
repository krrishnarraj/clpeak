#ifndef PEAK_H
#define PEAK_H

#include <memory>
#include <string>
#include "backend_gating.h"
#include "calibrate.h"
#include "logger.h"

struct CliOptions;

class Peak {
public:
    std::unique_ptr<logger> log;
    BackendGating gating;

    unsigned int warmupCount = 2;
    unsigned int specifiedIters = 0;
    unsigned int targetTimeUs = DEFAULT_TARGET_TIME_US;
    bool forceIters = false;

    Peak();
    virtual ~Peak() = default;

    // Copy common fields from CliOptions.  Derived classes MUST call this
    // base implementation in their override.
    virtual void applyOptions(const CliOptions &opts);

    // Run all enabled benchmarks on available devices.
    virtual int runAll() = 0;
};

#endif // PEAK_H
