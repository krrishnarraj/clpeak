#ifndef PEAK_H
#define PEAK_H

#include <memory>
#include <string>
#include <bitset>
#include "common.h"
#include "logger.h"

struct CliOptions;

class Peak {
public:
    std::unique_ptr<logger> log;

    unsigned int warmupCount = 2;
    unsigned int specifiedIters = 0;
    unsigned int targetTimeUs = DEFAULT_TARGET_TIME_US;
    bool forceIters = false;

    // ---- Gating state (was BackendGating) -------------------------------
    std::bitset<static_cast<size_t>(Benchmark::COUNT)> enabledTests;
    // Unknown is the sentinel and never gets a bit (isCategoryEnabled
    // rejects it), so it doubles as the count of real categories.
    std::bitset<static_cast<size_t>(Category::Unknown)> enabledCategories;

    Peak() {
        enabledTests.set();
        enabledCategories.set();
    }

    bool isTestEnabled(Benchmark b) const {
        return enabledTests.test(static_cast<size_t>(b));
    }

    bool isCategoryEnabled(Category c) const {
        if (c == Category::Unknown) return false;
        return enabledCategories.test(static_cast<size_t>(c));
    }

    // A requested cancellation (clpeak::requestCancel) gates every remaining
    // test off, so runAll() unwinds quickly at the next test boundary.
    bool isAllowed(Benchmark b) const {
        return !clpeak::cancelRequested() &&
               isCategoryEnabled(categoryOf(b)) && isTestEnabled(b);
    }

    bool isAllowedAs(Benchmark b, Category c) const {
        return !clpeak::cancelRequested() &&
               isCategoryEnabled(c) && isTestEnabled(b);
    }
    // --------------------------------------------------------------------

    virtual ~Peak() = default;

    // Copy common fields from CliOptions.  Derived classes MUST call this
    // base implementation in their override.
    virtual void applyOptions(const CliOptions &opts);

    // Run all enabled benchmarks on available devices.
    virtual int runAll() = 0;
};

#endif // PEAK_H
