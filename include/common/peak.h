#ifndef PEAK_H
#define PEAK_H

#include <memory>
#include <string>
#include <bitset>
#include <vector>
#include "common.h"
#include "logger.h"

struct CliOptions;

class Peak {
public:
    std::unique_ptr<logger> log;

    // Which backend this is, for the device selector and the run loop.
    virtual Backend backend() const = 0;

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

    // ---- Device selection ------------------------------------------------
    // The --device items that name this backend (or none), resolved by
    // applyOptions.  Empty = every device.  `index` is the backend's own
    // numbering, the one --list-devices prints and the document records.
    std::vector<int> selectedDevices;

    bool isDeviceSelected(int index) const {
        if (selectedDevices.empty()) return true;
        for (int i : selectedDevices)
            if (i == index) return true;
        return false;
    }
    // --------------------------------------------------------------------

    virtual ~Peak() = default;

    // Copy common fields from CliOptions, including the device selection
    // for backend().  A derived class that overrides this MUST call the base
    // implementation.
    virtual void applyOptions(const CliOptions &opts);

    // Run all enabled benchmarks on available devices.
    virtual int runAll() = 0;
};

#endif // PEAK_H
