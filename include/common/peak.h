#ifndef PEAK_H
#define PEAK_H

#include <memory>
#include <string>
#include <bitset>
#include <map>
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

    // What the run was asked for -- the category and test flags -- and
    // nothing else: blind to a cancel and to a resume.  For a test that asks
    // whether another one measures its family (the CPU's int16 dot product
    // shares the int8 test), where "already measured by the part of the run
    // this one resumes" must read as measured, not as deselected.
    bool isSelected(Benchmark b) const {
        return isCategoryEnabled(categoryOf(b)) && isTestEnabled(b);
    }

    // A requested cancellation (clpeak::requestCancel) gates every remaining
    // test off, so runAll() unwinds quickly at the next test boundary.  So
    // does a Benchmark the part of the run this one resumes already finished
    // on the open device (--resume).
    bool isAllowed(Benchmark b) const {
        return !clpeak::cancelRequested() && isSelected(b) && !resumedPast(b);
    }

    // ---- Benchmark admission --------------------------------------------
    //
    // A backend's one gate per Benchmark per device:
    //
    //   if (auto run = admit(Benchmark::ComputeSP)) runComputeSP(dev, cfg);
    //
    // True when isAllowed() is.  When the guard goes out of scope -- the
    // runner has returned -- the Benchmark is finished on the open device:
    // the run's checkpoint records it, and a run resumed from that
    // checkpoint skips it (include/common/run_checkpoint.h).  A cancel or an
    // exception that cuts it short leaves it unfinished.
    //
    // isAllowed() stays the question a backend asks without starting
    // anything -- which kernels to build, whether a shared probe is needed.
    // A Benchmark measured by several runners takes one guard around all of
    // them: one per runner would mark it finished after the first.
    class Admission {
    public:
        Admission(Peak *peak, Benchmark b, bool admitted);
        ~Admission();
        Admission(const Admission &) = delete;
        Admission &operator=(const Admission &) = delete;
        Admission(Admission &&other) noexcept;
        Admission &operator=(Admission &&) = delete;

        explicit operator bool() const { return admitted; }

    private:
        Peak     *peak;
        Benchmark bench;
        bool      admitted;
        int       exceptionsAtEntry;
    };

    Admission admit(Benchmark b);

    // ---- Device selection ------------------------------------------------
    // The --devices items that name this backend, resolved by applyOptions.
    // Empty = every device.  `index` is the backend's own numbering, the one
    // --list-devices prints and the document records.
    std::vector<int> selectedDevices;

    // --resume: this backend's devices the part of the run before finished
    // whole, and on the rest the Benchmarks it finished, by device index.
    std::vector<int> resumeDevicesDone;
    std::map<int, std::bitset<static_cast<size_t>(Benchmark::COUNT)>> resumeBenchmarksDone;

    bool isDeviceSelected(int index) const {
        for (int i : resumeDevicesDone)
            if (i == index) return false;
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

private:
    // Whether the part of the run this one resumes finished `b` on the
    // device open in this backend's logger.
    bool resumedPast(Benchmark b) const;

    // `b` finished on the open device: tell the run's checkpoint.
    void benchmarkFinished(Benchmark b);

    // --skip-crashed: the Benchmark a resumed run reaches first is where the
    // part before it died, and is recorded as such instead of run again.
    void recordCrashed(Benchmark b);
};

#endif // PEAK_H
