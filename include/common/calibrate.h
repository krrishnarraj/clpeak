#ifndef CLPEAK_CALIBRATE_H
#define CLPEAK_CALIBRATE_H

// Default --max-time budget (microseconds).  500 ms is comfortably above
// the empirical M1 clock-ramp window (220-440 ms) so peak-frequency steady
// state is reached, while still leaving usable headroom under Adreno's
// 500 ms hangcheck.  This is the single source of truth -- CliOptions,
// benchmark_config_t::forDevice, and the backend constructors all read it.
// Keep the "500 ms" mention in the --help text in src/options.cpp in sync.
static const unsigned int DEFAULT_TARGET_TIME_US = 500000;

// Pick an iteration count from a measured per-iter time and a per-test
// time budget.  Used by every backend's runKernel/runDispatches helper to
// size the timed batch so it lands at ~target_us regardless of device
// speed (avoids GPU watchdog hits on slow paths and clock-ramp
// under-measurement on fast paths).
//
//   per_iter_us  measured time per dispatch from a calibration run
//   target_us    per-test budget (cfg.targetTimeUs); 0 => fall back to
//                a 5 s budget (matches the legacy BLAS pickIters
//                behaviour)
//   forced       if non-zero, short-circuit and return this value (the
//                user passed --iters)
//
// Result is clamped to [1, 10000] so a single dispatch/copy can be used when
// one iteration already exceeds the target budget, while still bounding
// command-buffer / event-pool size on fast paths.
unsigned int pickIters(double per_iter_us, unsigned int target_us, unsigned int forced);

#endif // CLPEAK_CALIBRATE_H
