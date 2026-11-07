#ifndef CLPEAK_THERMAL_H
#define CLPEAK_THERMAL_H

// Pauses a run while the OS says the device is too hot to touch.
//
// A phone has no fan: minutes of every core or the GPU at full load take it
// to the temperature where the OS warns its user and throttles everything,
// and a Galaxy S24 run got there.  Each reading taken that hot is also a
// throttled one, below what the device does on a cool start.  So at every
// boundary between readings -- a test starting, a row recorded -- the run
// asks the OS for its thermal status, and from severe up it waits for the
// device to fall back to light before measuring again, for at most five
// minutes a pause.  Per platform:
//
//   Android  AThermal_getCurrentThermalStatus (API 30; libandroid is opened
//            at run time, so a build for an older API still links), whose
//            NONE..SHUTDOWN levels are the scale here.
//   iOS      NSProcessInfo.thermalState: nominal, fair, serious and critical
//            read as none, light, severe and critical.
//   Others   nothing: a desktop or laptop cools itself, and there the run
//            never pauses.
//
// Every status the gate reads differently from the last is logged, so a
// run's log says which rows were measured hot; a pause is a warning, which
// the CLI prints with the results, and its end an info line.

#include <functional>

namespace clpeak {

// The OS's thermal status is watched for as long as one lives: one per run,
// beside KeepAwake, in the CLI's run loop and the GUI bridge's alike.
// Outside a run -- the device listing, the GUI's catalog -- the gate does
// nothing.
class ThermalWatch
{
public:
  ThermalWatch();
  ~ThermalWatch();

  ThermalWatch(const ThermalWatch &) = delete;
  ThermalWatch &operator=(const ThermalWatch &) = delete;
};

// A boundary between readings: log a changed status, and wait while it is
// severe or worse.  A cancelled run stops waiting at once.  `beforePausing`
// runs once, when the gate is about to wait (see holdGate in screen_hold.h).
void thermalGate(const std::function<void()> &beforePausing = {});

} // namespace clpeak

#endif // CLPEAK_THERMAL_H
