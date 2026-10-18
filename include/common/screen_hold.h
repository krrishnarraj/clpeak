#ifndef CLPEAK_SCREEN_HOLD_H
#define CLPEAK_SCREEN_HOLD_H

// Holds a run while the app hosting it is off screen.
//
// On a phone a run is the app's own process, and an app off screen is not
// left to run as it was: Android and iOS move it to background cores at
// background clocks, then freeze it, and end it when memory is wanted.  A
// Galaxy Z Fold8 froze clpeak about 45 and 70 s after it left the screen,
// once inside a test, and a CPU backend that started as the app came back
// ran on four of the eight cores.  A reading taken off screen measures the
// background; one a freeze lands inside measures the freeze.  So at every
// boundary between readings -- the thermal gate's: a test starting, a row
// recorded -- a run whose app is off screen waits until it is back or the
// run is cancelled.  With no limit: what the phone does to the app meanwhile
// is the phone's, and a run that waited at a boundary resumes cleanly.
//
// Only an app says where it is (clpeak_set_on_screen, src/ffi/clpeak_ffi.h).
// The CLI and the desktop engine process never do, so their runs never wait.

namespace clpeak {

// Whether the app is on screen; true until said otherwise.  Each change
// during a run is recorded on its log (RunLog::hostNote).  Safe from any
// thread.
void setHostOnScreen(bool onScreen);

// A boundary between readings: while the app is off screen, log where the
// run stopped and wait.  A cancelled run stops waiting at once.
void screenGate();

} // namespace clpeak

#endif // CLPEAK_SCREEN_HOLD_H
