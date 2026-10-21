#ifndef CLPEAK_FFI_LAUNCH_H
#define CLPEAK_FFI_LAUNCH_H

#include "clpeak_ffi.h"

// clpeak_launch() itself, with the choice of whether it clears a pending
// cancellation and pause first.  clpeak_launch() always does: each belongs
// to the launch it was aimed at.  The engine process (engine.cpp) does not,
// since its app may send them before the launch has begun and a reset would
// lose them.
int clpeakLaunch(int argc, const char **argv, ClpeakEventCallback on_event,
                 void *user_data, bool resetControls);

#endif // CLPEAK_FFI_LAUNCH_H
