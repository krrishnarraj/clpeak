#ifndef CLPEAK_FFI_LAUNCH_H
#define CLPEAK_FFI_LAUNCH_H

#include "clpeak_ffi.h"

// clpeak_launch() itself, with the choice of whether it clears a pending
// cancellation first.  clpeak_launch() always does: a cancel belongs to the
// launch it was aimed at.  The engine process (engine.cpp) does not, since
// its app may send the cancel before the launch has begun and a reset would
// lose it.
int clpeakLaunch(int argc, const char **argv, ClpeakEventCallback on_event,
                 void *user_data, bool resetCancel);

#endif // CLPEAK_FFI_LAUNCH_H
