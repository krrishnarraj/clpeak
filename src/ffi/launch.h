#ifndef XPUPEAK_FFI_LAUNCH_H
#define XPUPEAK_FFI_LAUNCH_H

#include "xpupeak_ffi.h"

// xpupeak_launch() itself, with the choice of whether it clears a pending
// cancellation and pause first.  xpupeak_launch() always does: each belongs
// to the launch it was aimed at.  The engine process (engine.cpp) does not,
// since its app may send them before the launch has begun and a reset would
// lose them.
int xpupeakLaunch(int argc, const char **argv, XpupeakEventCallback on_event,
                  void *user_data, bool resetControls);

#endif // XPUPEAK_FFI_LAUNCH_H
