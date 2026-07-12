#ifndef LOGGER_FFI_H
#define LOGGER_FFI_H

#include <common/logger.h>
#include "clpeak_ffi.h"

// ── FFI logger ──────────────────────────────────────────────────────────────
//
// Forwards the LogEvent stream as malloc'd JSON documents through the
// C event callback (see clpeak_ffi.h for the schema).  This is the single
// bridge every GUI platform consumes via Dart FFI.

class LoggerFfi : public logger
{
public:
    LoggerFfi(ClpeakEventCallback onEventCb, void *userData)
        : onEventCb(onEventCb), userData(userData) {}

protected:
    void onEvent(const LogEvent &e) override;

private:
    ClpeakEventCallback onEventCb;
    void               *userData;
};

// Serialize one event to its JSON document (exposed for the synthetic
// `note`/`done` events clpeak_launch emits itself).
std::string ffiEventToJson(const LogEvent &e);

// malloc-copy a string and hand it to the callback (no-op when cb is null).
void ffiEmitJson(ClpeakEventCallback cb, void *userData, const std::string &json);

#endif // LOGGER_FFI_H
