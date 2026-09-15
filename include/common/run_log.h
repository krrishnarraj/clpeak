#ifndef CLPEAK_RUN_LOG_H
#define CLPEAK_RUN_LOG_H

#include <chrono>
#include <functional>
#include <mutex>
#include <string>

#include <common/common.h>
#include <common/run_document.h>

class logger;

// ── The run's diagnostic stream ────────────────────────────────────────────
//
// One RunLog per run, owned by the host (the CLI's main(), clpeak_launch())
// and alive for the whole of it.  It is the process's LogSink while it lives:
// every clpeak::logMessage() -- a CLPEAK_VLOG line in a backend, a logger's
// note(), a warning relayed from ONNX Runtime's logger, a line captured off
// a muted console -- lands here, and from here on the document's `log`.
//
// The per-backend loggers come and go underneath it.  The one that is open
// (attach()) supplies the scope an entry fired in -- backend, device, test --
// and renders it on its channel (the CLI's text, the GUI's event stream);
// with none open, the entry is recorded scope-less and rendered by the host's
// fallback, so a message between two backends is neither lost nor misfiled.
//
// The cap: a runtime that logs per node per session can produce more text
// than a phone wants to share.  Past kMaxMessageBytes of message text, Debug
// and Info entries are dropped (and counted); warnings and errors always get
// through, and finish() records how much was dropped.
//
// Callbacks from vendor runtimes fire on their own threads, so record() is
// serialised.  Scope is read from the open logger without locking it: the
// callbacks that arrive off-thread do so while the calling thread is inside
// the library call that triggered them, with its scope standing still.
class RunLog : public clpeak::LogSink
{
public:
    // Entries go to `doc.log`.
    explicit RunLog(RunDocument &doc);
    ~RunLog() override;

    RunLog(const RunLog &) = delete;
    RunLog &operator=(const RunLog &) = delete;

    // The run's RunLog, or nullptr outside a run.
    static RunLog *current();

    // Seconds since this RunLog was created -- the run's t=0, which is also
    // what `generated_at` names.
    double elapsedS() const;

    // ---- Recording -------------------------------------------------------

    // Append a fully-formed entry (scope already filled) -- what a logger
    // calls after adding its scope.  Stamps `elapsedS` on the caller's copy
    // too, so the event it goes on to render carries the same time.
    void record(LogEntry &entry);

    // LogSink: a message with no logger context of its own.  Scoped by the
    // attached logger when there is one, else recorded as-is.
    void onLog(clpeak::LogLevel level, const std::string &source,
               const std::string &message) override;

    // The logger whose scope and channel entries use while it lives.
    void attach(logger *l);
    void detach(logger *l);

    // How an entry is shown when no logger is attached.  The CLI prints;
    // the GUI bridge forwards an event.  Unset: stderr, as with no run.
    void setFallbackRenderer(std::function<void(const LogEntry &)> render);

    // Record the truncation summary, if anything was dropped.  Call before
    // the document is saved.
    void finish();

    // Message text the log keeps before it starts dropping Debug and Info.
    static constexpr std::size_t kMaxMessageBytes = 4u << 20;

private:
    void appendLocked(LogEntry &entry);

    RunDocument &doc;
    std::chrono::steady_clock::time_point start;
    std::mutex   mutex;
    logger      *active = nullptr;
    std::function<void(const LogEntry &)> fallback;

    std::size_t  messageBytes = 0;
    std::size_t  dropped      = 0;
};

#endif  // CLPEAK_RUN_LOG_H
