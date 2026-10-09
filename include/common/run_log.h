#ifndef CLPEAK_RUN_LOG_H
#define CLPEAK_RUN_LOG_H

#include <chrono>
#include <cstdio>
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
// Two things make the stream worth more than a terminal's:
//
//   The sidecar.  With -o, every entry is also appended to `<output>.log`
//   (`.json` swapped for `.log`) and flushed before the call returns, one
//   JSON object per line under a header line naming the run.  The document
//   is written when the run ends; a native crash inside a driver -- the case
//   --verbose exists for -- means it never is, and the sidecar is then the
//   only record.  It is removed once the document has been saved.
//
//   The caps.  A runtime that logs per node per session can produce more
//   text than a phone wants to share.  Only Debug is capped: Info is where
//   the run is and what it read, and with warnings and errors it always gets
//   through.  The document keeps kMaxDocumentDebugBytes of Debug text and
//   finish() counts what it left out; the sidecar, which is on disk, gone
//   once the document is saved, and read for the run's last minutes, keeps
//   kMaxSidecarDebugBytes.  Each says so in a warning the moment it starts
//   dropping, so a run that dies past its cap says it was cut.
//
// Callbacks from vendor runtimes fire on their own threads, so record() is
// serialised.  Scope is read from the open logger without locking it: the
// callbacks that arrive off-thread do so while the calling thread is inside
// the library call that triggered them, with its scope standing still.
class RunLog : public clpeak::LogSink
{
public:
    // Entries go to `doc.log`; `doc.meta` is what the sidecar header names,
    // so stamp it before openSidecar().
    explicit RunLog(RunDocument &doc);
    ~RunLog() override;

    RunLog(const RunLog &) = delete;
    RunLog &operator=(const RunLog &) = delete;

    // The run's RunLog, or nullptr outside a run.
    static RunLog *current();

    // Seconds since this RunLog was created -- the run's t=0, which is also
    // what `generated_at` names.
    double elapsedS() const;

    // ---- Sidecar ---------------------------------------------------------

    // Start mirroring entries to the sidecar of `outputFile`.  Entries
    // already recorded are written first, so nothing recorded before the
    // output path was known is missing from it.
    void openSidecar(const std::string &outputFile);

    // Stop mirroring; remove the file when the document it stood in for has
    // been saved.
    void closeSidecar(bool remove);

    // The sidecar path for an output file: `run.clpeak.json` ->
    // `run.clpeak.log`; a name without `.json` gets `.log` appended.
    static std::string sidecarPathFor(const std::string &outputFile);

    // ---- Recording -------------------------------------------------------

    // Append a fully-formed entry (scope already filled) -- what a logger
    // calls after adding its scope.  Stamps `elapsedS` on the caller's copy
    // too, so the event it goes on to render carries the same time.
    void record(LogEntry &entry);

    // The same, onto the sidecar alone: a reading without --verbose, which
    // the document holds in its device tree and keeps off its log.  A run
    // that dies leaves no tree, and the sidecar keeps the numbers it got to.
    void recordToSidecar(LogEntry &entry);

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

    // Record how much Debug the document left out, if any.  Call before
    // the document is saved.
    void finish();

    // Debug message text each keeps before it starts dropping Debug.
    static constexpr std::size_t kMaxDocumentDebugBytes = 64u << 20;
    static constexpr std::size_t kMaxSidecarDebugBytes  = 128u << 20;

private:
    void appendLocked(LogEntry &entry, bool toDocument);
    void sidecarWriteLocked(const std::string &line);

    RunDocument &doc;
    std::chrono::steady_clock::time_point start;
    std::mutex   mutex;
    logger      *active = nullptr;
    std::function<void(const LogEntry &)> fallback;

    FILE        *sidecar = nullptr;
    std::string  sidecarPath;

    std::size_t  documentDebugBytes = 0;
    std::size_t  documentDropped    = 0;
    std::size_t  sidecarDebugBytes  = 0;
    std::size_t  sidecarDropped     = 0;
};

#endif  // CLPEAK_RUN_LOG_H
