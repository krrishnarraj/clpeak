#include <common/run_log.h>
#include <common/logger.h>

#include <cerrno>
#include <cstdio>
#include <system_error>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <share.h>
#include <windows.h>
#endif

namespace {
RunLog *g_current = nullptr;
// Held while g_current changes and while a host note uses it, so a note
// from another thread never meets a RunLog that is going away.
std::mutex g_currentMutex;

// The warning a side records the moment it starts dropping Debug: in the
// first dropped entry's scope and time, so it sits where the gap begins.
LogEntry capNotice(const LogEntry &dropped, std::size_t cap, const char *where)
{
    LogEntry e;
    e.elapsedS    = dropped.elapsedS;
    e.level       = clpeak::LogLevel::Warning;
    e.backend     = dropped.backend;
    e.device      = dropped.device;
    e.deviceIndex = dropped.deviceIndex;
    e.test        = dropped.test;
    e.message     = "log full: debug entries past " + std::to_string(cap >> 20) +
                    " MiB are left out of " + where + " from here on";
    return e;
}
}

RunLog::RunLog(RunDocument &doc)
    : doc(doc), start(std::chrono::steady_clock::now())
{
    {
        std::lock_guard<std::mutex> lock(g_currentMutex);
        g_current = this;
    }
    clpeak::setLogSink(this);
}

RunLog::~RunLog()
{
    if (clpeak::logSink() == this) clpeak::setLogSink(nullptr);
    {
        std::lock_guard<std::mutex> lock(g_currentMutex);
        if (g_current == this) g_current = nullptr;
    }
    // Not removed: a RunLog that goes away without closeSidecar(true) is a
    // run whose document was never saved, and the sidecar is its record.
    closeSidecar(false);
}

RunLog *RunLog::current()
{
    return g_current;
}

double RunLog::elapsedS() const
{
    return partElapsedS() + clockOffset.load(std::memory_order_relaxed);
}

double RunLog::partElapsedS() const
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
        .count();
}

// ── Sidecar ────────────────────────────────────────────────────────────────

std::string RunLog::sidecarPathFor(const std::string &outputFile)
{
    const std::string ext = ".json";
    if (outputFile.size() > ext.size() &&
        outputFile.compare(outputFile.size() - ext.size(), ext.size(), ext) == 0)
        return outputFile.substr(0, outputFile.size() - ext.size()) + ".log";
    return outputFile + ".log";
}

void RunLog::openSidecar(const std::string &outputFile, bool append)
{
    std::string failed;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (sidecar) return;
        sidecarPath = sidecarPathFor(outputFile);
        const char *mode = append ? "ab" : "wb";
        errno = 0;
#ifdef _WIN32
        // Not fopen_s: it opens the file unshared, so nothing can read the
        // sidecar while this run writes it -- which is when a hung run's
        // sidecar is read.  Readers in, other writers out.
        SetLastError(0);
        sidecar = _fsopen(sidecarPath.c_str(), mode, _SH_DENYWR);
#else
        sidecar = std::fopen(sidecarPath.c_str(), mode);
#endif
        if (!sidecar)
        {
            // Windows' own reason where there is one: a file another process
            // is writing reads "being used by another process", where errno
            // says only "Permission denied".  It can be the run before:
            // Windows Error Reporting keeps a crashed process alive, handles
            // and all, while it collects the crash.
            std::string why;
#ifdef _WIN32
            if (const DWORD err = GetLastError())
                why = std::system_category().message(static_cast<int>(err));
#endif
            if (why.empty() && errno != 0)
                why = std::generic_category().message(errno);
            failed = sidecarPath;
            if (!why.empty())
                failed += " (" + why + ")";
            sidecarPath.clear();
        }
        else
        {
            // Continued: the header and the part before are on disk already.
            const std::size_t from = append ? priorEntries : 0;
            if (!append)
                sidecarWriteLocked(runLogHeaderJson(doc.meta));
            for (std::size_t i = from; i < doc.log.size(); i++)
            {
                const LogEntry &e = doc.log[i];
                if (e.level == clpeak::LogLevel::Debug)
                    sidecarDebugBytes += e.message.size();
                sidecarWriteLocked(logEntryToJson(e));
            }
        }
    }
    if (!failed.empty())
        clpeak::logMessage(clpeak::LogLevel::Warning, "",
                           "cannot open the run log sidecar " + failed +
                               "; diagnostics will only reach the document");
}

void RunLog::closeSidecar(bool remove)
{
    std::lock_guard<std::mutex> lock(mutex);
    if (!sidecar) return;
    (void)std::fclose(sidecar);
    sidecar = nullptr;
    if (remove) (void)std::remove(sidecarPath.c_str());
    sidecarPath.clear();
}

void RunLog::sidecarWriteLocked(const std::string &line)
{
    if (!sidecar) return;
    (void)std::fputs(line.c_str(), sidecar);
    (void)std::fputc('\n', sidecar);
    // Flushed per line: what the sidecar is for is the crash that never
    // returns from the next call.  stdio's buffer would be the one that
    // held the line that mattered.
    (void)std::fflush(sidecar);
}

// ── Recording ──────────────────────────────────────────────────────────────

void RunLog::record(LogEntry &entry)
{
    entry.elapsedS = elapsedS();
    std::lock_guard<std::mutex> lock(mutex);
    appendLocked(entry, /*toDocument=*/true);
}

void RunLog::hostNote(clpeak::LogLevel level, const std::string &source,
                      const std::string &message)
{
    std::lock_guard<std::mutex> current(g_currentMutex);
    RunLog *log = g_current;
    if (!log || message.empty()) return;
    LogEntry e;
    e.level    = level;
    e.source   = source;
    e.message  = message;
    e.elapsedS = log->elapsedS();
    std::lock_guard<std::mutex> lock(log->mutex);
    log->appendLocked(e, /*toDocument=*/!log->finished);
}

void RunLog::recordToSidecar(LogEntry &entry)
{
    entry.elapsedS = elapsedS();
    std::lock_guard<std::mutex> lock(mutex);
    appendLocked(entry, /*toDocument=*/false);
}

void RunLog::appendLocked(LogEntry &entry, bool toDocument)
{
    bool toSidecar = sidecar != nullptr;
    if (entry.level == clpeak::LogLevel::Debug)
    {
        // Each side drops on its own cap, from the first entry that does not
        // fit on -- a later, shorter one would make a liar of the notice --
        // and says so on that side only: the sidecar goes on taking what the
        // document no longer does.
        const std::size_t n = entry.message.size();
        if (toDocument)
        {
            toDocument = documentDropped == 0 &&
                         documentDebugBytes + n <= kMaxDocumentDebugBytes;
            if (toDocument)
                documentDebugBytes += n;
            else if (documentDropped++ == 0)
                doc.log.push_back(capNotice(entry, kMaxDocumentDebugBytes, "the document"));
        }
        if (toSidecar)
        {
            toSidecar = sidecarDropped == 0 &&
                        sidecarDebugBytes + n <= kMaxSidecarDebugBytes;
            if (toSidecar)
                sidecarDebugBytes += n;
            else if (sidecarDropped++ == 0)
                sidecarWriteLocked(
                    logEntryToJson(capNotice(entry, kMaxSidecarDebugBytes, "this file")));
        }
    }
    if (toDocument) doc.log.push_back(entry);
    if (toSidecar) sidecarWriteLocked(logEntryToJson(entry));
}

void RunLog::onLog(clpeak::LogLevel level, const std::string &source,
                   const std::string &message)
{
    // The attached logger records the entry with its scope and renders it on
    // its channel.  attach()/detach() happen between backends, on the run's
    // own thread, when no callback is in flight -- so this read needs no lock.
    if (logger *l = active)
    {
        l->log(level, message, source);
        return;
    }
    LogEntry e;
    e.level   = level;
    e.source  = source;
    e.message = message;
    record(e);
    if (fallback)
    {
        fallback(e);
        return;
    }
    if (level == clpeak::LogLevel::Debug && !clpeak::verboseEnabled()) return;
    clpeak::stderrWrite(message + "\n");
}

void RunLog::attach(logger *l)
{
    active = l;
}

void RunLog::detach(logger *l)
{
    if (active == l) active = nullptr;
}

void RunLog::setFallbackRenderer(std::function<void(const LogEntry &)> render)
{
    fallback = std::move(render);
}

void RunLog::seedPrior(std::vector<LogEntry> prior, double clockOffsetS)
{
    std::lock_guard<std::mutex> lock(mutex);
    for (LogEntry &e : doc.log)
        e.elapsedS += clockOffsetS;
    clockOffset.store(clockOffsetS, std::memory_order_relaxed);
    // The part before's Debug counts against both caps: the document keeps
    // it, and the sidecar holds it already.
    for (const LogEntry &e : prior)
        if (e.level == clpeak::LogLevel::Debug)
        {
            documentDebugBytes += e.message.size();
            sidecarDebugBytes  += e.message.size();
        }
    priorEntries = prior.size();
    doc.log.insert(doc.log.begin(), std::make_move_iterator(prior.begin()),
                   std::make_move_iterator(prior.end()));
}

void RunLog::finish()
{
    std::size_t n;
    {
        std::lock_guard<std::mutex> lock(mutex);
        n = documentDropped;
    }
    if (n > 0)
        clpeak::logMessage(clpeak::LogLevel::Warning, "",
                           "log truncated: " + std::to_string(n) +
                               " debug entries left out after " +
                               std::to_string(kMaxDocumentDebugBytes >> 20) +
                               " MiB of debug messages");
    std::lock_guard<std::mutex> lock(mutex);
    finished = true;
}
