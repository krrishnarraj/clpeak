#include <common/run_log.h>
#include <common/logger.h>

#include <cstdio>
#include <utility>

namespace {
RunLog *g_current = nullptr;
}

RunLog::RunLog(RunDocument &doc)
    : doc(doc), start(std::chrono::steady_clock::now())
{
    g_current = this;
    clpeak::setLogSink(this);
}

RunLog::~RunLog()
{
    if (clpeak::logSink() == this) clpeak::setLogSink(nullptr);
    if (g_current == this) g_current = nullptr;
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

void RunLog::openSidecar(const std::string &outputFile)
{
    std::string failed;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (sidecar) return;
        sidecarPath = sidecarPathFor(outputFile);
#ifdef _MSC_VER
        (void)fopen_s(&sidecar, sidecarPath.c_str(), "wb");
#else
        sidecar = std::fopen(sidecarPath.c_str(), "wb");
#endif
        if (!sidecar)
        {
            failed = sidecarPath;
            sidecarPath.clear();
        }
        else
        {
            sidecarWriteLocked(runLogHeaderJson(doc.meta));
            for (const LogEntry &e : doc.log) sidecarWriteLocked(logEntryToJson(e));
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
    appendLocked(entry);
}

void RunLog::appendLocked(LogEntry &entry)
{
    const bool essential = entry.level == clpeak::LogLevel::Error ||
                           entry.level == clpeak::LogLevel::Warning;
    if (!essential && messageBytes + entry.message.size() > kMaxMessageBytes)
    {
        dropped++;
        return;
    }
    messageBytes += entry.message.size();
    doc.log.push_back(entry);
    sidecarWriteLocked(logEntryToJson(entry));
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

void RunLog::finish()
{
    std::size_t n;
    {
        std::lock_guard<std::mutex> lock(mutex);
        n = dropped;
    }
    if (n == 0) return;
    clpeak::logMessage(clpeak::LogLevel::Warning, "",
                       "log truncated: " + std::to_string(n) +
                           " debug/info entries dropped after " +
                           std::to_string(kMaxMessageBytes >> 20) +
                           " MiB of messages");
}
