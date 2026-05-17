#include "logger_android.h"
#include <sstream>

// ── note ───────────────────────────────────────────────────────────────────

void LoggerAndroid::note(const std::string &msg)
{
    jstring jstr = jEnv->NewStringUTF(msg.c_str());
    jEnv->CallVoidMethod(jObj, printCallback, jstr);
    jEnv->DeleteLocalRef(jstr);
}

// ── onBackendBegin ─────────────────────────────────────────────────────────

void LoggerAndroid::onBackendBegin(const std::string &name)
{
    if (!firstBackend)
        note("\n");
    firstBackend = false;

    note("Backend: " + name + "\n");
}

// ── onDeviceBegin ──────────────────────────────────────────────────────────

void LoggerAndroid::onDeviceBegin(const std::string &name,
                                  const std::string &platform,
                                  const std::string &driverVersion,
                                  const std::vector<Prop> &props,
                                  bool showPlatformLine)
{
    int deviceIndent = showPlatformLine ? 2 : 1;
    propIndent       = deviceIndent + 1;
    metricIndent     = propIndent + 1;

    if (showPlatformLine)
        writeNote(1, "Platform: " + platform);

    writeNote(deviceIndent, "Device: " + name);

    if (!driverVersion.empty())
    {
        std::string dvLabel = "Driver version";
        while (dvLabel.size() < 17)
            dvLabel += ' ';
        writeNote(propIndent, dvLabel + ": " + driverVersion);
    }

    for (const auto &prop : props)
    {
        std::string line = prop.key;
        while (line.size() < 17)
            line += ' ';
        line += ": " + prop.value;
        writeNote(propIndent, line);
    }
}

// ── onTestBegin ────────────────────────────────────────────────────────────

void LoggerAndroid::onTestBegin(const std::string & /*tag*/,
                                const std::string &display,
                                const std::string & /*unit*/)
{
    note("\n");
    writeNote(propIndent, display);
}

// ── onMetricEmitted ────────────────────────────────────────────────────────

void LoggerAndroid::onMetricEmitted(const ResultEntry &e, float value, bool /*subMetric*/)
{
    if (!recordMetricCallback)
        return;

    jstring jBackend  = jEnv->NewStringUTF(e.backend.c_str());
    jstring jPlatform = jEnv->NewStringUTF(e.platform.c_str());
    jstring jDevice   = jEnv->NewStringUTF(e.device.c_str());
    jstring jDriver   = jEnv->NewStringUTF(e.driver.c_str());
    jstring jCategory = jEnv->NewStringUTF(e.category.c_str());
    jstring jTest     = jEnv->NewStringUTF(e.test.c_str());
    jstring jMetric   = jEnv->NewStringUTF(e.metric.c_str());
    jstring jUnit     = jEnv->NewStringUTF(e.unit.c_str());

    jEnv->CallVoidMethod(
        jObj,
        recordMetricCallback,
        jBackend, jPlatform, jDevice, jDriver,
        jCategory, jTest, jMetric, jUnit,
        static_cast<jfloat>(value));

    jEnv->DeleteLocalRef(jBackend);
    jEnv->DeleteLocalRef(jPlatform);
    jEnv->DeleteLocalRef(jDevice);
    jEnv->DeleteLocalRef(jDriver);
    jEnv->DeleteLocalRef(jCategory);
    jEnv->DeleteLocalRef(jTest);
    jEnv->DeleteLocalRef(jMetric);
    jEnv->DeleteLocalRef(jUnit);
}

// ── onMetricSkipped ────────────────────────────────────────────────────────

void LoggerAndroid::onMetricSkipped(const ResultEntry &e)
{
    const char *tag = "";
    switch (e.status)
    {
    case ResultStatus::Unsupported: tag = "unsupported"; break;
    case ResultStatus::Skipped:     tag = "skipped";     break;
    case ResultStatus::Error:       tag = "error";       break;
    default: break;
    }

    std::stringstream ss;
    ss << indentStr(metricIndent) << e.metric << " : ["
       << tag << "] " << e.reason << "\n";
    note(ss.str());
}

// ── onTestSkippedAll ───────────────────────────────────────────────────────

void LoggerAndroid::onTestSkippedAll(ResultStatus status, const std::string &reason)
{
    const char *tag = "";
    switch (status)
    {
    case ResultStatus::Unsupported: tag = "unsupported"; break;
    case ResultStatus::Skipped:     tag = "skipped";     break;
    case ResultStatus::Error:       tag = "error";       break;
    default: break;
    }

    writeNote(metricIndent, std::string("[") + tag + "] " + reason);
}

// ── onTestEnd / onDeviceEnd / onBackendEnd ─────────────────────────────────

void LoggerAndroid::onTestEnd()    { /* no-op */ }
void LoggerAndroid::onDeviceEnd()  { /* no-op */ }
void LoggerAndroid::onBackendEnd() { /* no-op */ }

// ── Helpers ────────────────────────────────────────────────────────────────

std::string LoggerAndroid::indentStr(int level) const
{
    if (level <= 0)
        return "";
    return std::string(static_cast<size_t>(level) * 2, ' ');
}

void LoggerAndroid::writeNote(int level, const std::string &text)
{
    note(indentStr(level) + text + "\n");
}
