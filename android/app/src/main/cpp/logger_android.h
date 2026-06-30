#ifndef LOGGER_ANDROID_HPP
#define LOGGER_ANDROID_HPP

#include <common/logger_text.h>
#include <android/log.h>
#include <jni.h>
#include <ostream>
#include <streambuf>
#include <string>

// ── std::ostream → logcat ──────────────────────────────────────────────────
//
// The shared text formatter (LoggerText) writes to a std::ostream.  This
// streambuf buffers characters until a newline (or an explicit flush) and
// forwards each completed line to logcat under a fixed tag, so the output is
// filterable in Android Studio / `adb logcat` (e.g. `adb logcat -s clpeak`).

class LogcatStreambuf : public std::streambuf
{
public:
    explicit LogcatStreambuf(const char *tag, int priority = ANDROID_LOG_INFO)
        : tag(tag), priority(priority) {}

protected:
    int overflow(int ch) override
    {
        if (ch == traits_type::eof())
            return ch;
        if (ch == '\n')
            emitLine();
        else
            buffer += static_cast<char>(ch);
        return ch;
    }

    // Flush (std::flush / ostream::flush()) emits whatever has accumulated.
    // The text logger only flushes at line boundaries, so this never splits a
    // metric line mid-way — it just lets newline-less notes reach logcat.
    int sync() override
    {
        if (!buffer.empty())
            emitLine();
        return 0;
    }

private:
    void emitLine()
    {
        __android_log_write(priority, tag.c_str(), buffer.c_str());
        buffer.clear();
    }

    std::string tag;
    int         priority;
    std::string buffer;
};

// Owns the logcat streambuf + ostream.  Declared as a separate base so it is
// constructed *before* the LoggerText base (base classes initialise in
// declaration order), letting us hand the ostream to LoggerText's constructor.
class LogcatChannel
{
protected:
    LogcatStreambuf logcatBuf{"clpeak"};
    std::ostream    logcatOut{&logcatBuf};
};

// ── Android logger ─────────────────────────────────────────────────────────
//
// Reuses the shared text formatter (LoggerText) pointed at logcat, so the text
// is identical to the CLI app, and additionally forwards structured data to the
// Kotlin UI via two JNI callbacks:
//   - record_metric_callback_from_c  — Ok + skipped metrics (extended with
//                                      status and reason fields)
//   - device_info_callback_from_c    — device properties (once per device)

class LoggerAndroid : private LogcatChannel, public LoggerText
{
public:
    JNIEnv   *jEnv = nullptr;
    jobject   jObj = nullptr;
    jmethodID recordMetricCallback = nullptr;
    jmethodID deviceInfoCallback   = nullptr;

    LoggerAndroid() : LogcatChannel(), LoggerText(logcatOut) {}
    ~LoggerAndroid() override { logcatOut.flush(); }

protected:
    void onDeviceBegin(const std::string &name,
                       const std::string &platform,
                       const std::string &driverVersion,
                       const std::vector<Prop> &props,
                       bool showPlatformLine,
                       int platformIndex,
                       int deviceIndex) override;
    void onTestBegin(const std::string &tag,
                     const std::string &display,
                     const std::string &unit) override;
    void onMetricEmitted(const ResultEntry &e,
                         float value,
                         bool subMetric) override;
    void onMetricSkipped(const ResultEntry &e) override;

private:
    std::string curDisplay;
};

#endif // LOGGER_ANDROID_HPP
