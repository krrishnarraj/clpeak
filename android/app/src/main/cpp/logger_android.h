#ifndef LOGGER_ANDROID_HPP
#define LOGGER_ANDROID_HPP

#include <common/logger.h>
#include <jni.h>

// ── Android logger ─────────────────────────────────────────────────────────
//
// Forwards all output to Java via JNI callbacks:
//   - Text messages (notes, headers, skip reasons) → print_callback_from_c
//   - Structured metric records → record_metric_callback_from_c
//
// JNI wiring must be performed after construction via wireJni().

class LoggerAndroid : public logger
{
public:
    JNIEnv   *jEnv = nullptr;
    jobject   jObj = nullptr;
    jmethodID printCallback = nullptr;
    jmethodID recordMetricCallback = nullptr;

    using logger::logger;  // inherit constructor

    void note(const std::string &msg) override;

protected:
    void onBackendBegin(const std::string &name) override;
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
    void onTestSkippedAll(ResultStatus status,
                          const std::string &reason) override;
    void onTestEnd() override;
    void onDeviceEnd() override;
    void onBackendEnd() override;

private:
    bool firstBackend = true;
    int  propIndent   = 0;    // indent for device properties / test headers
    int  metricIndent = 0;    // indent for metric lines

    std::string indentStr(int level) const;
    void writeNote(int level, const std::string &text);
};

#endif // LOGGER_ANDROID_HPP
