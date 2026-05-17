#ifndef LOGGER_ANDROID_HPP
#define LOGGER_ANDROID_HPP

#include <common/logger.h>
#include <jni.h>

// ── Android logger ─────────────────────────────────────────────────────────
//
// Forwards structured data to Java via two JNI callbacks:
//   - record_metric_callback_from_c  — Ok + skipped metrics (extended with
//                                      status and reason fields)
//   - device_info_callback_from_c    — device properties (once per device)
//
// Text formatting is handled entirely by the Kotlin UI layer.

class LoggerAndroid : public logger
{
public:
    JNIEnv   *jEnv = nullptr;
    jobject   jObj = nullptr;
    jmethodID recordMetricCallback = nullptr;
    jmethodID deviceInfoCallback   = nullptr;

    using logger::logger;

    void note(const std::string &msg) override;

protected:
    void onBackendBegin(const std::string &name) override           { (void)name; }
    void onDeviceBegin(const std::string &name,
                       const std::string &platform,
                       const std::string &driverVersion,
                       const std::vector<Prop> &props,
                       bool showPlatformLine,
                       int platformIndex,
                       int deviceIndex) override;
    void onTestBegin(const std::string &tag,
                     const std::string &display,
                     const std::string &unit) override {
        (void)tag; (void)unit;
        curDisplay = display;
    }
    void onMetricEmitted(const ResultEntry &e,
                         float value,
                         bool subMetric) override;
    void onMetricSkipped(const ResultEntry &e) override;
    void onTestSkippedAll(ResultStatus status,
                          const std::string &reason) override       { (void)status; (void)reason; }
    void onTestEnd() override                                       {}
    void onDeviceEnd() override                                     {}
    void onBackendEnd() override                                    {}

private:
    std::string curDisplay;
};

#endif // LOGGER_ANDROID_HPP
