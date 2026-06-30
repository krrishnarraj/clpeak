#include "logger_android.h"
#include <sstream>

// ── onDeviceBegin → text + device_info_callback_from_c ─────────────────────

void LoggerAndroid::onDeviceBegin(const std::string &name,
                                  const std::string &platform,
                                  const std::string &driverVersion,
                                  const std::vector<Prop> &props,
                                  bool showPlatformLine,
                                  int platformIndex,
                                  int deviceIndex)
{
    // Emit the formatted device block to logcat.
    LoggerText::onDeviceBegin(name, platform, driverVersion, props,
                             showPlatformLine, platformIndex, deviceIndex);

    if (!deviceInfoCallback)
        return;

    // Build a compact JSON array of props: [{"k":"Compute units","v":"16"},...]
    std::stringstream json;
    json << "[";
    for (size_t i = 0; i < props.size(); i++)
    {
        if (i > 0) json << ",";
        json << "{\"k\":\"" << props[i].key << "\",\"v\":\"" << props[i].value << "\"}";
    }
    json << "]";

    jstring jBackend       = jEnv->NewStringUTF(curBackend.c_str());
    jstring jPlatform      = jEnv->NewStringUTF(platform.c_str());
    jstring jDevice        = jEnv->NewStringUTF(name.c_str());
    jstring jDriver        = jEnv->NewStringUTF(driverVersion.c_str());
    jstring jProps         = jEnv->NewStringUTF(json.str().c_str());
    jstring jPlatformIndex = jEnv->NewStringUTF(std::to_string(platformIndex).c_str());
    jstring jDeviceIndex   = jEnv->NewStringUTF(std::to_string(deviceIndex).c_str());

    jEnv->CallVoidMethod(jObj, deviceInfoCallback,
                         jBackend, jPlatform, jDevice, jDriver,
                         jProps, jPlatformIndex, jDeviceIndex);

    jEnv->DeleteLocalRef(jBackend);
    jEnv->DeleteLocalRef(jPlatform);
    jEnv->DeleteLocalRef(jDevice);
    jEnv->DeleteLocalRef(jDriver);
    jEnv->DeleteLocalRef(jProps);
    jEnv->DeleteLocalRef(jPlatformIndex);
    jEnv->DeleteLocalRef(jDeviceIndex);
}

// ── onTestBegin → text + remember display name for JNI ─────────────────

void LoggerAndroid::onTestBegin(const std::string &tag,
                                const std::string &display,
                                const std::string &unit)
{
    LoggerText::onTestBegin(tag, display, unit);
    curDisplay = display;
}

// ── onMetricEmitted → text + record_metric_callback_from_c ─────────────

void LoggerAndroid::onMetricEmitted(const ResultEntry &e, float value, bool subMetric)
{
    LoggerText::onMetricEmitted(e, value, subMetric);

    if (!recordMetricCallback)
        return;

    jstring jBackend  = jEnv->NewStringUTF(e.backend.c_str());
    jstring jPlatform = jEnv->NewStringUTF(e.platform.c_str());
    jstring jDevice   = jEnv->NewStringUTF(e.device.c_str());
    jstring jDriver   = jEnv->NewStringUTF(e.driver.c_str());
    jstring jCategory = jEnv->NewStringUTF(e.category.c_str());
    jstring jTest     = jEnv->NewStringUTF(e.test.c_str());
    jstring jDisplay  = jEnv->NewStringUTF(curDisplay.c_str());
    jstring jMetric   = jEnv->NewStringUTF(e.metric.c_str());
    jstring jUnit     = jEnv->NewStringUTF(e.unit.c_str());
    jstring jStatus   = jEnv->NewStringUTF("ok");
    jstring jReason   = jEnv->NewStringUTF("");

    jEnv->CallVoidMethod(jObj, recordMetricCallback,
                         jBackend, jPlatform, jDevice, jDriver,
                         jCategory, jTest, jDisplay, jMetric, jUnit,
                         static_cast<jfloat>(value),
                         jStatus, jReason);

    jEnv->DeleteLocalRef(jBackend);
    jEnv->DeleteLocalRef(jPlatform);
    jEnv->DeleteLocalRef(jDevice);
    jEnv->DeleteLocalRef(jDriver);
    jEnv->DeleteLocalRef(jCategory);
    jEnv->DeleteLocalRef(jTest);
    jEnv->DeleteLocalRef(jDisplay);
    jEnv->DeleteLocalRef(jMetric);
    jEnv->DeleteLocalRef(jUnit);
    jEnv->DeleteLocalRef(jStatus);
    jEnv->DeleteLocalRef(jReason);
}

// ── onMetricSkipped → text + record_metric_callback_from_c ─────────────

void LoggerAndroid::onMetricSkipped(const ResultEntry &e)
{
    LoggerText::onMetricSkipped(e);

    if (!recordMetricCallback)
        return;

    const char *statusStr = "";
    switch (e.status)
    {
    case ResultStatus::Unsupported: statusStr = "unsupported"; break;
    case ResultStatus::Skipped:     statusStr = "skipped";     break;
    case ResultStatus::Error:       statusStr = "error";       break;
    default: break;
    }

    jstring jBackend  = jEnv->NewStringUTF(e.backend.c_str());
    jstring jPlatform = jEnv->NewStringUTF(e.platform.c_str());
    jstring jDevice   = jEnv->NewStringUTF(e.device.c_str());
    jstring jDriver   = jEnv->NewStringUTF(e.driver.c_str());
    jstring jCategory = jEnv->NewStringUTF(e.category.c_str());
    jstring jTest     = jEnv->NewStringUTF(e.test.c_str());
    jstring jDisplay  = jEnv->NewStringUTF(curDisplay.c_str());
    jstring jMetric   = jEnv->NewStringUTF(e.metric.c_str());
    jstring jUnit     = jEnv->NewStringUTF(e.unit.c_str());
    jstring jStatus   = jEnv->NewStringUTF(statusStr);
    jstring jReason   = jEnv->NewStringUTF(e.reason.c_str());

    jEnv->CallVoidMethod(jObj, recordMetricCallback,
                         jBackend, jPlatform, jDevice, jDriver,
                         jCategory, jTest, jDisplay, jMetric, jUnit,
                         0.0f,
                         jStatus, jReason);

    jEnv->DeleteLocalRef(jBackend);
    jEnv->DeleteLocalRef(jPlatform);
    jEnv->DeleteLocalRef(jDevice);
    jEnv->DeleteLocalRef(jDriver);
    jEnv->DeleteLocalRef(jCategory);
    jEnv->DeleteLocalRef(jTest);
    jEnv->DeleteLocalRef(jDisplay);
    jEnv->DeleteLocalRef(jMetric);
    jEnv->DeleteLocalRef(jUnit);
    jEnv->DeleteLocalRef(jStatus);
    jEnv->DeleteLocalRef(jReason);
}
