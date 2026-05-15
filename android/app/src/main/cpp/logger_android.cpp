#include "logger_android.h"
#include <sstream>

// ---- print → JNI ---------------------------------------------------------

void LoggerAndroid::print(std::string str)
{
    jstring jstr = jEnv->NewStringUTF(str.c_str());
    jEnv->CallVoidMethod(jObj, printCallback, jstr);
    jEnv->DeleteLocalRef(jstr);
}

void LoggerAndroid::print(double val)
{
    std::stringstream ss;
    ss << std::setprecision(2) << std::fixed << val;
    jstring jstr = jEnv->NewStringUTF(ss.str().c_str());
    jEnv->CallVoidMethod(jObj, printCallback, jstr);
    jEnv->DeleteLocalRef(jstr);
}

void LoggerAndroid::print(float val)
{
    std::stringstream ss;
    ss << std::setprecision(2) << std::fixed << val;
    jstring jstr = jEnv->NewStringUTF(ss.str().c_str());
    jEnv->CallVoidMethod(jObj, printCallback, jstr);
    jEnv->DeleteLocalRef(jstr);
}

void LoggerAndroid::print(int val)
{
    std::stringstream ss;
    ss << val;
    jstring jstr = jEnv->NewStringUTF(ss.str().c_str());
    jEnv->CallVoidMethod(jObj, printCallback, jstr);
    jEnv->DeleteLocalRef(jstr);
}

void LoggerAndroid::print(unsigned int val)
{
    std::stringstream ss;
    ss << val;
    jstring jstr = jEnv->NewStringUTF(ss.str().c_str());
    jEnv->CallVoidMethod(jObj, printCallback, jstr);
    jEnv->DeleteLocalRef(jstr);
}

// ---- onMetricEmitted → JNI -----------------------------------------------

void LoggerAndroid::onMetricEmitted(const ResultEntry &e, float value)
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
