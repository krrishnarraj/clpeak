#ifndef LOGGER_ANDROID_HPP
#define LOGGER_ANDROID_HPP

#include <common/logger.h>
#include <jni.h>

// Android logger: print() → JNI print callback, onMetricEmitted() → JNI record callback.
class LoggerAndroid : public logger
{
public:
    JNIEnv   *jEnv = nullptr;
    jobject   jObj = nullptr;
    jmethodID printCallback = nullptr;
    jmethodID recordMetricCallback = nullptr;

    using logger::logger;  // inherit constructor

    void print(std::string str) override;
    void print(double val) override;
    void print(float val) override;
    void print(int val) override;
    void print(unsigned int val) override;

protected:
    void onMetricEmitted(const ResultEntry &e, float value) override;
};

#endif // LOGGER_ANDROID_HPP
