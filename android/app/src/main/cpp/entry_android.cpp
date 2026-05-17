#include <opencl/cl_peak.h>
#include <common/peak.h>
#include <common/options.h>
#include "logger_android.h"
#include "jni_entry.h"

#ifdef ENABLE_VULKAN
#include <vulkan/vk_peak.h>
#endif

#define RECORD_METRIC_CALLBACK "record_metric_callback_from_c"
#define DEVICE_INFO_CALLBACK   "device_info_callback_from_c"

static void wireLoggerToJni(LoggerAndroid *lg, JNIEnv *jniEnv, jobject jObj, jclass cls)
{
    lg->jEnv = jniEnv;
    lg->jObj = jObj;
    lg->recordMetricCallback = jniEnv->GetMethodID(cls,
        RECORD_METRIC_CALLBACK,
        "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;"
        "Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;F"
        "Ljava/lang/String;Ljava/lang/String;)V");
    lg->deviceInfoCallback = jniEnv->GetMethodID(cls,
        DEVICE_INFO_CALLBACK,
        "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;"
        "Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)V");
}

jint JNICALL Java_kr_clpeak_BenchmarkRepository_launchClpeak(JNIEnv *_jniEnv,
                                                             jobject _jObject, jint argc, jobjectArray _argv)
{
    char **argv = (char **)malloc(sizeof(char *) * argc);
    for (int i = 0; i < argc; i++)
    {
        jstring strObj = (jstring)_jniEnv->GetObjectArrayElement(_argv, i);
        argv[i] = (char *)_jniEnv->GetStringUTFChars(strObj, 0);
    }

    CliOptions opts;
    parseCliOptions(argc, argv, opts);

    jclass cls = _jniEnv->GetObjectClass(_jObject);

    int clStatus = 0;
    if (!opts.skipOpenCL)
    {
        clPeak clObj;
        clObj.log.reset(new LoggerAndroid());
        clObj.applyOptions(opts);
        wireLoggerToJni(static_cast<LoggerAndroid *>(clObj.log.get()), _jniEnv, _jObject, cls);
        clStatus = clObj.runAll();
    }

#ifdef ENABLE_VULKAN
    if (!opts.skipVulkan)
    {
        vkPeak vkObj;
        vkObj.log.reset(new LoggerAndroid());
        vkObj.applyOptions(opts);
        wireLoggerToJni(static_cast<LoggerAndroid *>(vkObj.log.get()), _jniEnv, _jObject, cls);
        int vkStatus = vkObj.runAll();
        clStatus |= vkStatus;
    }
#endif

    if (argv)
        free(argv);

    return clStatus;
}

#include <version.h>

jstring Java_kr_clpeak_AboutBottomSheet_nativeGetVersion(JNIEnv *jniEnv, jobject _jObj)
{
    (void)_jObj;
    return jniEnv->NewStringUTF(CLPEAK_VERSION_STR);
}
