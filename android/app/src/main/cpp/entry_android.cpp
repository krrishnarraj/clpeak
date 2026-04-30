#include <clpeak.h>
#include "jni_entry.h"

#ifdef ENABLE_VULKAN
#include <vk_peak.h>
#endif

#define PRINT_CALLBACK         "print_callback_from_c"
#define RECORD_METRIC_CALLBACK "record_metric_callback_from_c"

static void wireLoggerToJni(logger *lg, JNIEnv *jniEnv, jobject *jObj, jclass cls)
{
  lg->jEnv = jniEnv;
  lg->jObj = jObj;
  lg->printCallback = jniEnv->GetMethodID(cls,
      PRINT_CALLBACK, "(Ljava/lang/String;)V");
  lg->recordMetricCallback = jniEnv->GetMethodID(cls,
      RECORD_METRIC_CALLBACK,
      "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;"
      "Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;F)V");
}

jint JNICALL Java_kr_clpeak_BenchmarkRepository_launchClpeak(JNIEnv *_jniEnv,
                                                             jobject _jObject, jint argc, jobjectArray _argv)
{
  char **argv;
  clPeak clObj;

  argv = (char **)malloc(sizeof(char *) * argc);

  for (int i = 0; i < argc; i++)
  {
    jstring strObj = (jstring)_jniEnv->GetObjectArrayElement(_argv, i);
    argv[i] = (char *)_jniEnv->GetStringUTFChars(strObj, 0);
  }
  clObj.parseArgs(argc, argv);

  jclass cls = _jniEnv->GetObjectClass(_jObject);
  wireLoggerToJni(clObj.log.get(), _jniEnv, &_jObject, cls);
  int clStatus = clObj.runAll();

#ifdef ENABLE_VULKAN
  {
    vkPeak vkObj;
    vkObj.parseArgs(argc, argv);
    wireLoggerToJni(vkObj.log.get(), _jniEnv, &_jObject, cls);
    vkObj.runAll();
  }
#endif

  if (argv)
  {
    free(argv);
  }

  return clStatus;
}

void Java_kr_clpeak_MainActivity_nativeSetenv(JNIEnv *jniEnv,
                                              jobject _jObj, jstring key, jstring value)
{
  (void)_jObj;
  setenv((char *)jniEnv->GetStringUTFChars(key, 0),
         (char *)jniEnv->GetStringUTFChars(value, 0), 1);
}

#include <version.h>

jstring Java_kr_clpeak_AboutBottomSheet_nativeGetVersion(JNIEnv *jniEnv, jobject _jObj)
{
  (void)_jObj;
  return jniEnv->NewStringUTF(CLPEAK_VERSION_STR);
}
