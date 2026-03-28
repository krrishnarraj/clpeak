#include <clpeak.h>
#include <jni_entry.h>

#define PRINT_CALLBACK         "print_callback_from_c"
#define RECORD_METRIC_CALLBACK "record_metric_callback_from_c"

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

  if (argv)
  {
    free(argv);
  }

  clObj.log->jEnv = _jniEnv;
  clObj.log->jObj = &(_jObject);

  jclass cls = _jniEnv->GetObjectClass(_jObject);

  clObj.log->printCallback = _jniEnv->GetMethodID(cls,
      PRINT_CALLBACK, "(Ljava/lang/String;)V");

  clObj.log->recordMetricCallback = _jniEnv->GetMethodID(cls,
      RECORD_METRIC_CALLBACK,
      "(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;"
      "Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;F)V");

  return clObj.runAll();
}

void Java_kr_clpeak_MainActivity_nativeSetenv(JNIEnv *jniEnv,
                                              jobject _jObj, jstring key, jstring value)
{
  (void)_jObj;
  setenv((char *)jniEnv->GetStringUTFChars(key, 0),
         (char *)jniEnv->GetStringUTFChars(value, 0), 1);
}
