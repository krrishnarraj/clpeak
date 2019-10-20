#include <clpeak.h>
#include <jni_entry.h>

#define PRINT_CALLBACK "print_callback_from_c"

jint JNICALL Java_kr_clpeak_jni_1connect_launchClpeak(JNIEnv *_jniEnv,
                                                      jobject _jObject, jint argc, jobjectArray _argv)
{
  char **argv;
  clPeak clObj;

  argv = (char **)malloc(sizeof(char *) * argc);

  // Convert jobjectArray to string array
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
  clObj.log->printCallback = _jniEnv->GetMethodID(_jniEnv->GetObjectClass(_jObject),
                                                  PRINT_CALLBACK, "(Ljava/lang/String;)V");

  return clObj.runAll();
}

void Java_kr_clpeak_MainActivity_setenv(JNIEnv *jniEnv,
                                        jobject _jObj, jstring key, jstring value)
{
  setenv((char *)jniEnv->GetStringUTFChars(key, 0),
         (char *)jniEnv->GetStringUTFChars(value, 0), 1);
}
