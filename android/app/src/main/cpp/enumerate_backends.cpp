// Thin JNI wrapper around the shared backend enumeration in src/inventory.cpp.
// All real enumeration work lives in core (clpeak.cpp / vk_peak.cpp) so the
// desktop --list-devices flow and this Kotlin-facing JSON come from the same
// source of truth.

#include <jni.h>

#include <inventory.h>
#include <options.h>

extern "C" JNIEXPORT jstring JNICALL
Java_kr_clpeak_BenchmarkRepository_nativeEnumerateBackends(JNIEnv *env, jobject)
{
  CliOptions opts;
  return env->NewStringUTF(inventoryToJson(enumerateAllBackends(opts)).c_str());
}
