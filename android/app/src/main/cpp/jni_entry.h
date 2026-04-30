/* Header for Android JNI entry points */
#include <jni.h>

#ifndef _Included_kr_clpeak_BenchmarkRepository
#define _Included_kr_clpeak_BenchmarkRepository
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     kr.clpeak.BenchmarkRepository
 * Method:    launchClpeak
 * Signature: (I[Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_kr_clpeak_BenchmarkRepository_launchClpeak
(JNIEnv *, jobject, jint, jobjectArray);

#ifdef __cplusplus
}
#endif
#endif


#ifndef _Included_kr_clpeak_MainActivity
#define _Included_kr_clpeak_MainActivity
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     kr.clpeak.AboutBottomSheet
 * Method:    nativeGetVersion
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_kr_clpeak_AboutBottomSheet_nativeGetVersion
(JNIEnv *, jobject);

#ifdef __cplusplus
}
#endif
#endif
