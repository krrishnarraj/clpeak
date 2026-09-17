plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

// Qualcomm's QNN (QAIRT) runtime for the Hexagon NPU, opt-in:
// `clpeakQnn=true` in android/gradle.properties (which `flutter build` and
// `flutter run` read), or `-PclpeakQnn=true` on a direct gradlew call.
//
// It is 67 MB compressed / 200 MB installed (libQnnHtpPrepare.so alone is
// 86 MB; one Skel per Hexagon generation v68..v81), only a Snapdragon can
// use it, and Qualcomm publishes it on Maven Central under its AI Hub Model
// License (com.qualcomm.qti:qnn-runtime), so it is not in the default APK.
// Two stacks reach the NPU through it, and the flag brings both:
//  - LiteRT, through its Qualcomm shims (libLiteRtDispatch_Qualcomm.so and
//    the compiler plugin, from tool/fetch_litert_npu.sh) which must sit
//    beside it -- built against QAIRT 2.47 for LiteRT 2.2.0 (the release's
//    fetch_qualcomm_library.sh names it);
//  - ONNX Runtime, through Qualcomm's plugin execution provider
//    (com.qualcomm.qti:onnxruntime-android-qnn, 4 MB: one
//    libonnxruntime_providers_qnn.so that a stock onnxruntime-android 1.24.1+
//    registers at run time; the app does so by its bare soname, see
//    SettingsService.effectiveOnnxEpLibraries) -- validated by Qualcomm
//    against QAIRT 2.50.
// One QNN runtime serves both: 2.50, the newer of the two.  LiteRT's
// QnnManager accepts a runtime whose API minor version is newer than the
// one its shims were built against, with a warning (a major mismatch is
// refused); the ONNX plugin gets exactly what it was tested with.
// Unverified on a Snapdragon: the first tester's run is the proof.
val clpeakQnn = (project.findProperty("clpeakQnn")?.toString() ?: "false") == "true"

// NPU shims staged by tool/fetch_litert_npu.sh (any vendor, or all: the
// backend picks the SoC's vendor at launch and stages its shims in a
// directory of their own).  LiteRT finds a dispatch library by listing the
// directory it is told (litert_dispatch.cc), as the backend does to pick,
// and an APK's internal lib/ path is not a directory anyone can list, so an
// app carrying shims has its native libraries extracted at install; the
// Qualcomm runtime needs that anyway (see below).
val clpeakNpuStaged = file("src/main/jniLibs/arm64-v8a").isDirectory

android {
    namespace = "kr.clpeak"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    defaultConfig {
        // Keeps the identity of the retired native app (Play Store update path).
        applicationId = "kr.clpeak"
        // The native backends (OpenCL stub dlopen, Vulkan 1.3 expectations)
        // assume Android 13+, matching the retired app.
        minSdk = 33
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    // clpeak_ffi native bridge: same CMake superproject layout as the other
    // platforms — see src/ffi/android/CMakeLists.txt.
    externalNativeBuild {
        cmake {
            path = file("../../../src/ffi/android/CMakeLists.txt")
        }
    }

    // ONNX Runtime ships as a real .so on Android, so the ONNX backend loads
    // it the same way it does on desktop -- see src/onnx/onnx_runtime.cpp.
    //
    // Bundled for arm64-v8a (devices) and x86_64 (emulator / Chromebooks).
    // With AAB delivery Play serves a split APK per ABI, so per-device
    // download size stays bounded; x86/armeabi-v7a are excluded as legacy
    // 32-bit ABIs (minSdk 33 makes a 32-bit-only handset a rounding error).
    // The Java API that came with the AAR goes too -- clpeak talks to the C
    // API through the FFI library.
    //
    // On the excluded ABIs the backend simply reports itself unavailable, and
    // the settings screen can still point it at a runtime by path.
    packaging {
        jniLibs {
            excludes += setOf(
                "lib/armeabi-v7a/libonnxruntime.so",
                "lib/x86/libonnxruntime.so",
                "**/libonnxruntime4j_jni.so",
                "lib/armeabi-v7a/libLiteRt.so",
                "lib/armeabi-v7a/libLiteRtClGlAccelerator.so",
            )
            // Without NPU libraries the .so files stay uncompressed and
            // page-aligned in the APK and are dlopen'd from there.  With
            // them, they are extracted at install: LiteRT lists a directory
            // to find its dispatch shim, and the Qualcomm runtime's
            // Hexagon-side libraries (libQnnHtpV*Skel.so) are opened by the
            // DSP's loader from ADSP_LIBRARY_PATH, which LiteRT points at
            // that same directory -- both need real files.
            useLegacyPackaging = clpeakQnn || clpeakNpuStaged
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
            // R8 rules for the runtime AARs whose Java surface is unused.
            proguardFile("proguard-rules.pro")
        }
    }
}

dependencies {
    // Packaged for its jni/<abi>/libonnxruntime.so; the Java API that comes
    // with it is unused (clpeak talks to the C API through the FFI library).
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.29.0")

    // LiteRT: jni/<abi>/libLiteRt.so plus its OpenCL GPU accelerator
    // (libLiteRtClGlAccelerator.so), 8.6 MB for arm64-v8a.  The AAR's own
    // manifest carries the `uses-native-library` declarations the runtime
    // needs on Android 12+ (libOpenCL, Qualcomm's libcdsprpc, the Google
    // Tensor and MediaTek NPU system libraries) and the merger brings them
    // into ours.  NPU dispatch libraries are not on Maven: see
    // tool/fetch_litert_npu.sh, which stages them under src/main/jniLibs.
    // The AAR's Kotlin/Java surface (and the `litert-api` it depends on,
    // which declares the same namespace and trips AGP 9's uniqueness
    // check) is unused: only the .so files are wanted.
    implementation("com.google.ai.edge.litert:litert:2.2.0") {
        exclude(group = "com.google.ai.edge.litert", module = "litert-api")
    }

    // See `clpeakQnn` at the top of this file.
    if (clpeakQnn) {
        implementation("com.qualcomm.qti:qnn-runtime:2.50.0")
        implementation("com.qualcomm.qti:onnxruntime-android-qnn:2.6.0")
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
