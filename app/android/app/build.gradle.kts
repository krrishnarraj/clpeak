plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

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
            // A LiteRT NPU dispatch library staged by tool/fetch_litert_npu.sh
            // and the runtime's own .so must both stay uncompressed and
            // page-aligned to be dlopen'd out of the APK.
            useLegacyPackaging = false
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
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
