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
            )
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

dependencies {
    // Packaged for its jni/<abi>/libonnxruntime.so; the Java API that comes
    // with it is unused (clpeak talks to the C API through the FFI library).
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.29.0")
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
