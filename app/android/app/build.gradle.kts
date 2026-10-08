plugins {
    id("com.android.application")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

// NPU libraries staged by tools/fetch_android_npu.sh -- every vendor's, or
// none -- into the NPU runtime modules (../npu/), which
// settings.gradle.kts includes as it finds them staged: there is no switch
// beside the staging.  Each is a dynamic feature module Play installs with
// the app on the chipsets of its device groups only
// (device_targeting_configuration.xml), so a phone downloads its own
// vendor's libraries and nobody else's.  Qualcomm's set serves two stacks,
// one QNN runtime under both:
//  - LiteRT, through its Qualcomm shims (libLiteRtDispatch_Qualcomm.so and
//    the compiler plugin), built against QAIRT 2.47 for LiteRT 2.2.0;
//  - ONNX Runtime, through Qualcomm's plugin execution provider
//    (libonnxruntime_providers_qnn.so, which the app registers on the
//    packaged onnxruntime at run time by its bare soname, see
//    SettingsService.effectiveOnnxEpLibraries), validated by Qualcomm
//    against QAIRT 2.50, the runtime staged.
// It is 74 MB compressed in all: qualcomm_runtime (45 MB, libQnnHtpPrepare
// alone 35) and one 4-5 MB Skel/Stub module per Hexagon generation
// v68..v81, of which a Snapdragon gets its own.  LiteRT finds a dispatch
// library by listing the directory it is told (litert_dispatch.cc), as the
// backend does to pick a vendor, and an APK's internal lib/ path is not a
// directory anyone can list, so an app carrying NPU libraries has them
// extracted at install -- the modules' too, into the one lib dir of the
// app; the Qualcomm runtime needs that anyway (see below).
val npuDir = rootProject.file("npu")
val npuModules = rootProject.subprojects.filter { it.projectDir.parentFile == npuDir }
val npuStaged = npuModules.isNotEmpty()

// Why the NPU libraries staged cannot reach an NPU as Play would deliver
// them, or null.  The shims and the ONNX plugin run nothing on their own:
// they open the QNN runtime at run time (libQnnHtp and libQnnSystem,
// through them libQnnHtpPrepare and the Skel/Stub of the phone's Hexagon
// generation), and LiteRT's compiler plugin is also linked against
// libQnnIr.so and libQnnSaver.so -- it never calls either, but Android's
// linker refuses a library one of whose DT_NEEDED entries it cannot find,
// and without the plugin LiteRT compiles nothing for the NPU.  A partial
// set builds an APK that installs, runs and lists no Hexagon NPU on any
// Snapdragon (the Play build of 3.0.1 on a Galaxy S24 Ultra carried the
// shims alone), so the build stops instead.  So does a set an older fetch
// script staged into the base, which every phone would download.
val npuStagingProblem: String? = run {
    val legacy = file("src/main/jniLibs/arm64-v8a").list()?.filter { it.endsWith(".so") } ?: emptyList()
    val common = npuDir.resolve("qualcomm_runtime/jniLibs/arm64-v8a").list()?.toSet() ?: emptySet()
    val qualcomm = npuModules.any { it.name.startsWith("qualcomm_runtime") }
    val missing = listOf(
        "libQnnHtp.so", "libQnnSystem.so", "libQnnHtpPrepare.so", "libQnnIr.so", "libQnnSaver.so",
    ).filter { it !in common }
    if (legacy.isNotEmpty())
        "app/src/main/jniLibs/arm64-v8a holds ${legacy.size} NPU libraries an older " +
            "tools/fetch_android_npu.sh staged into the base, which every phone would download. " +
            "Stage them again with tools/fetch_android_npu.sh, which moves them into npu/."
    else if (qualcomm && missing.isNotEmpty())
        "Qualcomm's NPU libraries in npu/qualcomm_runtime are incomplete " +
            "(${missing.joinToString()} missing), so no Snapdragon would list a Hexagon NPU. " +
            "Stage them again with tools/fetch_android_npu.sh."
    else null
}

android {
    namespace = "kr.clpeak"
    compileSdk = flutter.compileSdkVersion
    // NDK r30 (clang 21), not flutter.ndkVersion (r28c, clang 19): clang 19
    // fails the CPU backend's SME probes (src/cpu/CMakeLists.txt), so an APK
    // built with it carries no SME kernels and an SME phone (Snapdragon 8
    // Elite Gen 5) reports those matrix rows unsupported.  Drop the pin once
    // Flutter's default reaches r30.
    ndkVersion = "30.0.16248370"

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
            // that same directory -- both need real files.  The bundle's
            // setting covers its feature modules too.
            useLegacyPackaging = npuStaged
        }
    }

    // The NPU runtime modules staged, and the chipsets Play delivers each
    // to: the whitelist, kept current by hand (app/AGENTS.md).
    dynamicFeatures += npuModules.map { it.path }
    bundle {
        deviceTargetingConfig.set(file("device_targeting_configuration.xml"))
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
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")

    // LiteRT: jni/<abi>/libLiteRt.so plus its OpenCL GPU accelerator
    // (libLiteRtClGlAccelerator.so), 8.6 MB for arm64-v8a.  The AAR's own
    // manifest carries the `uses-native-library` declarations the runtime
    // needs on Android 12+ (libOpenCL, Qualcomm's libcdsprpc, the Google
    // Tensor and MediaTek NPU system libraries) and the merger brings them
    // into ours.  NPU dispatch libraries are not on Maven: see
    // tools/fetch_android_npu.sh, which stages them into the NPU modules (../npu/).
    // The AAR's Kotlin/Java surface (and the `litert-api` it depends on,
    // which declares the same namespace and trips AGP 9's uniqueness
    // check) is unused: only the .so files are wanted.
    implementation("com.google.ai.edge.litert:litert:2.2.0") {
        exclude(group = "com.google.ai.edge.litert", module = "litert-api")
    }
}

// Ahead of every variant's build, so a build that could not reach the NPU
// it carries libraries for stops before compiling anything, saying why
// (npuStagingProblem above).
val checkNpuStaging = tasks.register("checkNpuStaging") {
    val problem = npuStagingProblem
    doLast { problem?.let { throw GradleException(it) } }
}
tasks.named("preBuild") { dependsOn(checkNpuStaging) }

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}
