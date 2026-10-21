pluginManagement {
    val flutterSdkPath =
        run {
            val properties = java.util.Properties()
            file("local.properties").inputStream().use { properties.load(it) }
            val flutterSdkPath = properties.getProperty("flutter.sdk")
            require(flutterSdkPath != null) { "flutter.sdk not set in local.properties" }
            flutterSdkPath
        }

    includeBuild("$flutterSdkPath/packages/flutter_tools/gradle")

    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

plugins {
    id("dev.flutter.flutter-plugin-loader") version "1.0.0"
    id("com.android.application") version "9.0.1" apply false
    id("com.android.dynamic-feature") version "9.0.1" apply false
    id("org.jetbrains.kotlin.android") version "2.3.20" apply false
}

include(":app")

// The NPU runtime modules: each directory of npu/ that
// tools/fetch_android_npu.sh staged libraries into, built by
// npu/module.gradle.kts.  Nothing staged, no modules (the CI build).
file("npu").listFiles()
    ?.filter { dir -> dir.resolve("jniLibs/arm64-v8a").list()?.any { it.endsWith(".so") } == true }
    ?.sortedBy { it.name }
    ?.forEach { dir ->
        include(":${dir.name}")
        project(":${dir.name}").projectDir = dir
        project(":${dir.name}").buildFileName = "../module.gradle.kts"
    }
