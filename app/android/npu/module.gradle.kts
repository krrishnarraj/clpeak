// The build of every NPU runtime module: one directory beside this file per
// module, an AndroidManifest.xml naming the device groups Play installs it
// on (../app/device_targeting_configuration.xml) and, staged by
// tools/fetch_android_npu.sh and ignored by git, jniLibs/arm64-v8a/.
// settings.gradle.kts includes the modules with libraries staged and points
// each at this file.  A module carries native libraries and nothing else:
// the staged arm64 ones, and an empty one of its own for every ABI of the
// base (stub/CMakeLists.txt says why).
plugins {
    id("com.android.dynamic-feature")
}

// The app's SDK levels, NDK (the one that strips the libraries) and ABIs --
// Flutter's, set while :app evaluates, which settles before any module
// does (evaluationDependsOn in ../build.gradle.kts) -- so a module never
// drifts from the base it installs beside.
val app = project(":app").extensions.getByType(com.android.build.api.dsl.ApplicationExtension::class.java)

android {
    namespace = "kr.clpeak.npu.${project.name}"
    compileSdk = app.compileSdk
    ndkVersion = app.ndkVersion

    defaultConfig {
        minSdk = app.defaultConfig.minSdk
        ndk {
            abiFilters += app.defaultConfig.ndk.abiFilters
        }
        externalNativeBuild {
            cmake {
                arguments += "-DCLPEAK_NPU_STUB=clpeak_npu_${project.name}"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = rootProject.file("npu/stub/CMakeLists.txt")
        }
    }

    // Extracted at install like the base's (../app/build.gradle.kts says
    // why).  bundletool decides it per module: one left at the default
    // ships its libraries stored with extractNativeLibs=false.
    packaging {
        jniLibs {
            useLegacyPackaging = app.packaging.jniLibs.useLegacyPackaging
        }
    }

    // Every build type of the app, Flutter's `profile` included: a feature
    // module resolves against the base variant of the same name.
    buildTypes {
        create("profile") {
            initWith(getByName("debug"))
        }
    }

    sourceSets {
        getByName("main") {
            manifest.srcFile("AndroidManifest.xml")
            jniLibs.srcDirs("jniLibs")
        }
    }
}

dependencies {
    implementation(project(":app"))
}
