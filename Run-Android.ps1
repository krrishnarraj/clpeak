if (-not(Test-Path "build-android-aarch64")) {
    New-Item "build-android-aarch64" -ItemType Directory
}

$NdkHome = $null
if ($env:ANDROID_NDK -ne $null) {
    $NdkHome = $env:ANDROID_NDK
}
if ($env:ANDROID_NDK_HOME -ne $null) {
    $NdkHome = $env:ANDROID_NDK_HOME
}

if ($NdkHome -eq $null) {
    Write-Host "Couldn't find `Android_SDK` in environment variables. Is NDK installed?"
    return -1
}

Push-Location "build-android-aarch64"
cmake -DCMAKE_TOOLCHAIN_FILE="$NdkHome/build/cmake/android.toolchain.cmake" -DANDROID_ABI="arm64-v8a" -DANDROID_PLATFORM=android-21 -G "Ninja" ..
cmake --build .
Pop-Location

adb reconnect offline
adb push ./build-android-aarch64/clpeak /data/local/tmp/clpeak
adb shell chmod 777 /data/local/tmp/clpeak
adb shell "cd /data/local/tmp && ./clpeak"
