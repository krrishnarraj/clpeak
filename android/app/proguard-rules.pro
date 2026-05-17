# clpeak ProGuard rules
# Keep JNI callback methods — called from C++ via GetMethodID.

-keep class kr.clpeak.BenchmarkRepository {
    void record_metric_callback_from_c(...);
    void device_info_callback_from_c(...);
}

# Keep native methods
-keepclasseswithmembernames class kr.clpeak.BenchmarkRepository {
    native <methods>;
}

# Keep data classes used via reflection/channels
-keep class kr.clpeak.ResultEntry { *; }
-keep class kr.clpeak.DeviceInfo { *; }
-keep class kr.clpeak.BenchmarkCategory { *; }
