package kr.clpeak

import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ReceiveChannel

/**
 * Owns the JNI declarations and bridges C++ callbacks to Kotlin coroutine channels.
 * Create a fresh instance for each benchmark run — channels are single-use.
 */
class BenchmarkRepository {

    private val _metricChannel     = Channel<ResultEntry>(Channel.UNLIMITED)
    private val _deviceInfoChannel = Channel<DeviceInfo>(Channel.UNLIMITED)

    val metricChannel:     ReceiveChannel<ResultEntry> get() = _metricChannel
    val deviceInfoChannel: ReceiveChannel<DeviceInfo>   get() = _deviceInfoChannel

    // ---- JNI declarations --------------------------------------------------

    private external fun launchClpeak(argc: Int, argv: Array<String>): Int

    /** Non-destructive enumeration. Safe to call from any thread. */
    external fun nativeEnumerateBackends(): String

    // Called from C++ logger_android.cpp on every metric (Ok or skipped).
    @Suppress("unused")
    fun record_metric_callback_from_c(
        backend: String,
        platform: String,
        device: String,
        driver: String,
        category: String,
        test: String,
        metric: String,
        unit: String,
        value: Float,
        status: String,
        reason: String
    ) {
        _metricChannel.trySend(
            ResultEntry(backend, platform, device, driver, category,
                        test, metric, unit, value, status, reason)
        )
    }

    // Called from C++ once per device before its metrics arrive.
    @Suppress("unused")
    fun device_info_callback_from_c(
        backend: String,
        platform: String,
        device: String,
        driver: String,
        propsJson: String,
        platformIndex: String,
        deviceIndex: String
    ) {
        _deviceInfoChannel.trySend(
            DeviceInfo(backend, platform, device, driver, propsJson,
                       platformIndex.toIntOrNull() ?: -1,
                       deviceIndex.toIntOrNull() ?: -1)
        )
    }

    // ---- Run ---------------------------------------------------------------

    /**
     * Blocking call — runs the benchmark suite via JNI, then closes both
     * channels so that collector coroutines terminate cleanly.
     * Must be called on a background dispatcher (e.g. Dispatchers.IO).
     */
    fun runBenchmark(argv: Array<String>): Int {
        return try {
            launchClpeak(argv.size, argv)
        } finally {
            _metricChannel.close()
            _deviceInfoChannel.close()
        }
    }

    // Library is loaded by MainActivity on app start.
}
