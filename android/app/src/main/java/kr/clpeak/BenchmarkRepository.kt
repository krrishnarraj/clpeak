package kr.clpeak

import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.ReceiveChannel

/**
 * Owns the JNI declarations and bridges C++ callbacks to Kotlin coroutine channels.
 * Create a fresh instance for each benchmark run — channels are single-use.
 */
class BenchmarkRepository {

    // Unlimited capacity so trySend never drops items regardless of how fast
    // the native thread produces metrics relative to the collector coroutine.
    private val _logChannel    = Channel<String>(Channel.UNLIMITED)
    private val _metricChannel = Channel<ResultEntry>(Channel.UNLIMITED)

    val logChannel:    ReceiveChannel<String>      get() = _logChannel
    val metricChannel: ReceiveChannel<ResultEntry> get() = _metricChannel

    // ---- JNI declarations --------------------------------------------------

    private external fun launchClpeak(argc: Int, argv: Array<String>): Int

    /** Non-destructive enumeration. Safe to call from any thread. */
    external fun nativeEnumerateBackends(): String

    // Called from C++ logger_android.cpp::print() on the benchmark thread.
    @Suppress("unused")
    fun print_callback_from_c(str: String) {
        _logChannel.trySend(str)
    }

    // Called from C++ logger_android.cpp::emit() on the benchmark thread.
    // Signature mirrors the v2 ResultEntry layout (backend, platform, device,
    // driver, category, test, metric, unit, value) -- category is new in v2
    // and replaces the static CATEGORY_META lookup the ViewModel used to do.
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
        value: Float
    ) {
        _metricChannel.trySend(
            ResultEntry(backend, platform, device, driver, category, test, metric, unit, value)
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
            _logChannel.close()
            _metricChannel.close()
        }
    }

    // Library is loaded by MainActivity on app start.
}
