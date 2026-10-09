package kr.clpeak

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.os.Handler
import android.os.Looper
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

// The platform half of lib/src/services/process_exit.dart: the run in flight
// is stamped on this process, and Android's record of how an earlier process
// ended is read back with that stamp on it.
class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "kr.clpeak/process")
            .setMethodCallHandler { call, result ->
                val am = getSystemService(ActivityManager::class.java)
                when (call.method) {
                    "markRun" -> {
                        // At most 128 bytes, handed back on this process's
                        // exit record; null clears it.
                        val run = call.arguments as String?
                        am.setProcessStateSummary(run?.toByteArray(Charsets.UTF_8))
                        result.success(null)
                    }
                    "exits" -> {
                        // A tombstone is a file read: off the platform
                        // thread, with the answer posted back to it.
                        val main = Handler(Looper.getMainLooper())
                        Thread {
                            val exits = runCatching { exitsWithRun(am) }
                            main.post {
                                exits.fold(
                                    { result.success(it) },
                                    { result.error("exits", it.message, null) })
                            }
                        }.start()
                    }
                    else -> result.notImplemented()
                }
            }
    }
}

// The app's recent process deaths that had a run in flight.  The trace is
// read only where Android keeps one: a native crash's tombstone (protobuf)
// and an ANR's thread dump (text).
private fun exitsWithRun(am: ActivityManager): List<Map<String, Any?>> =
    am.getHistoricalProcessExitReasons(null, 0, 0).mapNotNull { info ->
        val run = info.processStateSummary?.toString(Charsets.UTF_8).orEmpty()
        if (run.isEmpty()) return@mapNotNull null
        val trace = when (info.reason) {
            ApplicationExitInfo.REASON_CRASH_NATIVE, ApplicationExitInfo.REASON_ANR ->
                runCatching { info.traceInputStream?.use { it.readNBytes(4 shl 20) } }.getOrNull()
            else -> null
        }
        mapOf(
            "run" to run,
            "timestamp_ms" to info.timestamp,
            "reason" to reasonName(info.reason),
            "status" to info.status,
            "description" to info.description,
            "importance" to info.importance,
            "pss_kb" to info.pss,
            "rss_kb" to info.rss,
            "trace" to trace,
        )
    }

private fun reasonName(reason: Int): String = when (reason) {
    ApplicationExitInfo.REASON_EXIT_SELF -> "EXIT_SELF"
    ApplicationExitInfo.REASON_SIGNALED -> "SIGNALED"
    ApplicationExitInfo.REASON_LOW_MEMORY -> "LOW_MEMORY"
    ApplicationExitInfo.REASON_CRASH -> "CRASH"
    ApplicationExitInfo.REASON_CRASH_NATIVE -> "CRASH_NATIVE"
    ApplicationExitInfo.REASON_ANR -> "ANR"
    ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "INITIALIZATION_FAILURE"
    ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "PERMISSION_CHANGE"
    ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "EXCESSIVE_RESOURCE_USAGE"
    ApplicationExitInfo.REASON_USER_REQUESTED -> "USER_REQUESTED"
    ApplicationExitInfo.REASON_USER_STOPPED -> "USER_STOPPED"
    ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "DEPENDENCY_DIED"
    ApplicationExitInfo.REASON_OTHER -> "OTHER"
    ApplicationExitInfo.REASON_FREEZER -> "FREEZER"
    ApplicationExitInfo.REASON_PACKAGE_STATE_CHANGE -> "PACKAGE_STATE_CHANGE"
    ApplicationExitInfo.REASON_PACKAGE_UPDATED -> "PACKAGE_UPDATED"
    else -> "UNKNOWN"
}
