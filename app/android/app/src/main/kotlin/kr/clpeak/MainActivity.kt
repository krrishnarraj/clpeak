package kr.clpeak

import android.Manifest
import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Handler
import android.os.Looper
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

// The platform half of lib/src/services/process_exit.dart: the run in flight
// is stamped on this process, and Android's record of how an earlier process
// ended is read back with that stamp on it.  And of lib/src/services/
// run_notice.dart: the notification a run posts when the app leaves the
// screen mid-run.
class MainActivity : FlutterActivity() {
    // The Dart call waiting on the permission dialog's answer.
    private var pendingAsk: MethodChannel.Result? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "kr.clpeak/notify")
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "ask" -> askToNotify(result)
                    "show" -> {
                        showRunNotice(
                            call.argument<String>("title").orEmpty(),
                            call.argument<String>("body").orEmpty())
                        result.success(null)
                    }
                    "clear" -> {
                        getSystemService(NotificationManager::class.java).cancel(RUN_NOTICE)
                        result.success(null)
                    }
                    else -> result.notImplemented()
                }
            }
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

    // POST_NOTIFICATIONS is a runtime permission from Android 13, which is
    // the app's minSdk: without it nothing posts, silently.
    private fun askToNotify(result: MethodChannel.Result) {
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) ==
            PackageManager.PERMISSION_GRANTED) {
            result.success(true)
            return
        }
        if (pendingAsk != null) {
            result.success(false)
            return
        }
        pendingAsk = result
        requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), ASK_TO_NOTIFY)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != ASK_TO_NOTIFY) return
        pendingAsk?.success(grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED)
        pendingAsk = null
    }

    // Quiet: it is posted as the app leaves, to someone who left on purpose.
    // A tap opens the task as the launcher would -- the run's, while the
    // process lives; a fresh launch onto the Continue offer once it is gone,
    // since Android keeps an app's notifications past its process.
    private fun showRunNotice(title: String, body: String) {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(RUN_CHANNEL) == null) {
            nm.createNotificationChannel(
                NotificationChannel(RUN_CHANNEL, "Run status", NotificationManager.IMPORTANCE_LOW)
                    .apply {
                        description = "A run waiting for clpeak to be back on screen"
                        setShowBadge(false)
                    })
        }
        val open = Intent(this, MainActivity::class.java).apply {
            action = Intent.ACTION_MAIN
            addCategory(Intent.CATEGORY_LAUNCHER)
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val tap = PendingIntent.getActivity(
            this, 0, open, PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        val notice = Notification.Builder(this, RUN_CHANNEL)
            .setSmallIcon(R.drawable.ic_stat_clpeak)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(Notification.BigTextStyle().bigText(body))
            .setContentIntent(tap)
            .setAutoCancel(true)
            .setOnlyAlertOnce(true)
            .setCategory(Notification.CATEGORY_STATUS)
            .build()
        nm.notify(RUN_NOTICE, notice)
    }

    private companion object {
        const val RUN_CHANNEL = "run_status"
        const val RUN_NOTICE = 1
        const val ASK_TO_NOTIFY = 1
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
