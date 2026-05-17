package kr.clpeak

// Mirrors the C++ ResultEntry produced by logger_android.cpp.
// status is "ok" | "unsupported" | "skipped" | "error".
data class ResultEntry(
    val backend: String,
    val platform: String,
    val device: String,
    val driver: String,
    val category: String,
    val test: String,
    val display: String,
    val metric: String,
    val unit: String,
    val value: Float,
    val status: String = "ok",
    val reason: String = ""
)
