package kr.clpeak

// Mirrors the C++ ResultEntry produced by logger_android.cpp::emit().
// `category` is the canonical lower-snake string ("fp_compute", "int_compute",
// "bandwidth", "latency"); empty string for unknown / shim-derived rows.
data class ResultEntry(
    val backend: String,
    val platform: String,
    val device: String,
    val driver: String,
    val category: String,
    val test: String,
    val metric: String,
    val unit: String,
    val value: Float
)
