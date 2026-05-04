package kr.clpeak

// Visual category used for tonal coloring + section grouping in the UI.
// Maps 1:1 with the C++ `Category` enum (fp_compute, int_compute,
// bandwidth, latency) plus an UNKNOWN fallback for unrecognised values.
enum class TestType { FP_COMPUTE, INT_COMPUTE, BANDWIDTH, LATENCY, UNKNOWN }

fun testTypeFromCategory(category: String): TestType = when (category) {
    "fp_compute"  -> TestType.FP_COMPUTE
    "int_compute" -> TestType.INT_COMPUTE
    "bandwidth"   -> TestType.BANDWIDTH
    "latency"     -> TestType.LATENCY
    else          -> TestType.UNKNOWN
}

data class BenchmarkCategory(
    val testName: String,
    val displayName: String,
    val category: String,           // canonical category string from C++
    val unit: String,
    val testType: TestType,
    val metrics: List<ResultEntry>,
    val isExpanded: Boolean = false
) {
    val peakValue: Float get() = metrics.maxOfOrNull { it.value } ?: 0f
    val backend: String get() = metrics.firstOrNull()?.backend ?: ""
}
